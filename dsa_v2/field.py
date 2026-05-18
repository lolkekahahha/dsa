import cv2
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from scipy.spatial import Delaunay, KDTree

from .config import PipelineConfig
from .features import gradient_magnitude
from .motion import MotionDecision, evaluate_affine, fit_affine


def warp_image(image: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    h, w = image.shape
    y, x = np.mgrid[0:h, 0:w]
    return map_coordinates(image.astype(np.float64), [(y - dy).ravel(), (x - dx).ravel()], order=1, mode="nearest").reshape(h, w)


def smooth_field(dx: np.ndarray, dy: np.ndarray, cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    if cfg.field_smoothing_sigma <= 0:
        return dx, dy
    dx = gaussian_filter(dx, sigma=cfg.field_smoothing_sigma)
    dy = gaussian_filter(dy, sigma=cfg.field_smoothing_sigma)
    mag = np.sqrt(dx * dx + dy * dy)
    scale = np.minimum(1.0, cfg.max_displacement / (mag + 1e-8))
    return dx * scale, dy * scale


def build_affine_field(coeffs: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    grid = np.column_stack([x.ravel(), y.ravel()])
    values = evaluate_affine(grid, coeffs)
    return values[:, 0].reshape(h, w), values[:, 1].reshape(h, w)


def _residual_control_points(points: np.ndarray, displacements: np.ndarray, confidences: np.ndarray, shape):
    h, w = shape
    coeffs = fit_affine(points, displacements, confidences)
    affine_at_points = evaluate_affine(points, coeffs)
    residual = displacements - affine_at_points
    margin = min(20.0, max(1.0, 0.05 * min(h, w)))
    anchors = np.array(
        [
            [margin, margin],
            [w - margin, margin],
            [w - margin, h - margin],
            [margin, h - margin],
        ],
        dtype=np.float64,
    )
    all_pts = np.vstack([points, anchors])
    all_disp = np.vstack([residual, np.zeros_like(anchors)])
    rounded = np.round(all_pts, 3)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return coeffs, all_pts[unique_idx], all_disp[unique_idx]


def build_delaunay_field(points: np.ndarray, displacements: np.ndarray, confidences: np.ndarray, shape, cfg: PipelineConfig):
    h, w = shape
    if len(points) == 0:
        return np.zeros((h, w), dtype=np.float64), np.zeros((h, w), dtype=np.float64)

    coeffs, all_pts, all_disp = _residual_control_points(points, displacements, confidences, shape)
    y, x = np.mgrid[0:h, 0:w]
    grid = np.column_stack([x.ravel(), y.ravel()])

    try:
        tri = Delaunay(all_pts)
    except Exception:
        residual = displacements - evaluate_affine(points, coeffs)
        residual_dx = griddata(points, residual[:, 0], grid, method="linear", fill_value=0.0).reshape(h, w)
        residual_dy = griddata(points, residual[:, 1], grid, method="linear", fill_value=0.0).reshape(h, w)
        affine = evaluate_affine(grid, coeffs)
        return smooth_field(residual_dx + affine[:, 0].reshape(h, w), residual_dy + affine[:, 1].reshape(h, w), cfg)

    simplex_idx = tri.find_simplex(grid)
    dx = np.zeros((h, w), dtype=np.float64)
    dy = np.zeros((h, w), dtype=np.float64)
    transforms = tri.transform
    for s in range(len(tri.simplices)):
        inside = simplex_idx == s
        if not np.any(inside):
            continue
        pts = grid[inside]
        bary2 = np.einsum("ij,kj->ki", transforms[s, :2], pts - transforms[s, 2])
        bary = np.column_stack([bary2, 1.0 - bary2.sum(axis=1)])
        interp = bary @ all_disp[tri.simplices[s]]
        rows = y.ravel()[inside]
        cols = x.ravel()[inside]
        dx[rows, cols] = interp[:, 0]
        dy[rows, cols] = interp[:, 1]

    outside = simplex_idx < 0
    if np.any(outside):
        inside = ~outside
        tree = KDTree(grid[inside])
        dists, idx = tree.query(grid[outside], k=min(5, np.sum(inside)))
        if idx.ndim == 1:
            idx = idx[:, None]
            dists = dists[:, None]
        inside_dx = dx[y.ravel()[inside], x.ravel()[inside]]
        inside_dy = dy[y.ravel()[inside], x.ravel()[inside]]
        wx = np.take(inside_dx, idx.ravel()).reshape(idx.shape)
        wy = np.take(inside_dy, idx.ravel()).reshape(idx.shape)
        weights = 1.0 / (dists + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)
        rows = y.ravel()[outside]
        cols = x.ravel()[outside]
        dx[rows, cols] = np.sum(weights * wx, axis=1)
        dy[rows, cols] = np.sum(weights * wy, axis=1)

    affine = evaluate_affine(grid, coeffs)
    return smooth_field(dx + affine[:, 0].reshape(h, w), dy + affine[:, 1].reshape(h, w), cfg)


def build_model_field(points: np.ndarray, displacements: np.ndarray, confidences: np.ndarray, decision: MotionDecision, shape, cfg: PipelineConfig):
    h, w = shape
    if not decision.apply:
        return np.zeros((h, w), dtype=np.float64), np.zeros((h, w), dtype=np.float64)
    if decision.model == "rigid":
        return np.full((h, w), decision.translation[0], dtype=np.float64), np.full((h, w), decision.translation[1], dtype=np.float64)
    if decision.model == "affine":
        return build_affine_field(decision.affine_coeffs, shape)
    return build_delaunay_field(points, displacements, confidences, shape, cfg)


def apply_soft_roi(dx: np.ndarray, dy: np.ndarray, roi: np.ndarray, cfg: PipelineConfig):
    if roi is None or not np.any(roi):
        return np.zeros_like(dx), np.zeros_like(dy)
    feather = max(3.0, min(cfg.roi_margin / 3.0, cfg.template_size / 6.0))
    weight = np.clip(distance_transform_edt(roi.astype(bool)) / feather, 0.0, 1.0)
    weight = gaussian_filter(weight, sigma=max(1.0, feather / 4.0))
    return dx * weight, dy * weight


def confidence_map(
    shape,
    points: np.ndarray,
    confidences: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    roi: np.ndarray | None,
    cfg: PipelineConfig,
):
    h, w = shape
    if len(points) == 0:
        return np.zeros((h, w), dtype=np.float64)
    y, x = np.mgrid[0:h, 0:w]
    grid = np.column_stack([x.ravel(), y.ravel()])
    tree = KDTree(points)
    k = min(3, len(points))
    dists, idx = tree.query(grid, k=k)
    if k == 1:
        dists, idx = dists[:, None], idx[:, None]
    cvals = np.take(confidences, idx.ravel()).reshape(idx.shape)
    weights = 1.0 / (dists + 1e-8)
    weights = weights / weights.sum(axis=1, keepdims=True)
    conf = np.sum(weights * cvals, axis=1).reshape(h, w)
    nearest_distance = dists[:, 0].reshape(h, w)
    support_radius = cfg.field_confidence_support_radius
    if support_radius <= 0:
        support_radius = max(float(cfg.template_size), 2.0 * float(cfg.control_point_spacing))
    distance_support = np.exp(-((nearest_distance / (support_radius + 1e-8)) ** 2))
    smooth_penalty = np.exp(-np.sqrt(gradient_magnitude(dx, 1.0) ** 2 + gradient_magnitude(dy, 1.0) ** 2) / 5.0)
    conf = np.clip(conf * distance_support * smooth_penalty, 0.0, 1.0)
    if roi is not None:
        conf = np.where(roi, conf, 0.0)
    return conf


def apply_confidence_gate(dx: np.ndarray, dy: np.ndarray, conf: np.ndarray, cfg: PipelineConfig, micro_motion: bool, nonrigid: bool = False):
    floor = 0.15 if micro_motion else cfg.field_confidence_floor
    full = 0.55 if micro_motion else cfg.field_confidence_full
    weight = np.clip((conf - floor) / (full - floor + 1e-8), 0.0, 1.0)
    gamma = float(np.clip(cfg.field_confidence_gate_gamma, 0.25, 1.0))
    weight = np.power(weight, gamma)
    support = conf >= max(1e-3, 0.5 * floor)
    min_weight = cfg.nonrigid_field_confidence_min_weight if nonrigid else cfg.field_confidence_min_weight
    weight = np.where(support, np.maximum(weight, float(min_weight)), weight)
    weight = gaussian_filter(weight, sigma=max(1.0, cfg.field_smoothing_sigma / 2.0))
    weight = np.where(support, np.maximum(weight, float(min_weight)), weight)
    return dx * weight, dy * weight, np.clip(weight, 0.0, 1.0)


def protect_vessel_field(
    dx: np.ndarray,
    dy: np.ndarray,
    vessels: np.ndarray,
    bed: np.ndarray | None,
    cfg: PipelineConfig,
    confidence: np.ndarray | None = None,
):
    if not cfg.protect_vessels_in_field:
        return dx, dy
    vessel_region = vessels.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_region = cv2.dilate(vessel_region.astype(np.uint8), kernel, iterations=1).astype(bool)
    vessel_protection = gaussian_filter(vessel_region.astype(np.float64), sigma=max(1.0, cfg.vessel_field_protection_sigma))
    vessel_protection = vessel_protection / (np.max(vessel_protection) + 1e-8)
    if confidence is not None:
        confident_surroundings = np.clip(confidence.astype(np.float64), 0.0, 1.0) * (~vessel_region).astype(np.float64)
        relief = np.clip(cfg.vessel_protection_confidence_relief * confident_surroundings, 0.0, 1.0)
        vessel_protection = np.clip(vessel_protection * (1.0 - relief), 0.0, 1.0)

    if cfg.vessel_field_strategy == "propagate":
        support = (~vessel_region).astype(np.float64)
        if confidence is not None:
            support *= np.clip(confidence.astype(np.float64), 0.0, 1.0)
        if np.sum(support > 1e-4) < 64:
            support = (~vessel_region).astype(np.float64)
        sigma = max(2.0, 2.5 * cfg.vessel_field_protection_sigma)
        norm = gaussian_filter(support, sigma=sigma)
        propagated_dx = gaussian_filter(dx * support, sigma=sigma) / (norm + 1e-8)
        propagated_dy = gaussian_filter(dy * support, sigma=sigma) / (norm + 1e-8)
        propagated_dx = np.where(norm > 1e-4, propagated_dx, dx)
        propagated_dy = np.where(norm > 1e-4, propagated_dy, dy)
        blend = np.clip(cfg.vessel_field_propagation_strength * vessel_protection, 0.0, 1.0)
        dx = dx * (1.0 - blend) + propagated_dx * blend
        dy = dy * (1.0 - blend) + propagated_dy * blend

        bed_protection = np.zeros_like(dx)
        if bed is not None and np.any(bed):
            bed_region = bed.astype(bool) & ~vessel_region
            bed_protection = gaussian_filter(
                bed_region.astype(np.float64),
                sigma=max(2.0, 2.0 * cfg.vessel_field_protection_sigma),
            )
            bed_protection = bed_protection / (np.max(bed_protection) + 1e-8)
        bed_weight = 1.0 - (1.0 - cfg.vascular_bed_motion_attenuation) * bed_protection
        return dx * bed_weight, dy * bed_weight

    bed_protection = np.zeros_like(dx)
    if bed is not None and np.any(bed):
        bed_region = bed.astype(bool) & ~vessel_region
        bed_protection = gaussian_filter(bed_region.astype(np.float64), sigma=max(2.0, 2.0 * cfg.vessel_field_protection_sigma))
        bed_protection = bed_protection / (np.max(bed_protection) + 1e-8)
    bed_weight = 1.0 - (1.0 - cfg.vascular_bed_motion_attenuation) * bed_protection
    vessel_strength = float(np.clip(cfg.vessel_field_protection_strength, 0.0, 1.0))
    vessel_weight = 1.0 - vessel_strength * vessel_protection
    motion_weight = np.clip(vessel_weight * bed_weight, 0.0, 1.0)
    return dx * motion_weight, dy * motion_weight


def local_improvement_gate(mask, live, dx, dy, roi, vessels, cfg: PipelineConfig, confidence: np.ndarray | None = None):
    if not cfg.enable_local_improvement_gate:
        return dx, dy, np.ones_like(dx)
    safe = roi.astype(bool) & ~vessels.astype(bool)
    if not np.any(safe):
        return dx, dy, np.zeros_like(dx)
    warped = warp_image(mask, dx, dy)
    before = np.abs(live.astype(np.float64) - mask.astype(np.float64))
    after = np.abs(live.astype(np.float64) - warped.astype(np.float64))
    weights = safe.astype(np.float64)
    sigma = max(2.0, cfg.template_size / 6.0)
    norm = gaussian_filter(weights, sigma=sigma) + 1e-8
    improvement = gaussian_filter((before - after) * weights, sigma=sigma) / norm
    values = improvement[safe]
    scale = max(0.5, float(np.percentile(np.abs(values), 75)) if values.size else 0.5)
    gate = np.clip((improvement + cfg.local_improvement_tolerance) / (scale + cfg.local_improvement_tolerance + 1e-8), 0.0, 1.0)
    preserve = None
    if confidence is not None:
        preserve = cfg.local_improvement_confidence_preserve * np.clip(confidence.astype(np.float64), 0.0, 1.0)
        preserve *= safe.astype(np.float64)
        gate = np.maximum(gate, preserve)
    gate = gaussian_filter(gate, sigma=max(1.0, sigma / 2.0))
    if preserve is not None:
        gate = np.maximum(gate, preserve)
    gate *= roi.astype(np.float64)
    return dx * gate, dy * gate, np.clip(gate, 0.0, 1.0)
