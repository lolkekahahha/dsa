import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from .config import PipelineConfig


def gradient_magnitude(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    image = image.astype(np.float64)
    gx = gaussian_filter(image, sigma=sigma, order=(1, 0))
    gy = gaussian_filter(image, sigma=sigma, order=(0, 1))
    return np.sqrt(gx * gx + gy * gy)


def plain_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(image.astype(np.float64))
    return np.sqrt(gx * gx + gy * gy)


def harris_response(image: np.ndarray) -> np.ndarray:
    norm = (image - np.min(image)) / (np.ptp(image) + 1e-8)
    response = cv2.cornerHarris(np.float32(norm), blockSize=5, ksize=3, k=0.12)
    response = np.maximum(response, 0.0)
    return response / (np.max(response) + 1e-8)


def tissue_importance(image: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    intensity = (image - np.min(image)) / (np.ptp(image) + 1e-8)
    grad = gradient_magnitude(image, cfg.gaussian_sigma)
    grad = grad / (np.max(grad) + 1e-8)
    return np.power(0.5 * intensity + 0.5 * grad, 0.7)


def point_reliability(
    image: np.ndarray,
    point: tuple[int, int],
    vessel_mask: np.ndarray,
    importance: np.ndarray,
    cfg: PipelineConfig,
    vascular_bed_mask: np.ndarray | None = None,
) -> float:
    x, y = point
    half = cfg.template_size // 2
    y0, y1 = max(0, y - half), min(image.shape[0], y + half + 1)
    x0, x1 = max(0, x - half), min(image.shape[1], x + half + 1)
    local = image[y0:y1, x0:x1]
    if local.size == 0:
        return 0.0

    grad = gradient_magnitude(local, cfg.gaussian_sigma)
    texture = min(float(np.std(local) / 35.0), 1.0)
    edge_strength = min(float(np.percentile(grad, 90) / (np.percentile(grad, 50) + 1e-8)), 2.0) / 2.0
    vessel_penalty = 1.0 - float(np.mean(vessel_mask[y0:y1, x0:x1]))
    bed_penalty = 1.0
    if vascular_bed_mask is not None:
        bed_penalty = 1.0 - 0.35 * float(np.mean(vascular_bed_mask[y0:y1, x0:x1]))
    importance_score = float(np.mean(importance[y0:y1, x0:x1]))
    return float(np.clip(0.35 * texture + 0.35 * edge_strength + 0.30 * importance_score, 0, 1) * vessel_penalty * bed_penalty)


def select_control_points(
    image: np.ndarray,
    vessel_mask: np.ndarray,
    importance: np.ndarray,
    cfg: PipelineConfig,
    roi_mask: np.ndarray,
    vascular_bed_mask: np.ndarray | None = None,
    safe_mask: np.ndarray | None = None,
    registration_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape
    margin = cfg.template_size // 2 + cfg.search_radius + 1
    safe_region = roi_mask.astype(bool) & ~vessel_mask.astype(bool)
    if safe_mask is not None and np.any(safe_mask):
        safe_region &= safe_mask.astype(bool)
    if registration_weight is not None:
        safe_region &= registration_weight >= cfg.registration_weight_min_for_points

    grad = gradient_magnitude(image, cfg.gaussian_sigma)
    grad = grad / (np.max(grad) + 1e-8)
    harris = harris_response(image)
    score = (0.60 * grad + 0.40 * harris) * importance * safe_region
    if registration_weight is not None:
        score *= np.clip(registration_weight, 0.0, 1.0)
    if vascular_bed_mask is not None:
        bed_soft = gaussian_filter(vascular_bed_mask.astype(np.float64), sigma=max(2.0, cfg.template_size / 10.0))
        bed_soft = bed_soft / (np.max(bed_soft) + 1e-8)
        score *= 1.0 - float(cfg.vascular_bed_point_penalty) * bed_soft

    valid_area = np.zeros_like(safe_region, dtype=bool)
    valid_area[margin : h - margin, margin : w - margin] = True
    positive = score[valid_area & safe_region & (score > 0)]
    threshold = cfg.gradient_threshold if positive.size == 0 else max(0.03, min(cfg.gradient_threshold, np.percentile(positive, 85)))
    peak_window = max(5, cfg.control_point_spacing // 2)
    candidate_mask = (score == maximum_filter(score, size=peak_window, mode="nearest")) & valid_area & (score > threshold)
    ys, xs = np.where(candidate_mask)

    points: list[list[int]] = []
    reliabilities: list[float] = []
    if len(xs):
        scores = score[ys, xs]
        order = np.argsort(scores)[::-1]
        min_dist = max(10, int(0.65 * cfg.control_point_spacing))
        cell_size = max(cfg.template_size, 2 * cfg.control_point_spacing)
        cell_counts: dict[tuple[int, int], int] = {}
        for idx in order:
            x, y = int(xs[idx]), int(ys[idx])
            cell = (y // cell_size, x // cell_size)
            if cell_counts.get(cell, 0) >= cfg.max_points_per_cell:
                continue
            if points:
                d = np.sqrt(np.sum((np.asarray(points) - np.array([x, y])) ** 2, axis=1))
                if np.any(d < min_dist):
                    continue
            rel = point_reliability(image, (x, y), vessel_mask, importance, cfg, vascular_bed_mask)
            if registration_weight is not None:
                rel *= float(registration_weight[y, x])
            points.append([x, y])
            reliabilities.append(rel)
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
            if len(points) >= cfg.max_control_points:
                break

    if len(points) < cfg.min_control_points:
        y_grid, x_grid = np.mgrid[margin : h - margin : cfg.control_point_spacing, margin : w - margin : cfg.control_point_spacing]
        for y, x in zip(y_grid.ravel(), x_grid.ravel()):
            if safe_region[int(y), int(x)]:
                points.append([int(x), int(y)])
                rel = 0.25
                if registration_weight is not None:
                    rel *= float(registration_weight[int(y), int(x)])
                reliabilities.append(rel)
            if len(points) >= cfg.min_control_points:
                break

    grid_n = max(2, int(cfg.coverage_grid))
    coverage_counts: dict[tuple[int, int], int] = {}
    for p in points:
        cell = (min(grid_n - 1, int(p[1] / max(1, h) * grid_n)), min(grid_n - 1, int(p[0] / max(1, w) * grid_n)))
        coverage_counts[cell] = coverage_counts.get(cell, 0) + 1
    target_cells = []
    for gy in range(grid_n):
        for gx in range(grid_n):
            y0, y1 = int(gy * h / grid_n), int((gy + 1) * h / grid_n)
            x0, x1 = int(gx * w / grid_n), int((gx + 1) * w / grid_n)
            cell_safe = safe_region[y0:y1, x0:x1] & valid_area[y0:y1, x0:x1]
            if np.any(cell_safe):
                target_cells.append((gy, gx, y0, y1, x0, x1))

    coverage_quota = max(1, int(cfg.coverage_points_per_cell))
    coverage_min_dist = max(6, int(0.30 * cfg.control_point_spacing))
    for gy, gx, y0, y1, x0, x1 in target_cells:
        local_score = score[y0:y1, x0:x1].copy()
        local_valid = safe_region[y0:y1, x0:x1] & valid_area[y0:y1, x0:x1]
        local_score[~local_valid] = 0.0
        while coverage_counts.get((gy, gx), 0) < coverage_quota and len(points) < cfg.max_control_points:
            if np.max(local_score) <= 0:
                break
            yy, xx = np.unravel_index(int(np.argmax(local_score)), local_score.shape)
            x, y = int(x0 + xx), int(y0 + yy)
            if points:
                d = np.sqrt(np.sum((np.asarray(points) - np.array([x, y])) ** 2, axis=1))
                if np.any(d < coverage_min_dist):
                    local_score[max(0, yy - coverage_min_dist) : yy + coverage_min_dist + 1, max(0, xx - coverage_min_dist) : xx + coverage_min_dist + 1] = 0.0
                    continue
            points.append([x, y])
            rel = point_reliability(image, (x, y), vessel_mask, importance, cfg, vascular_bed_mask)
            if registration_weight is not None:
                rel *= float(registration_weight[y, x])
            reliabilities.append(rel)
            coverage_counts[(gy, gx)] = coverage_counts.get((gy, gx), 0) + 1
            local_score[max(0, yy - coverage_min_dist) : yy + coverage_min_dist + 1, max(0, xx - coverage_min_dist) : xx + coverage_min_dist + 1] = 0.0

    if cfg.enable_texture_control_points and len(points) < cfg.max_control_points:
        half = cfg.template_size // 2
        stride = max(8, int(cfg.texture_point_stride))
        candidates = []
        for y in range(margin, h - margin, stride):
            for x in range(margin, w - margin, stride):
                if not safe_region[y, x] or not valid_area[y, x]:
                    continue
                y0, y1 = y - half, y + half + 1
                x0, x1 = x - half, x + half + 1
                local_vessels = vessel_mask[y0:y1, x0:x1]
                vessel_fraction = float(np.mean(local_vessels))
                if vessel_fraction > cfg.texture_point_max_vessel_fraction:
                    continue
                patch = image[y0:y1, x0:x1]
                texture = float(np.std(patch))
                if texture < cfg.texture_point_min_std:
                    continue
                local_grad = float(np.percentile(grad[y0:y1, x0:x1], 90))
                bed_penalty = 1.0
                if vascular_bed_mask is not None:
                    bed_penalty = 1.0 - 0.25 * float(np.mean(vascular_bed_mask[y0:y1, x0:x1]))
                candidate_score = (0.65 * min(texture / 35.0, 1.0) + 0.35 * local_grad) * bed_penalty
                if registration_weight is not None:
                    candidate_score *= float(np.mean(registration_weight[y0:y1, x0:x1]))
                candidates.append((candidate_score, x, y))
        candidates.sort(reverse=True)
        min_dist = max(6, int(0.45 * cfg.control_point_spacing))
        for _, x, y in candidates:
            if points:
                d = np.sqrt(np.sum((np.asarray(points) - np.array([x, y])) ** 2, axis=1))
                if np.any(d < min_dist):
                    continue
            points.append([int(x), int(y)])
            rel = point_reliability(image, (int(x), int(y)), vessel_mask, importance, cfg, vascular_bed_mask)
            if registration_weight is not None:
                rel *= float(registration_weight[int(y), int(x)])
            reliabilities.append(rel)
            if len(points) >= cfg.max_control_points:
                break

    if not points:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

    pts = np.asarray(points, dtype=np.float64)
    rel = np.asarray(reliabilities, dtype=np.float64)
    reliability_floor = 0.08 if registration_weight is not None else 0.20
    keep = rel >= max(reliability_floor, np.percentile(rel, 15))
    if np.sum(keep) >= min(cfg.min_control_points, len(rel)):
        pts, rel = pts[keep], rel[keep]
    return pts[: cfg.max_control_points], rel[: cfg.max_control_points]
