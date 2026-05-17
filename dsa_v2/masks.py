from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, uniform_filter

from .config import PipelineConfig
from .features import gradient_magnitude


def uses_compact_frame_defaults(shape) -> bool:
    h, w = shape[:2]
    aspect = w / max(h, 1)
    return 0.85 <= aspect <= 1.15 and min(h, w) <= 700


def apply_roi_margin(roi: np.ndarray, margin: int) -> np.ndarray:
    h, w = roi.shape
    margin = min(int(margin), h // 8, w // 8)
    roi = roi.astype(bool).copy()
    if margin > 0:
        roi[:margin, :] = False
        roi[-margin:, :] = False
        roi[:, :margin] = False
        roi[:, -margin:] = False
    return roi


def compute_roi(mask: np.ndarray, live: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    h, w = mask.shape
    if cfg.roi_bbox is not None:
        x, y, width, height = cfg.roi_bbox
        roi = np.zeros((h, w), dtype=bool)
        x0, y0 = int(np.clip(x, 0, w)), int(np.clip(y, 0, h))
        x1, y1 = int(np.clip(x + width, 0, w)), int(np.clip(y + height, 0, h))
        roi[y0:y1, x0:x1] = True
        return apply_roi_margin(roi, cfg.roi_margin)

    compact_frame = uses_compact_frame_defaults(mask.shape)

    mask_norm = (mask - np.min(mask)) / (np.ptp(mask) + 1e-8)
    live_norm = (live - np.min(live)) / (np.ptp(live) + 1e-8)
    motion = np.abs(live_norm - mask_norm)
    anatomy = gradient_magnitude(mask_norm, cfg.gaussian_sigma)
    anatomy = anatomy / (np.max(anatomy) + 1e-8)
    score = gaussian_filter(0.65 * motion + 0.35 * anatomy, sigma=3.0)

    threshold = np.percentile(score, 70 if compact_frame else 55)
    min_area = int((0.12 if compact_frame else 0.25) * h * w)
    roi = score >= threshold
    kernel_size = max(9, (min(h, w) // 24) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    roi_u8 = cv2.morphologyEx(roi.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    roi_u8 = cv2.morphologyEx(roi_u8, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(roi_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros((h, w), dtype=np.uint8)
    if contours:
        total = 0.0
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            area = cv2.contourArea(cnt)
            if area < 0.01 * h * w:
                continue
            cv2.drawContours(cleaned, [cnt], -1, 1, -1)
            total += area
            if total >= min_area:
                break
    if cleaned.sum() < min_area:
        cleaned[:] = 1
        if compact_frame:
            cleaned[: int(0.08 * h), :] = 0
            cleaned[:, : int(0.04 * w)] = 0
    roi = binary_dilation(apply_roi_margin(cleaned.astype(bool), cfg.roi_margin), structure=np.ones((7, 7)), iterations=2)
    return apply_roi_margin(roi, cfg.roi_margin)


def vesselness(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    result = np.zeros_like(image, dtype=np.float64)
    beta = 0.5
    for sigma in (1.0, 2.0, 3.0):
        hxx = gaussian_filter(image, sigma=sigma, order=(0, 2)) * sigma**2
        hyy = gaussian_filter(image, sigma=sigma, order=(2, 0)) * sigma**2
        hxy = gaussian_filter(image, sigma=sigma, order=(1, 1)) * sigma**2
        trace = hxx + hyy
        root = np.sqrt((hxx - hyy) ** 2 + 4.0 * hxy**2)
        l1 = 0.5 * (trace - root)
        l2 = 0.5 * (trace + root)
        swap = np.abs(l1) > np.abs(l2)
        l1s = np.where(swap, l2, l1)
        l2s = np.where(swap, l1, l2)
        rb = np.abs(l1s) / (np.abs(l2s) + 1e-8)
        s = np.sqrt(l1s**2 + l2s**2)
        c = np.percentile(s, 90) + 1e-8
        response = np.exp(-(rb**2) / (2 * beta**2)) * (1.0 - np.exp(-(s**2) / (2 * c**2)))
        response[np.abs(l2s) < 1e-8] = 0.0
        result = np.maximum(result, response)
    return result / (np.max(result) + 1e-8)


def keep_connected_vascular_tree(vessels: np.ndarray, roi: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    vessels_u8 = vessels.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vessels_u8, connectivity=8)
    if n_labels <= 2:
        return vessels

    h, w = vessels.shape
    image_area = float(h * w)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    largest_area = float(np.max(areas)) if areas.size else 0.0
    if largest_area < 0.002 * image_area:
        return vessels

    largest_label = int(np.argmax(areas) + 1)
    keep = labels == largest_label
    proximity = max(35.0, float(cfg.vessel_tree_proximity_fraction) * min(h, w))
    min_area = max(18.0, 0.000018 * image_area)
    border_margin = max(8, int(cfg.roi_margin))
    distance_to_seed = cv2.distanceTransform((~keep).astype(np.uint8), cv2.DIST_L2, 3)

    for label in range(1, n_labels):
        if label == largest_label:
            continue
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, width, height = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        touches_border = x <= border_margin or y <= border_margin or x + width >= w - border_margin or y + height >= h - border_margin
        if touches_border and area < 0.20 * largest_area:
            continue
        component = labels == label
        if float(np.min(distance_to_seed[component])) <= proximity:
            keep |= component

    kept_fraction = float(np.mean(keep[roi.astype(bool)])) if np.any(roi) else float(np.mean(keep))
    original_fraction = float(np.mean(vessels[roi.astype(bool)])) if np.any(roi) else float(np.mean(vessels))
    if kept_fraction < 0.18 * original_fraction:
        return vessels
    return keep & roi.astype(bool)


def detect_vessels(live: np.ndarray, mask: np.ndarray, roi: Optional[np.ndarray], cfg: PipelineConfig) -> np.ndarray:
    dsa = live.astype(np.float64) - mask.astype(np.float64)
    smoothed = gaussian_filter(dsa, sigma=1.0)
    local_mean = gaussian_filter(smoothed, sigma=12.0)
    region = roi if roi is not None and np.any(roi) else np.ones_like(dsa, dtype=bool)

    def normalize_signal(signal: np.ndarray, percentile: float = 99.0) -> np.ndarray:
        scale = np.percentile(signal[region], percentile) + 1e-8
        return np.clip(signal / scale, 0.0, 1.0)

    residual = smoothed - local_mean
    live_smoothed = gaussian_filter(live.astype(np.float64), sigma=1.0)
    live_background = gaussian_filter(live_smoothed, sigma=14.0)
    live_residual = live_smoothed - live_background
    mask_smoothed = gaussian_filter(mask.astype(np.float64), sigma=1.0)
    mask_background = gaussian_filter(mask_smoothed, sigma=14.0)
    mask_residual = mask_smoothed - mask_background

    brightness = live_background
    bright_excess = (
        (brightness - np.percentile(brightness[region], 55))
        / (np.percentile(brightness[region], 92) - np.percentile(brightness[region], 55) + 1e-8)
    )
    bright_background_penalty = np.exp(-4.0 * np.clip(bright_excess, 0.0, 1.0) ** 2)
    bright_background_penalty = np.clip(bright_background_penalty, 0.06, 1.0)

    def polarity_score(sign: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        live_signal = normalize_signal(np.maximum(sign * live_residual, 0.0))
        mask_signal = normalize_signal(np.maximum(sign * mask_residual, 0.0))
        new_signal = normalize_signal(
            np.maximum(live_signal - cfg.vessel_mask_existing_structure_penalty * mask_signal, 0.0),
            percentile=98.0,
        )
        residual_signal = normalize_signal(np.maximum(sign * residual, 0.0))
        raw_signal = normalize_signal(np.maximum(sign * dsa, 0.0))
        tubular = vesselness(live_signal)
        signed_support = np.maximum(residual_signal, raw_signal)
        score = (
            0.34 * new_signal
            + 0.24 * signed_support
            + 0.20 * tubular
            + 0.14 * live_signal
            + 0.08 * raw_signal
        )
        if sign < 0:
            score *= bright_background_penalty
        return score, live_signal, residual_signal, tubular, new_signal

    dark_score, dark_signal, dark_residual, dark_tubular, dark_new = polarity_score(-1.0)
    bright_score, bright_signal, bright_residual, bright_tubular, bright_new = polarity_score(1.0)
    use_dark = np.percentile(dark_score[region], 98) >= 0.72 * np.percentile(bright_score[region], 98)
    score = dark_score if use_dark else bright_score
    vessel_signal = dark_signal if use_dark else bright_signal
    residual_signal = dark_residual if use_dark else bright_residual
    tubular = dark_tubular if use_dark else bright_tubular
    new_signal = dark_new if use_dark else bright_new

    score_threshold = np.percentile(
        score[region],
        cfg.vessel_mask_dark_score_percentile if use_dark else cfg.vessel_mask_bright_score_percentile,
    )
    strong_score_threshold = np.percentile(score[region], 96)
    signal_threshold = np.percentile(vessel_signal[region], 55 if use_dark else 68)
    new_threshold = np.percentile(
        new_signal[region],
        cfg.vessel_mask_dark_new_signal_percentile if use_dark else cfg.vessel_mask_bright_new_signal_percentile,
    )
    residual_threshold = np.percentile(residual_signal[region], 55 if use_dark else 65)
    tubular_threshold = np.percentile(tubular[region], 65 if use_dark else 72)
    vessel_candidates = (score >= score_threshold) & (
        ((new_signal >= new_threshold) & (vessel_signal >= signal_threshold) & (residual_signal >= residual_threshold))
        | ((score >= strong_score_threshold) & (new_signal >= new_threshold) & (tubular >= tubular_threshold))
    )
    vessel_candidates &= region

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(vessel_candidates.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel)
    opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, open_kernel)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(opened)
    image_area = mask.shape[0] * mask.shape[1]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 8 < area < 0.06 * image_area:
            cv2.drawContours(cleaned, [cnt], -1, 1, -1)
    cleaned = cleaned.astype(bool)
    if not uses_compact_frame_defaults(mask.shape) and float(np.mean(cleaned[region])) > 0.025:
        cleaned = keep_connected_vascular_tree(cleaned, region, cfg)
    return cleaned.astype(bool)


def normalize_region_signal(signal: np.ndarray, region: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    values = signal[region.astype(bool)] if np.any(region) else signal.ravel()
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, percentile))
    return np.clip((signal.astype(np.float64) - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def grayscale_blackhat_multiscale(image: np.ndarray, sizes: tuple[int, ...]) -> np.ndarray:
    image_f = image.astype(np.float32)
    responses = []
    for size in sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(size), int(size)))
        closed = cv2.morphologyEx(image_f, cv2.MORPH_CLOSE, kernel)
        responses.append(np.maximum(closed.astype(np.float64) - image.astype(np.float64), 0.0))
    if not responses:
        return np.zeros_like(image, dtype=np.float64)
    return np.max(np.stack(responses, axis=0), axis=0)


def cleanup_binary_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    cleaned = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= int(min_area):
            cleaned[labels == label] = 1
    return cleaned.astype(bool)


def hysteresis_grow(seed: np.ndarray, support: np.ndarray, iterations: int = 96) -> np.ndarray:
    grown = seed.astype(bool) & support.astype(bool)
    for _ in range(int(iterations)):
        next_grown = binary_dilation(grown, structure=np.ones((3, 3))) & support
        if np.array_equal(next_grown, grown):
            break
        grown = next_grown
    return grown


def prune_bright_clutter(
    support: np.ndarray,
    motion_vessels: np.ndarray,
    bright_background: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    if not np.any(support) or not cfg.enable_vessel_support_clutter_pruning:
        return support.astype(bool)
    protected = binary_dilation(
        motion_vessels.astype(bool),
        structure=np.ones((5, 5)),
        iterations=5,
    )
    window = max(9, int(cfg.vessel_support_clutter_window) | 1)
    local_density = uniform_filter(support.astype(np.float64), size=window, mode="nearest")
    clutter = (
        (local_density > cfg.vessel_support_clutter_density)
        & (bright_background > cfg.vessel_support_clutter_brightness)
        & ~protected
    )
    clutter = gaussian_filter(clutter.astype(np.float64), sigma=1.0) > 0.2
    pruned = support.astype(bool) & ~clutter
    return cleanup_binary_components(pruned, cfg.vessel_support_min_component_area) | motion_vessels.astype(bool)


def compute_vessel_support_region(shape: tuple[int, int], roi: Optional[np.ndarray], cfg: PipelineConfig) -> np.ndarray:
    h, w = shape
    if roi is not None and np.any(roi) and not uses_compact_frame_defaults(shape):
        region = binary_dilation(roi.astype(bool), structure=np.ones((5, 5)), iterations=3)
    else:
        region = np.ones((h, w), dtype=bool)
    return apply_roi_margin(region, cfg.roi_margin)


def detect_vessel_support_mask(
    live: np.ndarray,
    mask: np.ndarray,
    roi: Optional[np.ndarray],
    motion_vessels: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    h, w = live.shape
    region = compute_vessel_support_region(live.shape, roi, cfg)
    if not np.any(region):
        return np.zeros((h, w), dtype=bool)

    live_smooth = gaussian_filter(live.astype(np.float64), sigma=0.7)
    blackhat = grayscale_blackhat_multiscale(live_smooth, sizes=(7, 11, 17, 25, 35))
    blackhat_signal = normalize_region_signal(blackhat, region, percentile=99.5)

    contrast_gain = mask.astype(np.float64) - live.astype(np.float64)
    local_contrast_gain = contrast_gain - gaussian_filter(contrast_gain, sigma=20.0)
    dsa_signal = normalize_region_signal(np.maximum(local_contrast_gain, 0.0), region, percentile=99.0)
    raw_dsa_signal = normalize_region_signal(np.maximum(contrast_gain, 0.0), region, percentile=99.3)

    tubular = vesselness(np.maximum(blackhat_signal, dsa_signal))
    background = gaussian_filter(live.astype(np.float64), sigma=16.0)
    bright_background = normalize_region_signal(background, region, percentile=99.0)
    bright_texture_penalty = np.exp(-4.0 * np.clip((bright_background - 0.58) / 0.42, 0.0, 1.0) ** 2)

    score = (
        0.40 * dsa_signal
        + 0.28 * blackhat_signal
        + 0.24 * tubular
        + 0.08 * raw_dsa_signal
    )
    score = gaussian_filter(np.clip(score * bright_texture_penalty, 0.0, 1.0), sigma=0.45)
    score *= region.astype(np.float64)

    valid_score = score[region]
    if valid_score.size == 0:
        return np.zeros((h, w), dtype=bool)
    support_threshold = np.percentile(valid_score, cfg.vessel_support_score_percentile)
    seed_threshold = np.percentile(valid_score, cfg.vessel_support_seed_percentile)

    dark_background_evidence = (
        ((dsa_signal > 0.14) & (tubular > 0.10))
        | ((blackhat_signal > 0.22) & (tubular > 0.16))
        | motion_vessels.astype(bool)
    )
    bright_background_evidence = (
        (dsa_signal > cfg.vessel_support_bright_dsa_min)
        & (raw_dsa_signal > cfg.vessel_support_bright_raw_min)
        & (tubular > cfg.vessel_support_bright_tubular_min)
    ) | motion_vessels.astype(bool)
    evidence = np.where(
        bright_background > cfg.vessel_support_bright_background_threshold,
        bright_background_evidence,
        dark_background_evidence,
    )
    weak_support = (score >= support_threshold) & region & evidence
    motion_seed = motion_vessels.astype(bool) & region
    if np.any(motion_seed):
        dilated_motion = binary_dilation(motion_seed, structure=np.ones((3, 3)), iterations=1)
        distance_to_motion = cv2.distanceTransform((~dilated_motion).astype(np.uint8), cv2.DIST_L2, 3)
        near_motion = distance_to_motion <= max(24.0, 0.08 * min(h, w))
        seed = motion_seed | ((score >= seed_threshold) & weak_support & near_motion)
    else:
        seed = (score >= seed_threshold) & weak_support
    grown = hysteresis_grow(seed, weak_support)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grown = cv2.morphologyEx(grown.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel).astype(bool)
    grown = cleanup_binary_components(grown, cfg.vessel_support_min_component_area)
    if cfg.vessel_support_dilation > 0:
        grown = binary_dilation(
            grown,
            structure=np.ones((3, 3)),
            iterations=int(cfg.vessel_support_dilation),
        )
    grown = (grown | motion_vessels.astype(bool)) & region
    grown = prune_bright_clutter(grown, motion_vessels, bright_background, cfg)
    return grown & region


def detect_vascular_bed(live: np.ndarray, mask: np.ndarray, vessels: np.ndarray, roi: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    dsa = live.astype(np.float64) - mask.astype(np.float64)
    region = roi if np.any(roi) else np.ones_like(dsa, dtype=bool)
    dsa_abs = np.abs(dsa - gaussian_filter(dsa, sigma=12.0))
    dsa_norm = np.clip(dsa_abs / (np.percentile(dsa_abs[region], 99) + 1e-8), 0.0, 1.0)
    tubular = gaussian_filter(vesselness(dsa_norm), sigma=2.0)
    score = gaussian_filter(0.65 * dsa_norm + 0.35 * tubular, sigma=3.0)
    compact_frame = uses_compact_frame_defaults(mask.shape)
    bed = score >= np.percentile(score[region], 58 if compact_frame else 62)
    bed &= roi
    seed = binary_dilation(vessels, structure=np.ones((3, 3)), iterations=max(1, int(cfg.vascular_bed_dilation)))
    bed |= seed & roi
    kernel_size = max(11, (min(mask.shape) // 20) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bed_u8 = cv2.morphologyEx(bed.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    bed_u8 = cv2.morphologyEx(bed_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return bed_u8.astype(bool) & roi


def compute_registration_safe_mask(
    mask: np.ndarray,
    live: np.ndarray,
    roi: np.ndarray,
    vessels: np.ndarray,
    vascular_bed: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    region = roi.astype(bool)
    if not np.any(region):
        return np.zeros_like(roi, dtype=bool)

    vessel_exclusion = binary_dilation(
        vessels.astype(bool),
        structure=np.ones((3, 3)),
        iterations=max(1, int(cfg.safe_background_vessel_dilation)),
    )

    dsa_abs = np.abs(live.astype(np.float64) - mask.astype(np.float64))
    max_dsa = np.percentile(dsa_abs[region], cfg.safe_background_max_abs_dsa_percentile)
    brightness = np.maximum(mask.astype(np.float64), live.astype(np.float64))
    max_brightness = np.percentile(brightness[region], cfg.safe_background_max_brightness_percentile)
    texture = gradient_magnitude(mask, cfg.gaussian_sigma)
    texture = gaussian_filter(texture, sigma=1.0)

    safe = (
        region
        & ~vessel_exclusion
        & (dsa_abs <= max_dsa)
        & (brightness <= max_brightness)
        & (texture >= cfg.safe_background_min_texture)
    )

    if np.mean(safe[region]) < 0.04:
        relaxed_dsa = np.percentile(dsa_abs[region], min(95.0, cfg.safe_background_max_abs_dsa_percentile + 5.0))
        relaxed_texture = max(1.5, cfg.safe_background_min_texture * 0.5)
        safe = region & ~vessel_exclusion & (dsa_abs <= relaxed_dsa) & (brightness <= max_brightness) & (texture >= relaxed_texture)

    safe &= roi
    return safe.astype(bool)


def compute_registration_weight_map(
    mask: np.ndarray,
    live: np.ndarray,
    roi: np.ndarray,
    vessels: np.ndarray,
    vascular_bed: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    region = roi.astype(bool)
    if not np.any(region):
        return np.zeros_like(mask, dtype=np.float64)

    dsa_abs = np.abs(live.astype(np.float64) - mask.astype(np.float64))
    dsa_scale = np.percentile(dsa_abs[region], cfg.registration_residual_percentile) + 1e-8
    dsa_weight = np.exp(-(dsa_abs / dsa_scale) ** 2)

    brightness = np.maximum(mask.astype(np.float64), live.astype(np.float64))
    brightness_scale = np.percentile(brightness[region], cfg.safe_background_max_brightness_percentile) + 1e-8
    brightness_excess = np.maximum(0.0, brightness - brightness_scale) / (255.0 - brightness_scale + 1e-8)
    brightness_weight = np.exp(-4.0 * brightness_excess**2)

    texture = gradient_magnitude(mask, cfg.gaussian_sigma)
    texture = gaussian_filter(texture, sigma=1.0)
    texture_weight = np.clip(texture / (cfg.safe_background_min_texture + 1e-8), 0.0, 1.0)

    mask_grad = gradient_magnitude(mask, cfg.gaussian_sigma)
    live_grad = gradient_magnitude(live, cfg.gaussian_sigma)
    grad_diff = np.abs(mask_grad - live_grad)
    grad_scale = np.percentile(grad_diff[region], 85) + 1e-8
    gradient_similarity = np.exp(-(grad_diff / grad_scale) ** 2)
    structural_weight = (1.0 - cfg.registration_gradient_similarity_weight) + cfg.registration_gradient_similarity_weight * gradient_similarity

    vessel_soft = binary_dilation(
        vessels.astype(bool),
        structure=np.ones((3, 3)),
        iterations=max(1, int(cfg.safe_background_vessel_dilation)),
    ).astype(np.float64)
    vessel_soft = gaussian_filter(vessel_soft, sigma=1.5)
    vessel_weight = 1.0 - cfg.registration_vessel_penalty * np.clip(vessel_soft, 0.0, 1.0)

    bed_soft = gaussian_filter(vascular_bed.astype(np.float64), sigma=3.0) if vascular_bed is not None else 0.0
    if np.max(bed_soft) > 0:
        bed_soft = bed_soft / (np.max(bed_soft) + 1e-8)
    bed_weight = 1.0 - cfg.registration_bed_penalty * bed_soft

    weight = dsa_weight * brightness_weight * texture_weight * structural_weight * vessel_weight * bed_weight
    weight = gaussian_filter(weight, sigma=1.0)
    weight = np.clip(weight, 0.0, 1.0)
    weight *= region.astype(np.float64)
    weight = np.where(region & (weight > 0), np.maximum(weight, cfg.registration_weight_floor), weight)
    return weight.astype(np.float64)

