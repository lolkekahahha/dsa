import numpy as np

from .config import PipelineConfig
from .features import gradient_magnitude

REGION_MOTION_WEIGHTS = {
    "stable_background": 1.00,
    "vascular_bed": 0.75,
    "near_vessel": 0.45,
    "bright_or_metal": 0.55,
    "low_texture": 0.55,
    "low_weight": 0.40,
    "outside_roi": 0.00,
}


def patch_bounds(shape: tuple[int, int], x: float, y: float, radius: int) -> tuple[int, int, int, int]:
    h, w = shape
    cx, cy = int(round(x)), int(round(y))
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    return y0, y1, x0, x1


def classify_point_region(
    mask: np.ndarray,
    live: np.ndarray,
    point: tuple[float, float],
    vessels: np.ndarray,
    vascular_bed: np.ndarray,
    roi: np.ndarray,
    safe_mask: np.ndarray,
    registration_weight: np.ndarray,
    cfg: PipelineConfig,
) -> dict:
    x, y = point
    h, w = mask.shape
    xi, yi = int(round(x)), int(round(y))
    radius = max(4, cfg.template_size // 4)
    y0, y1, x0, x1 = patch_bounds(mask.shape, x, y, radius)
    patch = mask[y0:y1, x0:x1]
    if patch.size == 0 or xi < 0 or yi < 0 or xi >= w or yi >= h:
        return {
            "region_type": "outside_roi",
            "region_flags": ["outside_image"],
            "vessel_fraction": 1.0,
            "vascular_bed_fraction": 1.0,
            "safe_fraction": 0.0,
            "registration_weight_point": 0.0,
            "registration_weight_mean": 0.0,
            "texture_std": 0.0,
            "gradient_p90": 0.0,
            "max_brightness": 0.0,
        }

    vessel_fraction = float(np.mean(vessels[y0:y1, x0:x1]))
    bed_fraction = float(np.mean(vascular_bed[y0:y1, x0:x1])) if vascular_bed is not None else 0.0
    safe_fraction = float(np.mean(safe_mask[y0:y1, x0:x1])) if safe_mask is not None else 0.0
    weight_patch = registration_weight[y0:y1, x0:x1] if registration_weight is not None else np.ones_like(patch, dtype=np.float64)
    weight_point = float(registration_weight[yi, xi]) if registration_weight is not None else 1.0
    weight_mean = float(np.mean(weight_patch))
    texture_std = float(np.std(patch))
    grad = gradient_magnitude(patch, cfg.gaussian_sigma)
    gradient_p90 = float(np.percentile(grad, 90)) if grad.size else 0.0
    brightness_patch = np.maximum(mask[y0:y1, x0:x1], live[y0:y1, x0:x1])
    max_brightness = float(np.max(brightness_patch)) if brightness_patch.size else 0.0
    roi_patch = roi[y0:y1, x0:x1]
    roi_fraction = float(np.mean(roi_patch)) if roi_patch.size else 0.0

    region = roi.astype(bool)
    brightness_limit = float(np.percentile(np.maximum(mask, live)[region], cfg.safe_background_max_brightness_percentile)) if np.any(region) else 255.0

    flags: list[str] = []
    if roi_fraction < 0.50:
        flags.append("outside_roi")
    if vessel_fraction > 0.08:
        flags.append("near_vessel")
    if bed_fraction > 0.25:
        flags.append("vascular_bed")
    if max_brightness >= brightness_limit:
        flags.append("bright_or_metal")
    if texture_std < cfg.texture_point_min_std:
        flags.append("low_texture")
    if weight_mean < cfg.registration_weight_min_for_points:
        flags.append("low_weight")

    priority = ["outside_roi", "near_vessel", "bright_or_metal", "low_texture", "low_weight", "vascular_bed"]
    region_type = "stable_background"
    for candidate in priority:
        if candidate in flags:
            region_type = candidate
            break

    return {
        "region_type": region_type,
        "region_flags": flags,
        "vessel_fraction": vessel_fraction,
        "vascular_bed_fraction": bed_fraction,
        "safe_fraction": safe_fraction,
        "registration_weight_point": weight_point,
        "registration_weight_mean": weight_mean,
        "texture_std": texture_std,
        "gradient_p90": gradient_p90,
        "max_brightness": max_brightness,
    }


def annotate_match_regions(
    matches: list[dict],
    mask: np.ndarray,
    live: np.ndarray,
    vessels: np.ndarray,
    vascular_bed: np.ndarray,
    roi: np.ndarray,
    safe_mask: np.ndarray,
    registration_weight: np.ndarray,
    cfg: PipelineConfig,
) -> list[dict]:
    annotated = []
    for item in matches:
        copy = dict(item)
        x, y = item["point"]
        dx, dy = item["displacement"]
        mask_region = classify_point_region(mask, live, (x, y), vessels, vascular_bed, roi, safe_mask, registration_weight, cfg)
        live_region = classify_point_region(mask, live, (x + dx, y + dy), vessels, vascular_bed, roi, safe_mask, registration_weight, cfg)
        copy["live_point"] = [float(x + dx), float(y + dy)]
        copy["mask_region"] = mask_region
        copy["live_region"] = live_region
        copy["region_type"] = mask_region["region_type"]
        copy["region_flags"] = sorted(set(mask_region["region_flags"] + live_region["region_flags"]))
        copy["region_motion_weight"] = float(REGION_MOTION_WEIGHTS.get(copy["region_type"], 0.50))
        annotated.append(copy)
    return annotated


def region_summary(matches: list[dict]) -> dict:
    summary: dict[str, dict[str, int]] = {}
    for item in matches:
        region_type = item.get("region_type", "unknown")
        if region_type not in summary:
            summary[region_type] = {"total": 0, "accepted": 0, "rejected": 0}
        summary[region_type]["total"] += 1
        if item.get("accepted"):
            summary[region_type]["accepted"] += 1
        else:
            summary[region_type]["rejected"] += 1
    return summary
