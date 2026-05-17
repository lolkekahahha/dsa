import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

from .config import PipelineConfig


def subtract_live_minus_mask(live: np.ndarray, warped_mask: np.ndarray) -> np.ndarray:
    return np.clip(live.astype(np.float64) - warped_mask.astype(np.float64), -255.0, 255.0)


def correct_warped_mask_intensity(
    warped_mask: np.ndarray,
    live: np.ndarray,
    weights: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, dict]:
    strength = float(np.clip(cfg.subtraction_intensity_correction_strength, 0.0, 1.0))
    if strength <= 0.0:
        return warped_mask.astype(np.float64), {"method": "local", "applied": False, "reason": "zero_strength"}

    source = warped_mask.astype(np.float64)
    target = live.astype(np.float64)
    w = np.clip(weights.astype(np.float64), 0.0, 1.0)
    valid = w > cfg.subtraction_intensity_min_support
    support_fraction = float(np.mean(valid))
    if np.sum(valid) < max(64, 0.01 * source.size) or support_fraction < cfg.subtraction_intensity_min_support_fraction:
        return source, {
            "method": "local",
            "applied": False,
            "reason": "insufficient_support",
            "support_fraction": support_fraction,
        }

    scale_min = float(cfg.subtraction_intensity_scale_min)
    scale_max = float(cfg.subtraction_intensity_scale_max)
    sigma = max(3.0, float(cfg.subtraction_intensity_correction_sigma))
    norm = gaussian_filter(w, sigma=sigma) + 1e-8
    x_mean = gaussian_filter(source * w, sigma=sigma) / norm
    y_mean = gaussian_filter(target * w, sigma=sigma) / norm
    xy_mean = gaussian_filter(source * target * w, sigma=sigma) / norm
    xx_mean = gaussian_filter(source * source * w, sigma=sigma) / norm
    var_x = np.maximum(xx_mean - x_mean * x_mean, 1e-8)
    cov_xy = xy_mean - x_mean * y_mean
    scale = np.clip(cov_xy / var_x, scale_min, scale_max)
    offset = y_mean - scale * x_mean
    corrected = scale * source + offset
    support = np.clip(norm / (np.percentile(norm[norm > 1e-8], 90) + 1e-8), 0.0, 1.0)
    blend = strength * support
    result = source * (1.0 - blend) + corrected * blend
    return np.clip(result, 0.0, 255.0), {
        "method": "local",
        "applied": True,
        "scale_mean": float(np.mean(scale[w > cfg.subtraction_intensity_min_support])),
        "offset_mean": float(np.mean(offset[w > cfg.subtraction_intensity_min_support])),
        "support_fraction": support_fraction,
    }


def correct_background_haze(
    dsa: np.ndarray,
    roi: np.ndarray,
    vessel_core: np.ndarray,
    support_weights: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, dict]:
    strength = float(np.clip(cfg.background_haze_correction_strength, 0.0, 1.0))
    if not cfg.enable_background_haze_correction or strength <= 0.0:
        return dsa.astype(np.float64), {"method": "weighted_low_frequency_bias", "applied": False, "reason": "disabled"}

    region = roi.astype(bool)
    if not np.any(region):
        return dsa.astype(np.float64), {"method": "weighted_low_frequency_bias", "applied": False, "reason": "empty_roi"}

    vessel_exclusion = binary_dilation(
        vessel_core.astype(bool),
        structure=np.ones((3, 3)),
        iterations=max(1, int(cfg.background_haze_vessel_dilation)),
    )
    weights = region.astype(np.float64) * (~vessel_exclusion).astype(np.float64)
    weights *= np.clip(support_weights.astype(np.float64), 0.0, 1.0)
    support_fraction = float(np.mean(weights[region] > 1e-3))
    if support_fraction < cfg.background_haze_min_support_fraction or np.sum(weights > 1e-3) < 64:
        return dsa.astype(np.float64), {
            "method": "weighted_low_frequency_bias",
            "applied": False,
            "reason": "insufficient_support",
            "support_fraction": support_fraction,
        }

    sigma = max(3.0, float(cfg.background_haze_sigma))
    norm = gaussian_filter(weights, sigma=sigma) + 1e-8
    bias = gaussian_filter(dsa.astype(np.float64) * weights, sigma=sigma) / norm
    support = np.clip(norm / (np.percentile(norm[norm > 1e-8], 90) + 1e-8), 0.0, 1.0)

    vessel_soft = gaussian_filter(vessel_exclusion.astype(np.float64), sigma=max(1.0, 0.20 * sigma))
    if np.max(vessel_soft) > 0:
        vessel_soft = vessel_soft / (np.max(vessel_soft) + 1e-8)
    protection = 1.0 - float(np.clip(cfg.background_haze_vessel_protection, 0.0, 1.0)) * vessel_soft
    correction_weight = strength * support * protection * region.astype(np.float64)
    corrected = dsa.astype(np.float64) - correction_weight * bias
    return np.clip(corrected, -255.0, 255.0), {
        "method": "weighted_low_frequency_bias",
        "applied": True,
        "support_fraction": support_fraction,
        "bias_abs_mean_roi": float(np.mean(np.abs(bias[region]))),
        "correction_weight_mean_roi": float(np.mean(correction_weight[region])),
    }


def displacement_magnitude(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    return np.sqrt(dx * dx + dy * dy)


def _abs_region_stats(values: np.ndarray, region: np.ndarray) -> dict:
    selected = np.abs(values.astype(np.float64)[region.astype(bool)])
    if selected.size == 0:
        return {"pixels": 0, "abs_mean": 0.0, "abs_std": 0.0, "abs_p90": 0.0, "abs_p95": 0.0, "abs_p99": 0.0}
    return {
        "pixels": int(selected.size),
        "abs_mean": float(np.mean(selected)),
        "abs_std": float(np.std(selected)),
        "abs_p90": float(np.percentile(selected, 90)),
        "abs_p95": float(np.percentile(selected, 95)),
        "abs_p99": float(np.percentile(selected, 99)),
    }


def _relative_drop(before: float, after: float) -> float:
    return float((before - after) / (before + 1e-8)) if before > 0 else 0.0


def _gradient_abs_mean(values: np.ndarray, region: np.ndarray) -> float:
    if not np.any(region):
        return 0.0
    gy, gx = np.gradient(values.astype(np.float64))
    grad = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(grad[region.astype(bool)]))


def basic_metrics(dsa_no_comp: np.ndarray, dsa_result: np.ndarray, roi: np.ndarray, vessels: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> dict:
    roi = roi.astype(bool)
    vessels = vessels.astype(bool)
    background = roi & ~vessels
    region = background if np.any(background) else roi
    mag = displacement_magnitude(dx, dy)
    before = np.abs(dsa_no_comp[region])
    after = np.abs(dsa_result[region])
    improvement = (float(np.std(before)) - float(np.std(after))) / (float(np.std(before)) + 1e-8)
    background_before = _abs_region_stats(dsa_no_comp, background)
    background_after = _abs_region_stats(dsa_result, background)
    vessel_region = roi & vessels
    vessel_before = _abs_region_stats(dsa_no_comp, vessel_region)
    vessel_after = _abs_region_stats(dsa_result, vessel_region)
    roi_before = _abs_region_stats(dsa_no_comp, roi)
    roi_after = _abs_region_stats(dsa_result, roi)
    background_mean_before = background_before["abs_mean"]
    background_mean_after = background_after["abs_mean"]
    vessel_mean_before = vessel_before["abs_mean"]
    vessel_mean_after = vessel_after["abs_mean"]
    contrast_before = vessel_mean_before - background_mean_before
    contrast_after = vessel_mean_after - background_mean_after
    return {
        "mean_displacement": float(np.mean(mag[roi])) if np.any(roi) else 0.0,
        "max_displacement": float(np.max(mag[roi])) if np.any(roi) else 0.0,
        "background_std_improvement": float(improvement),
        "background_abs_mean_improvement": _relative_drop(background_mean_before, background_mean_after),
        "background_p95_abs_improvement": _relative_drop(background_before["abs_p95"], background_after["abs_p95"]),
        "roi_abs_mean_improvement": _relative_drop(roi_before["abs_mean"], roi_after["abs_mean"]),
        "vessel_abs_signal_ratio": float(vessel_mean_after / (vessel_mean_before + 1e-8)) if vessel_before["pixels"] else 0.0,
        "vessel_contrast_before": float(contrast_before),
        "vessel_contrast_after": float(contrast_after),
        "vessel_contrast_ratio": float(contrast_after / (contrast_before + 1e-8)) if abs(contrast_before) > 1e-8 else 0.0,
        "vessel_edge_strength_ratio": (
            _gradient_abs_mean(dsa_result, vessel_region) / (_gradient_abs_mean(dsa_no_comp, vessel_region) + 1e-8)
            if np.any(vessel_region)
            else 0.0
        ),
        "background_before": background_before,
        "background_after": background_after,
        "vessel_before": vessel_before,
        "vessel_after": vessel_after,
        "roi_before": roi_before,
        "roi_after": roi_after,
        "result_vs_no_abs_mean_roi": float(np.mean(np.abs(dsa_result[roi] - dsa_no_comp[roi]))) if np.any(roi) else 0.0,
        "vessel_density_roi": float(np.mean(vessels[roi])) if np.any(roi) else 0.0,
    }
