from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_grayscale(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float64)


def normalize_to_uint8_range(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    max_value = float(np.max(image))
    if max_value > 255.0:
        image = image / (max_value + 1e-8) * 255.0
    return image.astype(np.float64)


def to_uint8_window(image: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    image = image.astype(np.float64)
    return np.clip((image - vmin) / (vmax - vmin + 1e-8) * 255.0, 0, 255).astype(np.uint8)


def enhance_dsa_display(image: np.ndarray) -> np.ndarray:
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    image_f = image_u8.astype(np.float64)

    local_mean = cv2.GaussianBlur(image_f, (0, 0), sigmaX=36.0)
    local_square_mean = cv2.GaussianBlur(image_f * image_f, (0, 0), sigmaX=36.0)
    local_std = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 1.0))
    target_mean = float(np.percentile(image_f, 50))
    gain = np.clip(39.0 / (local_std + 1e-8), 0.62, 2.20)
    locally_balanced = (image_f - local_mean) * gain + target_mean
    balanced = 0.66 * image_f + 0.34 * locally_balanced
    balanced_u8 = np.clip(balanced, 0, 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    dark_structures = cv2.morphologyEx(balanced_u8, cv2.MORPH_BLACKHAT, kernel).astype(np.float64)
    detail_mean = cv2.GaussianBlur(balanced_u8.astype(np.float64), (0, 0), sigmaX=18.0)
    detail_std = np.sqrt(
        cv2.GaussianBlur((balanced_u8.astype(np.float64) - detail_mean) ** 2, (0, 0), sigmaX=18.0)
    )
    dark_structure_gate = np.clip((48.0 - detail_std) / 35.0, 0.0, 1.0)
    enhanced = balanced - dark_structure_gate * dark_structures
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def enhance_vessel_display(
    image: np.ndarray,
    vessel_support: np.ndarray,
    roi: np.ndarray,
    strength: float = 0.45,
    background_smoothing: float = 0.28,
    support_sigma: float = 2.5,
) -> np.ndarray:
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    if strength <= 0.0:
        return image_u8

    image_f = image_u8.astype(np.float64)
    roi_mask = roi.astype(bool) if roi is not None and np.any(roi) else np.ones_like(image_u8, dtype=bool)
    support_mask = vessel_support.astype(bool) & roi_mask if vessel_support is not None else np.zeros_like(roi_mask)

    support = cv2.GaussianBlur(
        support_mask.astype(np.float64),
        (0, 0),
        sigmaX=max(0.5, float(support_sigma)),
    )
    if np.max(support) > 0:
        support = support / (np.max(support) + 1e-8)

    dark_score = np.zeros_like(image_f)
    for size in (9, 15, 23):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        dark_score = np.maximum(dark_score, cv2.morphologyEx(image_u8, cv2.MORPH_BLACKHAT, kernel).astype(np.float64))

    local_mean = cv2.GaussianBlur(image_f, (0, 0), sigmaX=18.0)
    darkness = np.clip((local_mean - image_f) / 35.0, 0.0, 1.0)
    roi_values = dark_score[roi_mask]
    if roi_values.size:
        lo = float(np.percentile(roi_values, 55))
        hi = float(np.percentile(roi_values, 98))
    else:
        lo, hi = 0.0, 1.0
    tubular = np.clip((dark_score - lo) / (hi - lo + 1e-8), 0.0, 1.0) * darkness * roi_mask.astype(np.float64)
    vessel_weight = np.clip(np.maximum(support, tubular), 0.0, 1.0)
    vessel_weight = cv2.GaussianBlur(vessel_weight, (0, 0), sigmaX=0.8)

    background_weight = roi_mask.astype(np.float64) * (1.0 - np.clip(1.35 * vessel_weight, 0.0, 1.0))
    denoised = cv2.bilateralFilter(image_u8, d=7, sigmaColor=18, sigmaSpace=5).astype(np.float64)
    smoothed = image_f * (1.0 - background_smoothing * background_weight) + denoised * (background_smoothing * background_weight)

    vessel_darkening = float(np.clip(strength, 0.0, 1.0)) * (8.0 + 28.0 * tubular) * vessel_weight
    enhanced = smoothed - vessel_darkening
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def dsa_display_window(*images: np.ndarray, percentile: float = 99.0) -> Tuple[float, float]:
    signed_values = np.concatenate([img.astype(np.float64).ravel() for img in images])
    positive_limit = float(np.percentile(np.maximum(signed_values, 0.0), percentile))
    negative_limit = float(np.percentile(np.maximum(-signed_values, 0.0), percentile))
    abs_limit = float(np.percentile(np.abs(signed_values), percentile))

    vmax = max(positive_limit, 0.65 * abs_limit, 1.0)
    tail_balance = negative_limit / (positive_limit + negative_limit + 1e-8)

    h, w = images[0].shape[:2]
    aspect = w / max(h, 1)
    compact_frame = 0.85 <= aspect <= 1.15 and min(h, w) <= 700
    dark_tail_level = 160.0 if compact_frame else 128.0
    if tail_balance <= 0.30:
        background_level = 98.0
    elif tail_balance >= 0.45:
        background_level = dark_tail_level
    else:
        t = (tail_balance - 0.30) / 0.15
        background_level = (1.0 - t) * 98.0 + t * dark_tail_level

    negative_scale = background_level / (255.0 - background_level)
    vmin = -negative_scale * vmax

    if negative_limit > 0:
        vmin = min(vmin, -0.80 * negative_limit)
    return float(vmin), float(vmax)


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_uint8(path: str | Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), np.clip(image, 0, 255).astype(np.uint8))
