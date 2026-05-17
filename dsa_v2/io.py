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


def percentile_window(*images: np.ndarray, percentiles: Tuple[float, float] = (1.0, 99.0)) -> Tuple[float, float]:
    values = np.concatenate([img.astype(np.float64).ravel() for img in images])
    lo, hi = np.percentile(values, percentiles)
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


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
