from dataclasses import dataclass

import numpy as np

from .config import PipelineConfig


@dataclass
class MotionDecision:
    model: str
    apply: bool
    count: int
    mean_confidence: float
    median_magnitude: float
    translation_residual: float | None
    affine_residual: float | None
    coherence: float
    translation: np.ndarray
    affine_coeffs: np.ndarray
    inlier_fraction: float = 1.0
    translation_variation_ratio: float = 0.0
    affine_variation_ratio: float = 0.0
    affine_improvement: float = 0.0
    preferred_nonrigid: bool = False

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "apply": self.apply,
            "count": self.count,
            "mean_confidence": self.mean_confidence,
            "median_magnitude": self.median_magnitude,
            "translation_residual": self.translation_residual,
            "affine_residual": self.affine_residual,
            "coherence": self.coherence,
            "inlier_fraction": self.inlier_fraction,
            "translation_variation_ratio": self.translation_variation_ratio,
            "affine_variation_ratio": self.affine_variation_ratio,
            "affine_improvement": self.affine_improvement,
            "preferred_nonrigid": self.preferred_nonrigid,
        }


def weighted_lstsq(design: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    sqrt_w = np.sqrt(np.clip(weights, 0.01, 1.0))[:, None]
    lhs = design * sqrt_w
    rhs = target * sqrt_w.ravel()
    coeffs, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return coeffs


def fit_affine(points: np.ndarray, displacements: np.ndarray, confidences: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    design = np.column_stack([np.ones(len(points)), points[:, 0], points[:, 1]])
    weights = np.clip(confidences, 0.01, 1.0)
    keep = np.ones(len(points), dtype=bool)
    for _ in range(3):
        cx = weighted_lstsq(design[keep], displacements[keep, 0], weights[keep])
        cy = weighted_lstsq(design[keep], displacements[keep, 1], weights[keep])
        pred = np.column_stack([design @ cx, design @ cy])
        residual = np.linalg.norm(displacements - pred, axis=1)
        med = np.median(residual)
        mad = np.median(np.abs(residual - med)) + 1e-8
        new_keep = residual <= max(1.5, med + 2.5 * 1.4826 * mad)
        if np.sum(new_keep) < max(3, len(points) // 3) or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return np.vstack(
        [
            weighted_lstsq(design[keep], displacements[keep, 0], weights[keep]),
            weighted_lstsq(design[keep], displacements[keep, 1], weights[keep]),
        ]
    )


def evaluate_affine(points: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(points)), points[:, 0], points[:, 1]])
    return np.column_stack([design @ coeffs[0], design @ coeffs[1]])


def filter_outliers(points: np.ndarray, displacements: np.ndarray, confidences: np.ndarray, cfg: PipelineConfig):
    if len(points) < 6:
        return points, displacements, confidences
    mags = np.linalg.norm(displacements, axis=1)
    keep = mags <= cfg.max_displacement
    return points[keep], displacements[keep], confidences[keep]


def ransac_translation(displacements: np.ndarray, confidences: np.ndarray, cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    if len(displacements) == 0:
        return np.zeros(2, dtype=np.float64), np.zeros(0, dtype=bool)
    best_inliers = np.zeros(len(displacements), dtype=bool)
    best_score = -1.0
    for candidate in displacements:
        residual = np.linalg.norm(displacements - candidate, axis=1)
        inliers = residual <= cfg.ransac_residual_threshold
        score = float(np.sum(confidences[inliers]))
        if score > best_score:
            best_score = score
            best_inliers = inliers
    if np.any(best_inliers):
        translation = np.average(displacements[best_inliers], axis=0, weights=np.clip(confidences[best_inliers], 0.01, 1.0))
    else:
        translation = np.average(displacements, axis=0, weights=np.clip(confidences, 0.01, 1.0))
    return translation, best_inliers


def ransac_affine(points: np.ndarray, displacements: np.ndarray, confidences: np.ndarray, cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 3:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64), np.zeros(len(points), dtype=bool)
    rng = np.random.default_rng(17)
    best_inliers = np.zeros(len(points), dtype=bool)
    best_score = -1.0
    iterations = min(max(1, cfg.ransac_iterations), 1000)
    for _ in range(iterations):
        sample = rng.choice(len(points), size=3, replace=False)
        coeffs = fit_affine(points[sample], displacements[sample], confidences[sample])
        pred = evaluate_affine(points, coeffs)
        residual = np.linalg.norm(displacements - pred, axis=1)
        inliers = residual <= cfg.ransac_residual_threshold
        score = float(np.sum(confidences[inliers]))
        if score > best_score:
            best_score = score
            best_inliers = inliers
    if np.sum(best_inliers) >= 3:
        coeffs = fit_affine(points[best_inliers], displacements[best_inliers], confidences[best_inliers])
    else:
        coeffs = fit_affine(points, displacements, confidences)
    return coeffs, best_inliers


def analyze_motion(points: np.ndarray, displacements: np.ndarray, confidences: np.ndarray, cfg: PipelineConfig) -> MotionDecision:
    empty_coeffs = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    if len(points) < cfg.min_motion_inliers:
        return MotionDecision("insufficient", False, int(len(points)), float(np.mean(confidences)) if len(confidences) else 0.0, 0.0, None, None, 0.0, np.zeros(2), empty_coeffs, 0.0)

    original_count = len(points)
    if cfg.enable_ransac:
        translation, trans_inliers = ransac_translation(displacements, confidences, cfg)
        _, affine_inliers = ransac_affine(points, displacements, confidences, cfg)
        candidate_inliers = affine_inliers if np.sum(affine_inliers) >= np.sum(trans_inliers) else trans_inliers
        if np.sum(candidate_inliers) >= cfg.min_motion_inliers:
            points = points[candidate_inliers]
            displacements = displacements[candidate_inliers]
            confidences = confidences[candidate_inliers]
    inlier_fraction = float(len(points) / max(1, original_count))

    weights = np.clip(confidences, 0.01, 1.0)
    translation = np.average(displacements, axis=0, weights=weights)
    trans_res = np.linalg.norm(displacements - translation, axis=1)
    trans_rmse = float(np.sqrt(np.average(trans_res**2, weights=weights)))

    coeffs = fit_affine(points, displacements, confidences)
    affine_pred = evaluate_affine(points, coeffs)
    affine_res = np.linalg.norm(displacements - affine_pred, axis=1)
    affine_rmse = float(np.sqrt(np.average(affine_res**2, weights=weights)))

    mags = np.linalg.norm(displacements, axis=1)
    median_mag = float(np.median(mags))
    mean_conf = float(np.mean(confidences))
    coherence = float(np.clip(1.0 - affine_rmse / max(1.0, median_mag + 1e-8), 0.0, 1.0))

    affine_improvement = 0.0 if trans_rmse <= 1e-8 else (trans_rmse - affine_rmse) / trans_rmse
    enough_nonrigid = len(points) >= max(cfg.min_motion_inliers, cfg.nonrigid_min_points)
    motion_scale = max(1.0, median_mag)
    translation_variation_ratio = float(trans_rmse / motion_scale)
    affine_variation_ratio = float(affine_rmse / motion_scale)
    coherent_nonrigid_variation = (
        enough_nonrigid
        and median_mag >= cfg.nonrigid_variation_min_magnitude
        and translation_variation_ratio >= cfg.nonrigid_translation_variation_ratio
        and affine_variation_ratio >= cfg.nonrigid_affine_variation_ratio
        and coherence >= cfg.nonrigid_coherent_variation_min
        and mean_conf >= cfg.nonrigid_min_confidence
    )
    preferred_nonrigid = (
        cfg.prefer_nonrigid_registration
        and len(points) >= max(cfg.min_motion_inliers, cfg.nonrigid_preference_min_points)
        and mean_conf >= cfg.nonrigid_preference_min_confidence
        and coherence >= cfg.nonrigid_preference_min_coherence
        and inlier_fraction >= cfg.nonrigid_preference_min_inlier_fraction
        and (
            translation_variation_ratio >= cfg.nonrigid_preference_min_variation_ratio
            or affine_variation_ratio >= cfg.nonrigid_preference_min_variation_ratio
        )
    )
    coherent_micro_motion = (
        median_mag <= cfg.micro_motion_max_median
        and mean_conf >= cfg.micro_motion_min_confidence
        and trans_rmse <= max(cfg.rigid_residual_threshold, cfg.micro_motion_rigid_residual_threshold)
        and inlier_fraction >= 0.50
    )

    if mean_conf < cfg.micro_motion_min_confidence or median_mag > cfg.max_displacement:
        model, apply = "unreliable", False
    elif coherent_nonrigid_variation or preferred_nonrigid:
        model, apply = "nonrigid", True
    elif coherent_micro_motion:
        model, apply = "rigid", True
    elif trans_rmse <= cfg.rigid_residual_threshold:
        model, apply = "rigid", True
    elif affine_rmse <= cfg.affine_residual_threshold and affine_improvement >= cfg.affine_min_improvement:
        model, apply = "affine", True
    elif enough_nonrigid and coherence >= cfg.nonrigid_min_coherence and mean_conf >= cfg.nonrigid_min_confidence:
        model, apply = "nonrigid", True
    elif affine_rmse <= cfg.affine_residual_threshold and mean_conf >= cfg.nonrigid_min_confidence:
        model, apply = "affine", True
    else:
        model, apply = "unreliable", False

    return MotionDecision(
        model,
        apply,
        int(len(points)),
        mean_conf,
        median_mag,
        trans_rmse,
        affine_rmse,
        coherence,
        translation,
        coeffs,
        inlier_fraction,
        translation_variation_ratio,
        affine_variation_ratio,
        affine_improvement,
        bool(preferred_nonrigid),
    )


def select_motion_profile(decision: MotionDecision, global_shift: np.ndarray, cfg: PipelineConfig) -> str:
    if not decision.apply:
        return "standard"
    global_mag = float(np.linalg.norm(global_shift))
    if (
        decision.median_magnitude <= cfg.micro_motion_max_median
        and decision.mean_confidence >= cfg.micro_motion_min_confidence
        and global_mag <= cfg.micro_motion_max_median
    ):
        return "micro_motion"
    return "standard"
