import numpy as np
import cv2
from scipy.ndimage import binary_dilation

from .config import PipelineConfig
from .features import plain_gradient_magnitude


def weighted_zncc(template: np.ndarray, candidate: np.ndarray, weights: np.ndarray) -> float:
    weights = weights.astype(np.float64)
    weight_sum = float(np.sum(weights))
    if weight_sum < max(16, 0.2 * template.size):
        return -1.0
    t = template.astype(np.float64)
    c = candidate.astype(np.float64)
    t_mean = np.sum(t * weights) / weight_sum
    c_mean = np.sum(c * weights) / weight_sum
    t = t - t_mean
    c = c - c_mean
    denom = np.sqrt(np.sum(weights * t * t) * np.sum(weights * c * c))
    if denom < 1e-8:
        return -1.0
    return float(np.sum(weights * t * c) / denom)


def fit_local_intensity(source: np.ndarray, target: np.ndarray, weights: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    if not cfg.intensity_variation_modeling:
        return source.astype(np.float64)
    valid = weights > 0
    if np.sum(valid) < max(16, 0.2 * source.size):
        return source.astype(np.float64)
    x = source.astype(np.float64)[valid].ravel()
    y = target.astype(np.float64)[valid].ravel()
    w = weights.astype(np.float64)[valid].ravel()
    w_sum = np.sum(w) + 1e-8
    x_mean = np.sum(w * x) / w_sum
    y_mean = np.sum(w * y) / w_sum
    var_x = np.sum(w * (x - x_mean) ** 2)
    if var_x < 1e-8:
        return source.astype(np.float64) + (y_mean - x_mean)
    scale = np.sum(w * (x - x_mean) * (y - y_mean)) / var_x
    scale = float(np.clip(scale, 0.25, 4.0))
    offset = y_mean - scale * x_mean
    return scale * source.astype(np.float64) + offset


def histogram_entropy_similarity(template: np.ndarray, candidate: np.ndarray, weights: np.ndarray, cfg: PipelineConfig) -> float:
    valid = weights > 0
    if np.sum(valid) < max(16, 0.2 * template.size):
        return -1.0
    diff = (template.astype(np.float64) - candidate.astype(np.float64))[valid]
    sample_weights = weights.astype(np.float64)[valid]
    spread = max(float(np.percentile(np.abs(diff), 98)), 8.0)
    bins = max(8, int(cfg.entropy_bins))
    hist, _ = np.histogram(diff, bins=bins, range=(-spread, spread), weights=sample_weights)
    prob = hist.astype(np.float64)
    prob_sum = np.sum(prob)
    if prob_sum <= 0:
        return -1.0
    prob = prob / prob_sum
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log(prob))
    entropy_norm = entropy / (np.log(bins) + 1e-8)
    return float(2.0 * (1.0 - np.clip(entropy_norm, 0.0, 1.0)) - 1.0)


def patch_score(
    mask: np.ndarray,
    live: np.ndarray,
    point: np.ndarray,
    dx: int,
    dy: int,
    exclusion_mask: np.ndarray,
    cfg: PipelineConfig,
    mask_grad: np.ndarray,
    live_grad: np.ndarray,
    registration_weight: np.ndarray | None = None,
) -> float:
    cx, cy = int(point[0]), int(point[1])
    half = cfg.template_size // 2
    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    yy0, yy1 = cy + dy - half, cy + dy + half + 1
    xx0, xx1 = cx + dx - half, cx + dx + half + 1
    if y0 < 0 or x0 < 0 or yy0 < 0 or xx0 < 0 or y1 > mask.shape[0] or x1 > mask.shape[1] or yy1 > live.shape[0] or xx1 > live.shape[1]:
        return -1.0

    template = mask[y0:y1, x0:x1]
    candidate = live[yy0:yy1, xx0:xx1]
    excluded_t = exclusion_mask[y0:y1, x0:x1]
    excluded_c = exclusion_mask[yy0:yy1, xx0:xx1]
    weights = (~(excluded_t | excluded_c)).astype(np.float64)
    if registration_weight is not None:
        wt = registration_weight[y0:y1, x0:x1]
        wc = registration_weight[yy0:yy1, xx0:xx1]
        weights *= np.minimum(wt, wc)

    candidate_adj = fit_local_intensity(candidate, template, weights, cfg)
    zncc = weighted_zncc(template, candidate_adj, weights)
    entropy = histogram_entropy_similarity(template, candidate_adj, weights, cfg)
    intensity = 0.75 * zncc + 0.25 * entropy

    t_grad = mask_grad[y0:y1, x0:x1]
    c_grad = live_grad[yy0:yy1, xx0:xx1]
    c_grad_adj = fit_local_intensity(c_grad, t_grad, weights, cfg)
    grad = weighted_zncc(t_grad, c_grad_adj, weights)
    return float(0.45 * intensity + 0.55 * grad)


def patch_score_fast(
    mask: np.ndarray,
    live: np.ndarray,
    point: np.ndarray,
    dx: int,
    dy: int,
    exclusion_mask: np.ndarray,
    cfg: PipelineConfig,
    mask_grad: np.ndarray,
    live_grad: np.ndarray,
    registration_weight: np.ndarray | None = None,
) -> float:
    cx, cy = int(point[0]), int(point[1])
    half = cfg.template_size // 2
    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    yy0, yy1 = cy + dy - half, cy + dy + half + 1
    xx0, xx1 = cx + dx - half, cx + dx + half + 1
    if y0 < 0 or x0 < 0 or yy0 < 0 or xx0 < 0 or y1 > mask.shape[0] or x1 > mask.shape[1] or yy1 > live.shape[0] or xx1 > live.shape[1]:
        return -1.0

    excluded_t = exclusion_mask[y0:y1, x0:x1]
    excluded_c = exclusion_mask[yy0:yy1, xx0:xx1]
    weights = (~(excluded_t | excluded_c)).astype(np.float64)
    if registration_weight is not None:
        wt = registration_weight[y0:y1, x0:x1]
        wc = registration_weight[yy0:yy1, xx0:xx1]
        weights *= np.minimum(wt, wc)

    intensity = weighted_zncc(mask[y0:y1, x0:x1], live[yy0:yy1, xx0:xx1], weights)
    grad = weighted_zncc(mask_grad[y0:y1, x0:x1], live_grad[yy0:yy1, xx0:xx1], weights)
    return float(0.45 * intensity + 0.55 * grad)


def score_candidate(
    mask: np.ndarray,
    live: np.ndarray,
    point: np.ndarray,
    dx: int,
    dy: int,
    exclusion_mask: np.ndarray,
    cfg: PipelineConfig,
    mask_grad: np.ndarray,
    live_grad: np.ndarray,
    registration_weight: np.ndarray | None,
    fast: bool,
) -> float:
    scorer = patch_score_fast if fast else patch_score
    return scorer(mask, live, point, dx, dy, exclusion_mask, cfg, mask_grad, live_grad, registration_weight)


def estimate_one_displacement(
    mask: np.ndarray,
    live: np.ndarray,
    point: np.ndarray,
    exclusion_mask: np.ndarray,
    cfg: PipelineConfig,
    mask_grad: np.ndarray,
    live_grad: np.ndarray,
    registration_weight: np.ndarray | None = None,
) -> np.ndarray:
    displacement, _, _, _ = estimate_one_displacement_with_scores(
        mask,
        live,
        point,
        exclusion_mask,
        cfg,
        mask_grad,
        live_grad,
        registration_weight,
    )
    return displacement


def estimate_one_displacement_with_scores(
    mask: np.ndarray,
    live: np.ndarray,
    point: np.ndarray,
    exclusion_mask: np.ndarray,
    cfg: PipelineConfig,
    mask_grad: np.ndarray,
    live_grad: np.ndarray,
    registration_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if cfg.enable_coarse_to_fine and cfg.search_radius >= 6:
        displacement = estimate_one_displacement_coarse_to_fine(mask, live, point, exclusion_mask, cfg, mask_grad, live_grad, registration_weight)
        best_score = patch_score(
            mask,
            live,
            point,
            int(round(float(displacement[0]))),
            int(round(float(displacement[1]))),
            exclusion_mask,
            cfg,
            mask_grad,
            live_grad,
            registration_weight,
        )
        return displacement, np.asarray([best_score], dtype=np.float64), np.asarray([[displacement[0], displacement[1]]], dtype=np.float64), float(best_score)

    best_score = -np.inf
    best = np.array([0.0, 0.0], dtype=np.float64)
    scores = []
    coords = []
    fast_prefilter = bool(cfg.enable_fast_matching_prefilter)
    step = 2 if cfg.search_radius >= 12 else 1
    for dy in range(-cfg.search_radius, cfg.search_radius + 1, step):
        for dx in range(-cfg.search_radius, cfg.search_radius + 1, step):
            score = score_candidate(mask, live, point, dx, dy, exclusion_mask, cfg, mask_grad, live_grad, registration_weight, fast_prefilter)
            scores.append(score)
            coords.append((dx, dy))
            if score > best_score:
                best_score = score
                best = np.array([dx, dy], dtype=np.float64)
    if step > 1:
        cx, cy = int(best[0]), int(best[1])
        for dy in range(max(-cfg.search_radius, cy - 2), min(cfg.search_radius, cy + 2) + 1):
            for dx in range(max(-cfg.search_radius, cx - 2), min(cfg.search_radius, cx + 2) + 1):
                score = score_candidate(mask, live, point, dx, dy, exclusion_mask, cfg, mask_grad, live_grad, registration_weight, fast_prefilter)
                scores.append(score)
                coords.append((dx, dy))
                if score > best_score:
                    best_score = score
                    best = np.array([dx, dy], dtype=np.float64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    coords_arr = np.asarray(coords, dtype=np.float64)
    if fast_prefilter and scores_arr.size:
        top_k = min(max(1, int(cfg.matching_entropy_rerank_top_k)), scores_arr.size)
        top_indices = np.argpartition(scores_arr, -top_k)[-top_k:]
        best_full_score = -np.inf
        best_full = best.copy()
        for idx in top_indices:
            dx, dy = coords_arr[idx]
            full_score = patch_score(mask, live, point, int(dx), int(dy), exclusion_mask, cfg, mask_grad, live_grad, registration_weight)
            if full_score > best_full_score:
                best_full_score = full_score
                best_full = np.array([dx, dy], dtype=np.float64)
        best = best_full
        matching_rows = np.all(coords_arr == best[None, :], axis=1)
        if np.any(matching_rows):
            best_score = float(np.max(scores_arr[matching_rows]))
        else:
            best_score = float(np.max(scores_arr))
    if not cfg.enable_subpixel:
        return best, scores_arr, coords_arr, float(best_score)
    refined = refine_subpixel(mask, live, point, best, exclusion_mask, cfg, mask_grad, live_grad, registration_weight, fast=fast_prefilter)
    refined_score = score_candidate(
        mask,
        live,
        point,
        int(round(float(refined[0]))),
        int(round(float(refined[1]))),
        exclusion_mask,
        cfg,
        mask_grad,
        live_grad,
        registration_weight,
        fast_prefilter,
    )
    return refined, scores_arr, coords_arr, float(refined_score)


def estimate_one_displacement_coarse_to_fine(
    mask: np.ndarray,
    live: np.ndarray,
    point: np.ndarray,
    exclusion_mask: np.ndarray,
    cfg: PipelineConfig,
    mask_grad: np.ndarray,
    live_grad: np.ndarray,
    registration_weight: np.ndarray | None = None,
) -> np.ndarray:
    scale = float(np.clip(cfg.coarse_scale, 0.25, 0.75))
    small_size = (max(1, int(round(mask.shape[1] * scale))), max(1, int(round(mask.shape[0] * scale))))
    small_mask = cv2.resize(mask.astype(np.float64), small_size, interpolation=cv2.INTER_AREA)
    small_live = cv2.resize(live.astype(np.float64), small_size, interpolation=cv2.INTER_AREA)
    small_exclusion = cv2.resize(exclusion_mask.astype(np.uint8), small_size, interpolation=cv2.INTER_NEAREST).astype(bool)
    small_weight = None
    if registration_weight is not None:
        small_weight = cv2.resize(registration_weight.astype(np.float64), small_size, interpolation=cv2.INTER_AREA)
    small_mask_grad = plain_gradient_magnitude(small_mask)
    small_live_grad = plain_gradient_magnitude(small_live)

    small_cfg = PipelineConfig(**vars(cfg))
    small_cfg.template_size = max(15, int(round(cfg.template_size * scale)) | 1)
    small_cfg.search_radius = max(3, int(round(cfg.search_radius * scale)))
    small_cfg.max_displacement = max(2.0, cfg.max_displacement * scale)
    small_cfg.enable_subpixel = False
    small_cfg.enable_coarse_to_fine = False
    small_cfg.vessel_exclusion_dilation = max(1, int(round(cfg.vessel_exclusion_dilation * scale)))

    small_point = point.astype(np.float64) * scale
    coarse = estimate_one_displacement(small_mask, small_live, small_point, small_exclusion, small_cfg, small_mask_grad, small_live_grad, small_weight) / scale
    center = np.round(coarse).astype(int)

    best_score = -np.inf
    best = center.astype(np.float64)
    radius = max(1, int(cfg.fine_refine_radius))
    for dy in range(int(center[1]) - radius, int(center[1]) + radius + 1):
        for dx in range(int(center[0]) - radius, int(center[0]) + radius + 1):
            if abs(dx) > cfg.search_radius or abs(dy) > cfg.search_radius:
                continue
            score = score_candidate(
                mask,
                live,
                point,
                dx,
                dy,
                exclusion_mask,
                cfg,
                mask_grad,
                live_grad,
                registration_weight,
                bool(cfg.enable_fast_matching_prefilter),
            )
            if score > best_score:
                best_score = score
                best = np.array([dx, dy], dtype=np.float64)
    if not cfg.enable_subpixel:
        return best
    return refine_subpixel(mask, live, point, best, exclusion_mask, cfg, mask_grad, live_grad, registration_weight, fast=bool(cfg.enable_fast_matching_prefilter))


def refine_subpixel(mask, live, point, displacement, exclusion_mask, cfg, mask_grad, live_grad, registration_weight=None, fast: bool = False) -> np.ndarray:
    dx0, dy0 = int(round(float(displacement[0]))), int(round(float(displacement[1])))
    center = score_candidate(mask, live, point, dx0, dy0, exclusion_mask, cfg, mask_grad, live_grad, registration_weight, fast)
    left = score_candidate(mask, live, point, dx0 - 1, dy0, exclusion_mask, cfg, mask_grad, live_grad, registration_weight, fast)
    right = score_candidate(mask, live, point, dx0 + 1, dy0, exclusion_mask, cfg, mask_grad, live_grad, registration_weight, fast)
    up = score_candidate(mask, live, point, dx0, dy0 - 1, exclusion_mask, cfg, mask_grad, live_grad, registration_weight, fast)
    down = score_candidate(mask, live, point, dx0, dy0 + 1, exclusion_mask, cfg, mask_grad, live_grad, registration_weight, fast)

    def offset(a, b, c):
        denom = a - 2.0 * b + c
        if abs(denom) < 1e-8:
            return 0.0
        return float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))

    return np.array([dx0 + offset(left, center, right), dy0 + offset(up, center, down)], dtype=np.float64)


def displacement_confidence_from_scores(displacement: np.ndarray, scores: np.ndarray, coords: np.ndarray, best: float, cfg: PipelineConfig) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    if scores.size < 2 or coords.shape[0] != scores.size:
        score_conf = np.clip((best + 1.0) / 2.0, 0.0, 1.0)
        peak_conf = score_conf
    else:
        far = np.linalg.norm(coords - displacement[None, :], axis=1) >= 3.0
        if np.any(far):
            second = float(np.max(scores[far]))
        else:
            second = float(np.partition(scores, -2)[-2])
        peak_z = (best - np.mean(scores)) / (np.std(scores) + 1e-8)
        margin = max(0.0, best - second)
        peak_conf = np.clip(0.50 * (peak_z / 4.0) + 0.50 * (margin / 0.08), 0.0, 1.0)
        score_conf = np.clip((best + 1.0) / 2.0, 0.0, 1.0)
    mag = float(np.linalg.norm(displacement))
    if mag > cfg.max_displacement:
        return 0.0
    prior = np.exp(-0.7 * mag / (cfg.max_displacement + 1e-8))
    edge_distance = max(abs(float(displacement[0])), abs(float(displacement[1]))) / (cfg.search_radius + 1e-8)
    if edge_distance >= cfg.search_edge_reject_fraction:
        return 0.0
    edge_penalty = 1.0
    if edge_distance > cfg.search_edge_penalty_fraction:
        edge_penalty = np.clip(
            (cfg.search_edge_reject_fraction - edge_distance)
            / (cfg.search_edge_reject_fraction - cfg.search_edge_penalty_fraction + 1e-8),
            0.0,
            1.0,
        )
    conf = 0.55 * peak_conf + 0.30 * score_conf + 0.15 * prior
    return float(np.clip(conf * edge_penalty, 0.0, 1.0))


def spatial_consistency_filter(
    points: np.ndarray,
    displacements: np.ndarray,
    confidences: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(points)
    if not cfg.enable_spatial_consistency_filter or count < cfg.spatial_consistency_min_points:
        return np.ones(count, dtype=bool), np.zeros(count, dtype=np.float64)

    k = min(max(2, int(cfg.spatial_consistency_neighbors)), count - 1)
    errors = np.zeros(count, dtype=np.float64)
    for idx in range(count):
        distances = np.linalg.norm(points - points[idx][None, :], axis=1)
        order = np.argsort(distances)
        neighbors = order[1 : k + 1]
        weights = np.clip(confidences[neighbors], 0.03, 1.0)
        local = np.average(displacements[neighbors], axis=0, weights=weights)
        errors[idx] = float(np.linalg.norm(displacements[idx] - local))

    median_error = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median_error))) + 1e-8
    threshold = max(float(cfg.spatial_consistency_tolerance), median_error + float(cfg.spatial_consistency_mad_factor) * 1.4826 * mad)
    keep = errors <= threshold

    min_keep = max(int(np.ceil(count * float(cfg.spatial_consistency_min_keep_fraction))), int(cfg.spatial_consistency_min_points))
    if np.sum(keep) < min_keep:
        order = np.argsort(errors)
        keep = np.zeros(count, dtype=bool)
        keep[order[: min(count, min_keep)]] = True
    return keep, errors


def estimate_displacements(mask, live, points, reliabilities, vessel_mask, roi_mask, cfg: PipelineConfig, safe_mask=None, registration_weight=None):
    if len(points) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)), []
    exclusion = binary_dilation(vessel_mask, structure=np.ones((3, 3)), iterations=max(1, cfg.vessel_exclusion_dilation)) & roi_mask
    if safe_mask is not None and np.any(safe_mask):
        exclusion = exclusion | (roi_mask.astype(bool) & ~safe_mask.astype(bool))
    mask_grad = plain_gradient_magnitude(mask)
    live_grad = plain_gradient_magnitude(live)
    half = cfg.template_size // 2
    displacements = []
    confidences = []
    diagnostics = []
    accepted_flags = []
    for point, base_rel in zip(points, reliabilities):
        disp, scores, coords, best_score = estimate_one_displacement_with_scores(
            mask,
            live,
            point,
            exclusion,
            cfg,
            mask_grad,
            live_grad,
            registration_weight,
        )
        conf = displacement_confidence_from_scores(disp, scores, coords, best_score, cfg)
        fb_error = None
        reverse = np.array([np.nan, np.nan], dtype=np.float64)
        if cfg.enable_forward_backward_check and conf > 0.0:
            target = point + disp
            tx, ty = int(round(float(target[0]))), int(round(float(target[1])))
            inside = (
                half + cfg.search_radius <= tx < mask.shape[1] - half - cfg.search_radius
                and half + cfg.search_radius <= ty < mask.shape[0] - half - cfg.search_radius
            )
            if inside:
                reverse = estimate_one_displacement(live, mask, target, exclusion, cfg, live_grad, mask_grad, registration_weight)
                fb_error = float(np.linalg.norm(disp + reverse))
                conf *= float(np.exp(-fb_error / (cfg.forward_backward_tolerance + 1e-8)))
            else:
                fb_error = float("inf")
                conf = 0.0
        mag = float(np.linalg.norm(disp))
        edge_fraction = max(abs(float(disp[0])), abs(float(disp[1]))) / (cfg.search_radius + 1e-8)
        reject_reasons = []
        if mag > cfg.max_displacement:
            reject_reasons.append("magnitude")
        if edge_fraction >= cfg.search_edge_reject_fraction:
            reject_reasons.append("search_edge")
        if conf < cfg.min_match_confidence:
            reject_reasons.append("match_confidence")
        combined_conf = float(base_rel * conf)
        if combined_conf < cfg.min_combined_confidence:
            reject_reasons.append("combined_confidence")
        if fb_error is not None and np.isfinite(fb_error) and fb_error > cfg.forward_backward_tolerance:
            reject_reasons.append("forward_backward")
        if fb_error is not None and not np.isfinite(fb_error):
            reject_reasons.append("forward_backward_outside")
        combined_conf = float(base_rel * conf)
        field_confidence = float(conf * np.sqrt(np.clip(base_rel, 0.0, 1.0)))
        fb_ok = fb_error is None or (np.isfinite(fb_error) and fb_error <= cfg.forward_backward_tolerance)
        accepted = bool(
            conf >= cfg.min_match_confidence
            and combined_conf >= cfg.min_combined_confidence
            and mag <= cfg.max_displacement
            and edge_fraction < cfg.search_edge_reject_fraction
            and fb_ok
        )
        displacements.append(disp)
        confidences.append(field_confidence)
        accepted_flags.append(accepted)
        diagnostics.append(
            {
                "point": [float(point[0]), float(point[1])],
                "displacement": [float(disp[0]), float(disp[1])],
                "reverse_displacement": [float(reverse[0]), float(reverse[1])],
                "magnitude": mag,
                "edge_fraction": float(edge_fraction),
                "forward_backward_error": fb_error,
                "base_reliability": float(base_rel),
                "match_confidence": float(conf),
                "combined_confidence": combined_conf,
                "field_confidence": field_confidence,
                "accepted": accepted,
                "reject_reasons": [] if accepted else reject_reasons,
            }
        )
    displacements = np.asarray(displacements, dtype=np.float64)
    confidences = np.asarray(confidences, dtype=np.float64)
    keep = np.asarray(accepted_flags, dtype=bool)

    accepted_indices = np.flatnonzero(keep)
    if accepted_indices.size:
        local_keep, local_errors = spatial_consistency_filter(points[accepted_indices], displacements[accepted_indices], confidences[accepted_indices], cfg)
        for local_idx, original_idx in enumerate(accepted_indices):
            diagnostics[original_idx]["spatial_consistency_error"] = float(local_errors[local_idx])
            if not bool(local_keep[local_idx]):
                keep[original_idx] = False
                diagnostics[original_idx]["accepted"] = False
                diagnostics[original_idx]["reject_reasons"].append("spatial_consistency")
        for original_idx in np.flatnonzero(~np.isin(np.arange(len(points)), accepted_indices)):
            diagnostics[original_idx]["spatial_consistency_error"] = None
    return points[keep], displacements[keep], confidences[keep], diagnostics
