from dataclasses import dataclass
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import csv

import cv2
import numpy as np

from .config import PipelineConfig
from .diagnostics import annotate_match_regions, region_summary
from .features import tissue_importance, select_control_points
from .field import (
    apply_confidence_gate,
    apply_soft_roi,
    build_model_field,
    confidence_map,
    local_improvement_gate,
    protect_vessel_field,
    warp_image,
)
from .io import dsa_display_window, enhance_dsa_display, ensure_dir, save_uint8, to_uint8_window
from .masks import (
    compute_registration_safe_mask,
    compute_registration_weight_map,
    compute_roi,
    compute_vessel_support_region,
    detect_vessel_support_mask,
    detect_vascular_bed,
    detect_vessels,
)
from .matching import estimate_displacements
from .motion import analyze_motion, filter_outliers, select_motion_profile
from .subtraction import basic_metrics, correct_background_haze, correct_warped_mask_intensity, subtract_live_minus_mask


@dataclass
class PipelineResult:
    diagnostic_mask: np.ndarray
    diagnostic_live: np.ndarray
    dsa_result: np.ndarray
    dsa_geometric: np.ndarray
    dsa_no_compensation: np.ndarray
    warped_mask: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    roi_mask: np.ndarray
    vessel_mask: np.ndarray
    vessel_support_mask: np.ndarray
    vascular_bed_mask: np.ndarray
    safe_mask: np.ndarray
    registration_weight: np.ndarray
    confidence_map: np.ndarray
    local_gate: np.ndarray
    motion_report: dict
    metrics: dict
    match_diagnostics: list[dict]


def estimate_global_shift(
    mask: np.ndarray,
    live: np.ndarray,
    roi: np.ndarray,
    vessels: np.ndarray,
    bed: np.ndarray,
    cfg: PipelineConfig,
    registration_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    if not cfg.enable_global_preregistration:
        return np.zeros(2, dtype=np.float64), 0.0
    fixed = cv2.normalize(live.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)
    moving = cv2.normalize(mask.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)
    safe = roi.astype(bool) & ~vessels.astype(bool) & ~bed.astype(bool)
    if np.mean(safe) < 0.05:
        safe = roi.astype(bool) & ~vessels.astype(bool)
    weights = safe.astype(np.float64)
    if registration_weight is not None:
        weighted = weights * np.clip(registration_weight.astype(np.float64), 0.0, 1.0)
        if np.sum(weighted > 0.05) >= max(64, 0.02 * np.sum(roi)):
            weights = weighted
    hann = cv2.createHanningWindow((mask.shape[1], mask.shape[0]), cv2.CV_64F)
    try:
        shift, response = cv2.phaseCorrelate((moving * weights * hann).astype(np.float64), (fixed * weights * hann).astype(np.float64))
    except cv2.error:
        return np.zeros(2, dtype=np.float64), 0.0
    response = float(np.clip(response, 0.0, 1.0))
    dx, dy = float(shift[0]), float(shift[1])
    mag = np.sqrt(dx * dx + dy * dy)
    if mag > cfg.max_global_shift:
        scale = cfg.max_global_shift / (mag + 1e-8)
        dx *= scale
        dy *= scale
    gate = np.clip((response - cfg.global_min_response) / (0.55 - cfg.global_min_response + 1e-8), 0.0, 1.0)
    return np.array([dx * gate, dy * gate], dtype=np.float64), response


def estimate_global_preregistration(
    mask: np.ndarray,
    live: np.ndarray,
    roi: np.ndarray,
    vessels: np.ndarray,
    bed: np.ndarray,
    cfg: PipelineConfig,
    registration_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    shift, response = estimate_global_shift(mask, live, roi, vessels, bed, cfg, registration_weight)
    dx = np.full_like(mask, shift[0], dtype=np.float64)
    dy = np.full_like(mask, shift[1], dtype=np.float64)
    report = {
        "method": "phase_correlation",
        "applied": bool(cfg.enable_global_preregistration),
        "response": float(response),
        "shift": [float(shift[0]), float(shift[1])],
    }
    return dx, dy, report


def accepted_match_coverage(match_diagnostics: list[dict], shape, cfg: PipelineConfig) -> dict:
    h, w = shape
    grid_n = max(2, int(cfg.coverage_grid))
    accepted = [m for m in match_diagnostics if m.get("accepted")]
    cells = set()
    for item in accepted:
        x, y = item["point"]
        gy = min(grid_n - 1, max(0, int(float(y) / max(1, h) * grid_n)))
        gx = min(grid_n - 1, max(0, int(float(x) / max(1, w) * grid_n)))
        cells.add((gy, gx))
    return {
        "accepted_count": int(len(accepted)),
        "accepted_cells": int(len(cells)),
        "coverage_grid": int(grid_n),
        "coverage_fraction": float(len(cells) / float(grid_n * grid_n)),
    }


def binary_mask_summary(mask: np.ndarray, roi: np.ndarray | None = None) -> dict:
    region = roi.astype(bool) if roi is not None and np.any(roi) else np.ones_like(mask, dtype=bool)
    mask_region = mask.astype(bool) & region
    components, _, stats, _ = cv2.connectedComponentsWithStats(mask_region.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if components > 1 else np.asarray([], dtype=np.int32)
    return {
        "pixels": int(np.sum(mask_region)),
        "fraction_roi": float(np.mean(mask_region[region])) if np.any(region) else 0.0,
        "components": int(max(0, components - 1)),
        "largest_component_pixels": int(np.max(areas)) if len(areas) else 0,
    }


def scalar_map_summary(values: np.ndarray, roi: np.ndarray | None = None) -> dict:
    region = roi.astype(bool) if roi is not None and np.any(roi) else np.ones_like(values, dtype=bool)
    selected = values.astype(np.float64)[region]
    if selected.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(selected)),
        "p50": float(np.percentile(selected, 50)),
        "p90": float(np.percentile(selected, 90)),
        "p99": float(np.percentile(selected, 99)),
        "max": float(np.max(selected)),
    }


def field_summary(dx: np.ndarray, dy: np.ndarray, roi: np.ndarray | None = None) -> dict:
    return scalar_map_summary(np.sqrt(dx * dx + dy * dy), roi)


def field_reliability_from_confidences(confidences: np.ndarray, decision, cfg: PipelineConfig, motion_profile: str) -> float:
    if len(confidences) == 0 or not decision.apply:
        return 0.0
    reliability = np.clip(
        (float(np.mean(confidences)) - max(0.30, cfg.field_confidence_floor))
        / (cfg.field_confidence_full - max(0.30, cfg.field_confidence_floor) + 1e-8),
        0.0,
        1.0,
    )
    if motion_profile == "micro_motion":
        reliability = max(reliability, cfg.micro_motion_min_field_reliability)
    if motion_profile != "micro_motion" and decision.model in {"rigid", "affine"}:
        reliability *= 0.85
    return float(reliability)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def run_pipeline(mask: np.ndarray, live: np.ndarray, cfg: PipelineConfig) -> PipelineResult:
    timings = {}
    pipeline_start = time.perf_counter()
    stage_start = pipeline_start
    roi = compute_roi(mask, live, cfg)
    vessels = detect_vessels(live, mask, roi, cfg)
    vessel_support_region = compute_vessel_support_region(live.shape, roi, cfg)
    if cfg.enable_vessel_support_mask:
        vessel_support = detect_vessel_support_mask(live, mask, roi, vessels, cfg)
    else:
        vessel_support = vessels.astype(bool) & vessel_support_region
    bed = detect_vascular_bed(live, mask, vessels, roi, cfg)
    safe_mask = compute_registration_safe_mask(mask, live, roi, vessels, bed, cfg)
    registration_weight = compute_registration_weight_map(mask, live, roi, vessels, bed, cfg)
    compensation_vessels = vessel_support if cfg.use_vessel_support_for_compensation else vessels
    registration_exclusion_mask = vessel_support if cfg.registration_use_vessel_support_exclusion else vessels
    registration_candidate_roi = (
        roi.astype(bool)
        & (registration_weight >= cfg.registration_weight_min_for_points)
        & ~registration_exclusion_mask.astype(bool)
        & ~bed.astype(bool)
    )
    timings["masks"] = time.perf_counter() - stage_start
    point_vessels = vessels
    if np.any(roi) and float(np.mean(vessels[roi])) > 0.06:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        point_vessels = cv2.erode(vessels.astype(np.uint8), kernel, iterations=1).astype(bool)
    safe_fraction = float(np.mean(safe_mask[roi])) if np.any(roi) else 0.0
    effective_safe_mask = safe_mask if safe_fraction >= cfg.safe_mask_min_roi_fraction else None

    global_dx, global_dy, global_report = estimate_global_preregistration(mask, live, roi, vessels, bed, cfg, registration_weight)
    timings["global_preregistration"] = time.perf_counter() - stage_start - timings["masks"]
    global_shift = np.asarray(global_report["shift"], dtype=np.float64)
    global_response = float(global_report["response"])
    global_magnitude = float(np.linalg.norm(global_shift))
    match_cfg = cfg
    if cfg.enable_search_shrink_after_global and global_response >= 0.75 and global_magnitude <= 1.0:
        match_cfg = replace(cfg, search_radius=min(cfg.search_radius, 8), max_displacement=min(cfg.max_displacement, 6.0))
    field_stages = {"global_shift": field_summary(global_dx, global_dy, roi)}
    prereg_mask = warp_image(mask, global_dx, global_dy)

    importance = tissue_importance(prereg_mask, cfg)
    points, reliabilities = select_control_points(
        prereg_mask,
        point_vessels,
        importance,
        match_cfg,
        roi,
        bed,
        safe_mask=None,
        registration_weight=registration_weight,
    )
    candidate_control_points = int(len(points))
    stage_start = time.perf_counter()
    points, displacements, confidences, match_diagnostics = estimate_displacements(
        prereg_mask,
        live,
        points,
        reliabilities,
        registration_exclusion_mask,
        roi,
        match_cfg,
        safe_mask=effective_safe_mask,
        registration_weight=registration_weight,
    )
    timings["matching_primary"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    match_diagnostics = annotate_match_regions(
        match_diagnostics,
        prereg_mask,
        live,
        vessels,
        bed,
        roi,
        safe_mask,
        registration_weight,
        match_cfg,
    )
    accepted_region_weights = np.asarray(
        [item.get("region_motion_weight", 1.0) for item in match_diagnostics if item.get("accepted")],
        dtype=np.float64,
    )
    if match_cfg.enable_region_confidence_weighting and len(accepted_region_weights) == len(confidences):
        confidences = confidences * accepted_region_weights
        accepted_idx = 0
        for item in match_diagnostics:
            if not item.get("accepted"):
                continue
            raw_field_confidence = float(item.get("field_confidence", 0.0))
            region_weight = float(accepted_region_weights[accepted_idx])
            item["region_weighted_field_confidence"] = raw_field_confidence * region_weight
            accepted_idx += 1
    else:
        for item in match_diagnostics:
            item["region_weighted_field_confidence"] = item.get("field_confidence", 0.0)
    points, displacements, confidences = filter_outliers(points, displacements, confidences, match_cfg)
    decision = analyze_motion(points, displacements, confidences, match_cfg)
    motion_profile = select_motion_profile(decision, global_shift, match_cfg)
    timings["motion_model"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    local_dx, local_dy = build_model_field(points, displacements, confidences, decision, mask.shape, match_cfg)
    field_stages["model_field"] = field_summary(local_dx, local_dy, roi)
    local_dx, local_dy = apply_soft_roi(local_dx, local_dy, roi, match_cfg)
    field_stages["soft_roi"] = field_summary(local_dx, local_dy, roi)
    conf_map = confidence_map(mask.shape, points, confidences, local_dx, local_dy, roi, match_cfg)
    local_dx, local_dy, reliability_weight = apply_confidence_gate(
        local_dx,
        local_dy,
        conf_map,
        match_cfg,
        motion_profile == "micro_motion",
        decision.model == "nonrigid",
    )
    field_stages["confidence_gate"] = field_summary(local_dx, local_dy, roi)
    global_field_reliability = field_reliability_from_confidences(confidences, decision, match_cfg, motion_profile)
    local_dx *= global_field_reliability
    local_dy *= global_field_reliability
    field_stages["global_reliability"] = field_summary(local_dx, local_dy, roi)

    timings["field_primary"] = time.perf_counter() - stage_start

    field_reliability = global_field_reliability

    stage_start = time.perf_counter()
    dx = global_dx + local_dx
    dy = global_dy + local_dy
    field_stages["combined_before_protection"] = field_summary(dx, dy, roi)
    dx, dy = protect_vessel_field(dx, dy, compensation_vessels, bed, match_cfg, reliability_weight)
    field_stages["vessel_protection"] = field_summary(dx, dy, roi)
    dx, dy, gate = local_improvement_gate(mask, live, dx, dy, roi, compensation_vessels, match_cfg, reliability_weight)
    field_stages["local_improvement_gate"] = field_summary(dx, dy, roi)

    warped = warp_image(mask, dx, dy)
    dsa_no = subtract_live_minus_mask(live, mask)
    dsa_geometric = subtract_live_minus_mask(live, warped)
    geometric_metrics = basic_metrics(dsa_no, dsa_geometric, roi, compensation_vessels, dx, dy)
    intensity_weights = roi.astype(np.float64)
    intensity_weights *= (~compensation_vessels.astype(bool)).astype(np.float64)
    intensity_weights *= np.clip(registration_weight.astype(np.float64), 0.0, 1.0)
    if not decision.apply:
        intensity_weights *= 0.0
    warped, intensity_correction = correct_warped_mask_intensity(warped, live, intensity_weights, match_cfg)
    dsa_result = subtract_live_minus_mask(live, warped)
    dsa_result, haze_correction = correct_background_haze(
        dsa_result,
        roi,
        vessels,
        np.clip(registration_weight.astype(np.float64), 0.0, 1.0),
        match_cfg,
    )
    metrics = basic_metrics(dsa_no, dsa_result, roi, compensation_vessels, dx, dy)
    displacement_magnitude = np.sqrt(dx * dx + dy * dy)
    timings["subtraction"] = time.perf_counter() - stage_start
    timings["total"] = time.perf_counter() - pipeline_start

    motion_report = {
        "timings_seconds": {key: float(value) for key, value in timings.items()},
        "motion_model": decision.to_dict(),
        "motion_profile": motion_profile,
        "vessel_detection": {
            "method": "contrast_appearance",
            "mask": binary_mask_summary(vessels, roi),
            "support_mask": binary_mask_summary(vessel_support, None),
            "support_mask_roi": binary_mask_summary(vessel_support, roi),
            "vascular_bed": binary_mask_summary(bed, roi),
            "support_enabled": bool(match_cfg.enable_vessel_support_mask),
            "support_used_for_compensation": bool(match_cfg.use_vessel_support_for_compensation),
            "clutter_pruning_enabled": bool(match_cfg.enable_vessel_support_clutter_pruning),
            "used_for_registration_exclusion": bool(match_cfg.registration_use_vessel_support_exclusion),
        },
        "roi_roles": {
            "analysis_roi": binary_mask_summary(roi, None),
            "registration_candidate_roi": binary_mask_summary(registration_candidate_roi, roi),
            "registration_safe_roi": binary_mask_summary(safe_mask, roi),
            "vessel_detection_region": binary_mask_summary(vessel_support_region, None),
            "motion_exclusion_mask": binary_mask_summary(vessels, roi),
            "vessel_support_mask": binary_mask_summary(vessel_support, roi),
            "compensation_vessel_mask": binary_mask_summary(compensation_vessels, roi),
            "background_metric_region": binary_mask_summary(roi & ~compensation_vessels, roi),
        },
        "global_shift": [float(global_shift[0]), float(global_shift[1])],
        "global_response": float(global_response),
        "global_preregistration": global_report,
        "effective_search_radius": int(match_cfg.search_radius),
        "effective_max_displacement": float(match_cfg.max_displacement),
        "field_reliability": float(field_reliability),
        "global_field_reliability": float(global_field_reliability),
        "field_stages": field_stages,
        "displacement_field": scalar_map_summary(displacement_magnitude, roi),
        "displacement_field_vessels": scalar_map_summary(displacement_magnitude, roi & vessel_support),
        "displacement_field_background": scalar_map_summary(displacement_magnitude, roi & ~vessel_support),
        "compensation_delta": scalar_map_summary(np.abs(dsa_result - dsa_no), roi),
        "candidate_control_points": candidate_control_points,
        "control_points": int(len(points)),
        "mean_point_confidence": float(np.mean(confidences)) if len(confidences) else 0.0,
        "match_coverage": accepted_match_coverage(match_diagnostics, mask.shape, cfg),
        "safe_mask_coverage_roi": float(np.mean(safe_mask[roi])) if np.any(roi) else 0.0,
        "safe_mask_used_for_global_matching": bool(effective_safe_mask is not None),
        "registration_weight_mean_roi": float(np.mean(registration_weight[roi])) if np.any(roi) else 0.0,
        "registration_weight_p50_roi": float(np.percentile(registration_weight[roi], 50)) if np.any(roi) else 0.0,
        "registration_weight_p90_roi": float(np.percentile(registration_weight[roi], 90)) if np.any(roi) else 0.0,
        "point_region_summary": region_summary(match_diagnostics),
        "accepted_region_motion_weight_mean": float(np.mean(accepted_region_weights)) if len(accepted_region_weights) else 0.0,
        "intensity_correction": intensity_correction,
        "background_haze_correction": haze_correction,
        "geometric_only_metrics": geometric_metrics,
        "protection": {
            "enable_search_shrink_after_global": bool(match_cfg.enable_search_shrink_after_global),
            "protect_vessels_in_field": bool(match_cfg.protect_vessels_in_field),
            "vessel_field_strategy": match_cfg.vessel_field_strategy,
            "vessel_field_protection_strength": float(match_cfg.vessel_field_protection_strength),
            "vessel_field_propagation_strength": float(match_cfg.vessel_field_propagation_strength),
            "subtraction_intensity_correction_strength": float(match_cfg.subtraction_intensity_correction_strength),
            "background_haze_correction_strength": float(match_cfg.background_haze_correction_strength),
            "vascular_bed_motion_attenuation": float(match_cfg.vascular_bed_motion_attenuation),
            "enable_local_improvement_gate": bool(match_cfg.enable_local_improvement_gate),
            "enable_spatial_consistency_filter": bool(match_cfg.enable_spatial_consistency_filter),
            "registration_use_vessel_support_exclusion": bool(match_cfg.registration_use_vessel_support_exclusion),
            "prefer_nonrigid_registration": bool(match_cfg.prefer_nonrigid_registration),
            "field_confidence_floor": float(match_cfg.field_confidence_floor),
            "field_confidence_full": float(match_cfg.field_confidence_full),
            "field_confidence_min_weight": float(match_cfg.field_confidence_min_weight),
            "nonrigid_field_confidence_min_weight": float(match_cfg.nonrigid_field_confidence_min_weight),
        },
    }

    return PipelineResult(
        diagnostic_mask=prereg_mask,
        diagnostic_live=live,
        dsa_result=dsa_result,
        dsa_geometric=dsa_geometric,
        dsa_no_compensation=dsa_no,
        warped_mask=warped,
        dx=dx,
        dy=dy,
        roi_mask=roi,
        vessel_mask=vessels,
        vessel_support_mask=vessel_support,
        vascular_bed_mask=bed,
        safe_mask=safe_mask,
        registration_weight=registration_weight,
        confidence_map=conf_map,
        local_gate=gate,
        motion_report=motion_report,
        metrics=metrics,
        match_diagnostics=match_diagnostics,
    )


def save_result(result: PipelineResult, output_dir: str | Path, cfg: PipelineConfig) -> None:
    out = ensure_dir(output_dir)
    managed_outputs = [
        "dsa_no_compensation.png",
        "dsa_compensated.png",
        "vessel_support_overlay.png",
        "displacement_magnitude.png",
        "match_vectors.png",
        "motion_report.json",
        "summary.csv",
        "dsa_v2_compensated_shared_window.png",
        "dsa_v2_no_compensation_shared_window.png",
        "dsa_v2_compensation_delta.png",
        "dsa_v2_displacement_magnitude.png",
        "dsa_v2_vessel_overlay.png",
        "dsa_v2_vessel_support_overlay.png",
        "dsa_v2_match_vectors.png",
        "dsa_v2_motion_report.json",
        "dsa_v2_geometric_compensated_shared_window.png",
        "dsa_v2_warped_mask_delta.png",
        "dsa_v2_vessel_mask.png",
        "dsa_v2_registration_safe_mask.png",
        "dsa_v2_point_pairs.csv",
    ]
    for name in managed_outputs:
        path = out / name
        if path.exists():
            path.unlink()

    vmin, vmax = dsa_display_window(
        result.dsa_no_compensation,
        result.dsa_result,
        percentile=cfg.output_window_percentiles[1],
    )
    compensated_display = enhance_dsa_display(to_uint8_window(result.dsa_result, vmin, vmax))
    no_compensation_display = enhance_dsa_display(to_uint8_window(result.dsa_no_compensation, vmin, vmax))
    save_uint8(out / "dsa_compensated.png", compensated_display)
    save_uint8(out / "dsa_no_compensation.png", no_compensation_display)

    vessel_base = cv2.cvtColor(
        to_uint8_window(result.diagnostic_live, float(np.min(result.diagnostic_live)), float(np.max(result.diagnostic_live))),
        cv2.COLOR_GRAY2BGR,
    )
    vessel_support_overlay = vessel_base.copy()
    vessel_support_overlay[result.vessel_support_mask.astype(bool)] = (0, 0, 220)
    cv2.imwrite(
        str(out / "vessel_support_overlay.png"),
        cv2.addWeighted(vessel_base, 0.72, vessel_support_overlay, 0.28, 0.0),
    )

    mag = np.sqrt(result.dx * result.dx + result.dy * result.dy)
    save_uint8(out / "displacement_magnitude.png", to_uint8_window(mag, 0.0, max(1.0, float(np.percentile(mag, 99)))))

    debug = cv2.cvtColor(no_compensation_display, cv2.COLOR_GRAY2BGR)
    for item in result.match_diagnostics:
        x, y = item["point"]
        dx, dy = item["displacement"]
        accepted = bool(item["accepted"])
        color = (0, 220, 0) if accepted else (0, 0, 255)
        p0 = (int(round(x)), int(round(y)))
        p1 = (int(round(x + dx)), int(round(y + dy)))
        cv2.circle(debug, p0, 2, color, -1)
        cv2.arrowedLine(debug, p0, p1, color, 1, tipLength=0.25)
    cv2.imwrite(str(out / "match_vectors.png"), debug)

    report = {
        "motion_report": result.motion_report,
        "metrics": result.metrics,
    }
    with (out / "motion_report.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(report), handle, ensure_ascii=False, indent=2)

    summary = {
        "motion_model": result.motion_report["motion_model"]["model"],
        "control_points": result.motion_report["control_points"],
        "candidate_control_points": result.motion_report["candidate_control_points"],
        "coverage_fraction": result.motion_report["match_coverage"]["coverage_fraction"],
        "total_seconds": result.motion_report["timings_seconds"]["total"],
        "background_std_improvement": result.metrics["background_std_improvement"],
        "background_abs_mean_improvement": result.metrics["background_abs_mean_improvement"],
        "background_p95_abs_improvement": result.metrics["background_p95_abs_improvement"],
        "vessel_abs_signal_ratio": result.metrics["vessel_abs_signal_ratio"],
        "vessel_contrast_ratio": result.metrics["vessel_contrast_ratio"],
        "vessel_edge_strength_ratio": result.metrics["vessel_edge_strength_ratio"],
        "mean_displacement": result.metrics["mean_displacement"],
        "max_displacement": result.metrics["max_displacement"],
    }
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
