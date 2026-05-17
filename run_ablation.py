import argparse
import csv
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from dsa_v2 import run_pipeline
from dsa_v2.io import dsa_display_window, enhance_dsa_display, load_grayscale, normalize_to_uint8_range, to_uint8_window
from dsa_v2.pipeline import save_result
from run_v2 import apply_runtime_profile, build_config


ROOT = Path(__file__).resolve().parent


PAIRS = {
    "0_coronary": ("0_coronary_mask.tiff", "0_coronary_live.tiff"),
    "coronary_jpg": ("coronary_mask.jpg", "coronary_live.jpg"),
    "cerebral": ("cerebral_mask.tiff", "cerebral_live.tiff"),
}


def variant_current(cfg) -> None:
    return None


def variant_no_vessel_support(cfg) -> None:
    cfg.enable_vessel_support_mask = False


def variant_no_intensity_correction(cfg) -> None:
    cfg.subtraction_intensity_correction_strength = 0.0


def variant_no_haze_correction(cfg) -> None:
    cfg.enable_background_haze_correction = False


def variant_no_nonrigid_preference(cfg) -> None:
    cfg.prefer_nonrigid_registration = False


def variant_rigid_affine_only(cfg) -> None:
    cfg.prefer_nonrigid_registration = False
    cfg.nonrigid_min_points = 10**9


VARIANTS: dict[str, Callable] = {
    "current": variant_current,
    "no_vessel_support": variant_no_vessel_support,
    "no_intensity_correction": variant_no_intensity_correction,
    "no_haze_correction": variant_no_haze_correction,
    "no_nonrigid_preference": variant_no_nonrigid_preference,
    "rigid_affine_only": variant_rigid_affine_only,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comparison for the DSA v2.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "ablation", help="Output directory.")
    parser.add_argument("--profile", choices=("fast", "balanced", "quality"), default="balanced", help="Runtime profile.")
    parser.add_argument("--pairs", nargs="*", choices=tuple(PAIRS.keys()), default=list(PAIRS.keys()), help="Pairs to run.")
    parser.add_argument("--variants", nargs="*", choices=tuple(VARIANTS.keys()), default=list(VARIANTS.keys()), help="Variants to run.")
    return parser.parse_args()


def load_pair(pair_name: str) -> tuple[np.ndarray, np.ndarray]:
    mask_name, live_name = PAIRS[pair_name]
    mask = normalize_to_uint8_range(load_grayscale(ROOT / mask_name))
    live = normalize_to_uint8_range(load_grayscale(ROOT / live_name))
    if mask.shape != live.shape:
        raise ValueError(f"Input shapes differ for {pair_name}: mask={mask.shape}, live={live.shape}")
    return mask, live


def flatten_summary(pair_name: str, variant_name: str, result) -> dict:
    report = result.motion_report
    metrics = result.metrics
    coverage = report["match_coverage"]
    timing = report.get("timings_seconds", {})
    vessel_report = report.get("vessel_detection", {})
    support_roi = vessel_report.get("support_mask_roi", {})
    model = report.get("motion_model", {})
    return {
        "pair": pair_name,
        "variant": variant_name,
        "model": model.get("model", ""),
        "apply_motion": model.get("apply", ""),
        "total_seconds": timing.get("total", 0.0),
        "matching_seconds": timing.get("matching_primary", 0.0),
        "control_points": report.get("control_points", 0),
        "candidate_control_points": report.get("candidate_control_points", 0),
        "coverage_fraction": coverage.get("coverage_fraction", 0.0),
        "background_std_improvement": metrics.get("background_std_improvement", 0.0),
        "background_abs_mean_improvement": metrics.get("background_abs_mean_improvement", 0.0),
        "background_p95_abs_improvement": metrics.get("background_p95_abs_improvement", 0.0),
        "roi_abs_mean_improvement": metrics.get("roi_abs_mean_improvement", 0.0),
        "vessel_abs_signal_ratio": metrics.get("vessel_abs_signal_ratio", 0.0),
        "vessel_contrast_ratio": metrics.get("vessel_contrast_ratio", 0.0),
        "vessel_edge_strength_ratio": metrics.get("vessel_edge_strength_ratio", 0.0),
        "mean_displacement": metrics.get("mean_displacement", 0.0),
        "max_displacement": metrics.get("max_displacement", 0.0),
        "vessel_density_roi": metrics.get("vessel_density_roi", 0.0),
        "support_fraction_roi": support_roi.get("fraction_roi", 0.0),
        "intensity_applied": report.get("intensity_correction", {}).get("applied", False),
        "haze_applied": report.get("background_haze_correction", {}).get("applied", False),
    }


def save_pair_visual(pair_name: str, outputs: list[tuple[str, object]], output_dir: Path) -> None:
    if not outputs:
        return
    images = [outputs[0][1].dsa_no_compensation] + [result.dsa_result for _, result in outputs]
    vmin, vmax = dsa_display_window(*images, percentile=99.0)
    tiles = []
    labels = [("no_compensation", outputs[0][1].dsa_no_compensation)] + [(name, result.dsa_result) for name, result in outputs]
    for label, image in labels:
        tile = enhance_dsa_display(to_uint8_window(image, vmin, vmax))
        tile_bgr = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(tile_bgr, (0, 0), (min(tile_bgr.shape[1] - 1, 270), 25), (0, 0, 0), -1)
        cv2.putText(tile_bgr, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile_bgr)

    columns = min(4, len(tiles))
    rows = int(np.ceil(len(tiles) / columns))
    h, w = tiles[0].shape[:2]
    canvas = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        y = (idx // columns) * h
        x = (idx % columns) * w
        canvas[y : y + h, x : x + w] = tile
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{pair_name}_visual_comparison.png"), canvas)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for pair_name in args.pairs:
        mask, live = load_pair(pair_name)
        pair_outputs = []
        for variant_name in args.variants:
            cfg = build_config(mask.shape)
            apply_runtime_profile(cfg, args.profile)
            VARIANTS[variant_name](cfg)
            output_dir = args.output_root / pair_name / variant_name
            result = run_pipeline(mask, live, cfg)
            save_result(result, output_dir, cfg)
            pair_outputs.append((variant_name, result))
            row = flatten_summary(pair_name, variant_name, result)
            rows.append(row)
            print(
                f"{pair_name}/{variant_name}: "
                f"bg_std={row['background_std_improvement']:.4f}, "
                f"bg_mean={row['background_abs_mean_improvement']:.4f}, "
                f"vessel_signal={row['vessel_abs_signal_ratio']:.4f}, "
                f"time={row['total_seconds']:.2f}s"
            )
        save_pair_visual(pair_name, pair_outputs, args.output_root / pair_name)

    summary_path = args.output_root / "summary.csv"
    if rows:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
