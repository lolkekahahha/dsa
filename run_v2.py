import argparse
from pathlib import Path

from dsa_v2 import default_config_for_shape, run_pipeline
from dsa_v2.io import load_grayscale, normalize_to_uint8_range
from dsa_v2.pipeline import save_result


def configure_working_profile(cfg) -> None:
    cfg.field_confidence_floor = min(cfg.field_confidence_floor, 0.12)
    cfg.field_confidence_full = min(cfg.field_confidence_full, 0.48)
    cfg.field_confidence_min_weight = max(cfg.field_confidence_min_weight, 0.78)
    cfg.nonrigid_field_confidence_min_weight = max(cfg.nonrigid_field_confidence_min_weight, 0.90)
    cfg.micro_motion_min_field_reliability = max(cfg.micro_motion_min_field_reliability, 0.90)
    cfg.field_confidence_support_radius = max(cfg.template_size, 2.0 * cfg.control_point_spacing)
    cfg.max_control_points = min(cfg.max_control_points, 180)

    cfg.prefer_nonrigid_registration = True
    cfg.nonrigid_preference_min_points = min(cfg.nonrigid_preference_min_points, 18)
    cfg.nonrigid_preference_min_confidence = min(cfg.nonrigid_preference_min_confidence, 0.24)
    cfg.nonrigid_preference_min_coherence = min(cfg.nonrigid_preference_min_coherence, 0.25)
    cfg.nonrigid_preference_min_variation_ratio = min(cfg.nonrigid_preference_min_variation_ratio, 0.30)

    cfg.vessel_field_strategy = "propagate"
    cfg.vessel_field_protection_strength = min(cfg.vessel_field_protection_strength, 0.55)
    cfg.vessel_field_propagation_strength = 0.85
    cfg.vessel_field_protection_sigma = min(cfg.vessel_field_protection_sigma, 1.25)
    cfg.vascular_bed_motion_attenuation = max(cfg.vascular_bed_motion_attenuation, 0.93)

    cfg.local_improvement_tolerance = max(cfg.local_improvement_tolerance, 0.30)
    cfg.local_improvement_confidence_preserve = max(cfg.local_improvement_confidence_preserve, 2.20)

    cfg.subtraction_intensity_correction_strength = 0.35
    cfg.subtraction_intensity_correction_sigma = max(25.0, cfg.template_size * 0.8)


def build_config(shape):
    cfg = default_config_for_shape(shape)
    configure_working_profile(cfg)
    return cfg


def apply_runtime_profile(cfg, profile: str) -> None:
    if profile == "balanced":
        return
    if profile == "fast":
        cfg.max_control_points = min(cfg.max_control_points, 120)
        cfg.matching_entropy_rerank_top_k = min(cfg.matching_entropy_rerank_top_k, 3)
        cfg.coverage_points_per_cell = 1
        return
    if profile == "quality":
        cfg.max_control_points = max(cfg.max_control_points, 240)
        cfg.matching_entropy_rerank_top_k = max(cfg.matching_entropy_rerank_top_k, 12)
        cfg.coverage_points_per_cell = max(cfg.coverage_points_per_cell, 3)
        cfg.nonrigid_preference_min_confidence = min(cfg.nonrigid_preference_min_confidence, 0.22)
        cfg.nonrigid_preference_min_variation_ratio = min(cfg.nonrigid_preference_min_variation_ratio, 0.24)
        return
    raise ValueError(f"Unknown runtime profile: {profile}")


def resolve_input_path(raw_path: str, root: Path) -> Path:
    cleaned = raw_path.strip().strip('"').strip("'")
    if not cleaned:
        raise ValueError("Image path cannot be empty.")
    path = Path(cleaned).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(f"Image file was not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Path is not a file: {path}")
    return path


def find_input_pair(root: Path) -> tuple[Path, Path]:
    mask_path = resolve_input_path(input("Mask image path: "), root)
    live_path = resolve_input_path(input("Live image path: "), root)
    return mask_path, live_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DSA v2 motion compensation pipeline.")
    parser.add_argument("--mask", type=Path, default=None, help="Path to the pre-contrast mask image.")
    parser.add_argument("--live", type=Path, default=None, help="Path to the contrast/live image.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for diagnostic outputs.")
    parser.add_argument("--profile", choices=("fast", "balanced", "quality"), default="balanced", help="Runtime quality/speed profile.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    if args.mask is not None or args.live is not None:
        if args.mask is None or args.live is None:
            raise ValueError("--mask and --live must be provided together.")
        mask_path = resolve_input_path(str(args.mask), root)
        live_path = resolve_input_path(str(args.live), root)
    else:
        mask_path, live_path = find_input_pair(root)
    output_dir = args.output_dir or root / "results"

    mask = normalize_to_uint8_range(load_grayscale(mask_path))
    live = normalize_to_uint8_range(load_grayscale(live_path))
    if mask.shape != live.shape:
        raise ValueError(f"Input shapes differ: mask={mask.shape}, live={live.shape}")

    cfg = build_config(mask.shape)
    apply_runtime_profile(cfg, args.profile)
    result = run_pipeline(mask, live, cfg)
    save_result(result, output_dir, cfg)

    motion = result.motion_report
    field = motion["displacement_field"]
    coverage = motion["match_coverage"]
    print("DSA v2 completed")
    print(f"mask: {mask_path.name}")
    print(f"live: {live_path.name}")
    print(f"output: {output_dir}")
    print(f"profile: {args.profile}")
    print(f"motion model: {motion['motion_model']['model']}")
    print(f"control points: {motion['control_points']} accepted / {motion['candidate_control_points']} candidates")
    print(f"coverage: {coverage['accepted_cells']}/{coverage['coverage_grid'] ** 2} cells")
    print(f"displacement: mean={field['mean']:.4f}px, p90={field['p90']:.4f}px, max={field['max']:.4f}px")
    print(f"compensation delta in ROI: {result.metrics['result_vs_no_abs_mean_roi']:.4f}")
    print(f"background std improvement: {result.metrics['background_std_improvement']:.4f}")


if __name__ == "__main__":
    main()
