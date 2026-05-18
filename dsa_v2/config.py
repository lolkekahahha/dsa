from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass
class PipelineConfig:
    roi_bbox: Optional[Tuple[int, int, int, int]] = None
    roi_margin: int = 24

    template_size: int = 41
    search_radius: int = 20
    control_point_spacing: int = 30
    min_control_points: int = 30
    max_control_points: int = 220
    max_points_per_cell: int = 2
    coverage_grid: int = 4
    coverage_points_per_cell: int = 2
    enable_texture_control_points: bool = True
    texture_point_stride: int = 24
    texture_point_min_std: float = 5.0
    texture_point_max_vessel_fraction: float = 0.20
    vascular_bed_point_penalty: float = 0.15
    safe_background_min_texture: float = 2.0
    safe_background_max_abs_dsa_percentile: float = 96.0
    safe_background_max_brightness_percentile: float = 99.5
    safe_background_vessel_dilation: int = 2
    safe_mask_min_roi_fraction: float = 0.45
    registration_weight_floor: float = 0.08
    registration_weight_min_for_points: float = 0.18
    enable_region_confidence_weighting: bool = True
    registration_residual_percentile: float = 90.0
    registration_vessel_penalty: float = 0.95
    registration_bed_penalty: float = 0.55
    registration_gradient_similarity_weight: float = 0.35
    registration_use_vessel_support_exclusion: bool = True
    enable_spatial_consistency_filter: bool = True
    spatial_consistency_min_points: int = 8
    spatial_consistency_neighbors: int = 6
    spatial_consistency_tolerance: float = 1.25
    spatial_consistency_mad_factor: float = 3.0
    spatial_consistency_min_keep_fraction: float = 0.60
    prefer_nonrigid_registration: bool = True
    nonrigid_preference_min_points: int = 18
    nonrigid_preference_min_confidence: float = 0.24
    nonrigid_preference_min_coherence: float = 0.25
    nonrigid_preference_min_variation_ratio: float = 0.30
    nonrigid_preference_min_inlier_fraction: float = 0.55

    gaussian_sigma: float = 1.5
    gradient_threshold: float = 0.20
    vessel_exclusion_dilation: int = 4
    vascular_bed_dilation: int = 8
    vessel_mask_existing_structure_penalty: float = 0.60
    vessel_mask_dark_score_percentile: float = 90.0
    vessel_mask_bright_score_percentile: float = 92.0
    vessel_mask_dark_new_signal_percentile: float = 58.0
    vessel_mask_bright_new_signal_percentile: float = 70.0
    vessel_tree_proximity_fraction: float = 0.055
    vessel_support_score_percentile: float = 84.0
    vessel_support_seed_percentile: float = 94.5
    vessel_support_min_component_area: int = 8
    vessel_support_dilation: int = 1
    vessel_support_bright_background_threshold: float = 0.55
    vessel_support_bright_dsa_min: float = 0.24
    vessel_support_bright_raw_min: float = 0.10
    vessel_support_bright_tubular_min: float = 0.12
    vessel_support_clutter_window: int = 21
    vessel_support_clutter_density: float = 0.24
    vessel_support_clutter_brightness: float = 0.42
    enable_vessel_support_mask: bool = True
    enable_vessel_support_clutter_pruning: bool = True
    use_vessel_support_for_compensation: bool = True

    entropy_bins: int = 64
    enable_fast_matching_prefilter: bool = True
    matching_entropy_rerank_top_k: int = 7
    enable_subpixel: bool = True
    enable_coarse_to_fine: bool = False
    coarse_scale: float = 0.5
    fine_refine_radius: int = 3
    intensity_variation_modeling: bool = True
    subtraction_intensity_correction_strength: float = 0.60
    subtraction_intensity_correction_sigma: float = 35.0
    subtraction_intensity_scale_min: float = 0.75
    subtraction_intensity_scale_max: float = 1.35
    subtraction_intensity_min_support: float = 1e-3
    subtraction_intensity_min_support_fraction: float = 0.05
    enable_background_haze_correction: bool = True
    background_haze_correction_strength: float = 0.25
    background_haze_sigma: float = 28.0
    background_haze_vessel_dilation: int = 4
    background_haze_vessel_protection: float = 1.00
    background_haze_min_support_fraction: float = 0.03

    max_displacement: float = 14.0
    search_edge_reject_fraction: float = 0.92
    search_edge_penalty_fraction: float = 0.75
    min_match_confidence: float = 0.30
    min_combined_confidence: float = 0.12
    forward_backward_tolerance: float = 2.0
    enable_forward_backward_check: bool = True
    field_smoothing_sigma: float = 3.0
    field_confidence_floor: float = 0.25
    field_confidence_full: float = 0.68
    field_confidence_gate_gamma: float = 0.65
    field_confidence_min_weight: float = 0.30
    nonrigid_field_confidence_min_weight: float = 0.75
    field_confidence_support_radius: float = 0.0

    enable_global_preregistration: bool = True
    enable_search_shrink_after_global: bool = True
    max_global_shift: float = 14.0
    global_min_response: float = 0.20

    enable_ransac: bool = True
    ransac_iterations: int = 120
    ransac_residual_threshold: float = 2.0
    min_motion_inliers: int = 5
    rigid_residual_threshold: float = 1.0
    affine_residual_threshold: float = 1.25
    affine_min_improvement: float = 0.20
    nonrigid_min_points: int = 12
    nonrigid_min_confidence: float = 0.34
    nonrigid_min_coherence: float = 0.35
    nonrigid_variation_min_magnitude: float = 1.50
    nonrigid_translation_variation_ratio: float = 0.30
    nonrigid_affine_variation_ratio: float = 0.25
    nonrigid_coherent_variation_min: float = 0.50
    micro_motion_max_median: float = 2.5
    micro_motion_min_confidence: float = 0.25
    micro_motion_min_field_reliability: float = 0.65
    micro_motion_rigid_residual_threshold: float = 1.25

    protect_vessels_in_field: bool = True
    vessel_field_strategy: Literal["attenuate", "propagate"] = "attenuate"
    vessel_field_protection_strength: float = 1.0
    vessel_field_propagation_strength: float = 1.0
    vessel_field_protection_sigma: float = 2.0
    vessel_protection_confidence_relief: float = 1.00
    vascular_bed_motion_attenuation: float = 0.85
    enable_local_improvement_gate: bool = True
    local_improvement_tolerance: float = 0.15
    local_improvement_confidence_preserve: float = 1.25

    enable_vessel_display_enhancement: bool = True
    vessel_display_enhancement_strength: float = 0.60
    vessel_display_background_smoothing: float = 0.34
    vessel_display_support_sigma: float = 2.5

    output_window_percentiles: Tuple[float, float] = (1.0, 99.0)


def default_config_for_shape(shape) -> PipelineConfig:
    min_size = min(shape[:2])
    h, w = shape[:2]
    aspect = w / max(h, 1)
    if min_size > 1000:
        return PipelineConfig(
            template_size=61,
            search_radius=25,
            control_point_spacing=50,
            min_control_points=50,
            max_control_points=260,
            max_displacement=18.0,
            field_smoothing_sigma=4.0,
            vessel_exclusion_dilation=5,
            vascular_bed_dilation=10,
            entropy_bins=64,
        )
    if 0.85 <= aspect <= 1.15 and min_size <= 700:
        return PipelineConfig(
            control_point_spacing=20,
            texture_point_stride=16,
            max_points_per_cell=3,
            max_control_points=260,
            min_match_confidence=0.26,
            min_combined_confidence=0.08,
        )
    if min_size > 500:
        return PipelineConfig()
    return PipelineConfig(
        template_size=31,
        search_radius=15,
        control_point_spacing=20,
        min_control_points=15,
        max_control_points=180,
        max_displacement=10.0,
        field_smoothing_sigma=2.0,
        vessel_exclusion_dilation=1,
        vascular_bed_dilation=6,
        entropy_bins=48,
        field_confidence_full=0.65,
    )
