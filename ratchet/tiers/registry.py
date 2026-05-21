"""The canonical TIERS registry — single source of truth for tier definitions.

Holds eight named Hardware instances representing the silicon classes the
ecosystem sizes against. Surfaces compose their visible ladders by selecting
from the registry; they don't define new tiers (except via make_custom_tier()).
"""
from ratchet.calibration.source import CalibrationSource
from ratchet.precision.capability import (
    NEUTRON_INT8_ONLY_CAPABILITY,
    NPU_FULL_DTYPE_CAPABILITY,
    SM120_BLACKWELL_CAPABILITY,
)
from ratchet.tiers.hardware import Hardware

# ─── Calibration source constants ───────────────────────────────────

_CANONICAL_SPEC_2026_05_14 = CalibrationSource(
    method="vendor_spec",
    reference="PAI deck Slide 11 (canonical, locked 2026-05-14)",
    confidence="high",
)

_BAKEOFF_MEASURED_2026_04 = CalibrationSource(
    method="measured",
    reference="5090 bake-off + Mid silicon measurements, 2026-04",
    confidence="high",
)

_IMX95_PRODUCTION = CalibrationSource(
    method="measured",
    reference="i.MX 95 production silicon, Kyle's NXP eIQ measurement",
    confidence="high",
)


# ─── Tier definitions ───────────────────────────────────────────────

NPU_LOW_LP4 = Hardware(
    name="NPU Low-LP4",
    peak_tops_bf16=0.0, peak_tops_int8=2.0, peak_tops_fp8=0.0,
    mem_bandwidth_gbs=17.064, mem_capacity_gb=8.0,
    mem_bus_width_bits=32, mem_type="LPDDR4", mem_data_rate_gtps=4.266,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=10.0,
    tier_family="Neutron-32-LP4",
    compute_util_factor=0.19,
    llm_prefill_util_factor=0.10,
    capability_levels=NEUTRON_INT8_ONLY_CAPABILITY,
    calibration_source=_CANONICAL_SPEC_2026_05_14,
)
"""The cheapest edge NPU class: 2-TOPS INT8-only silicon on 32-bit LPDDR4."""


NPU_LOW_LP5_32BIT = Hardware(
    name="NPU Low-LP5-32bit",
    peak_tops_bf16=0.0, peak_tops_int8=2.0, peak_tops_fp8=0.0,
    mem_bandwidth_gbs=25.6, mem_capacity_gb=16.0,
    mem_bus_width_bits=32, mem_type="LPDDR5", mem_data_rate_gtps=6.4,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=15.0,
    tier_family="Neutron-32-LP5",
    compute_util_factor=0.19,
    llm_prefill_util_factor=0.10,
    capability_levels=NEUTRON_INT8_ONLY_CAPABILITY,
    calibration_source=_CANONICAL_SPEC_2026_05_14,
)


NPU_LOW_LP5_64BIT = Hardware(
    name="NPU Low-LP5-64bit",
    peak_tops_bf16=0.0, peak_tops_int8=2.0, peak_tops_fp8=0.0,
    mem_bandwidth_gbs=51.2, mem_capacity_gb=16.0,
    mem_bus_width_bits=64, mem_type="LPDDR5", mem_data_rate_gtps=6.4,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=20.0,
    tier_family="Neutron-64-LP5",
    compute_util_factor=0.19,
    llm_prefill_util_factor=0.10,
    capability_levels=NEUTRON_INT8_ONLY_CAPABILITY,
    calibration_source=_CANONICAL_SPEC_2026_05_14,
)


NPU_LOW_LP5X = Hardware(
    name="NPU Low-LP5X",
    peak_tops_bf16=50.0, peak_tops_int8=100.0, peak_tops_fp8=100.0,
    mem_bandwidth_gbs=67.2, mem_capacity_gb=16.0,
    mem_bus_width_bits=64, mem_type="LPDDR5X", mem_data_rate_gtps=8.4,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=10.0,
    tier_family="LP5X-8.4-64b",
    compute_util_factor=0.19,
    llm_prefill_util_factor=0.10,
    capability_levels=NPU_FULL_DTYPE_CAPABILITY,
    calibration_source=_CANONICAL_SPEC_2026_05_14,
)


IMX95_MEASURED = Hardware(
    name="NPU i.MX 95 (ground truth)",
    peak_tops_bf16=0.0, peak_tops_int8=2.0, peak_tops_fp8=0.0,
    mem_bandwidth_gbs=25.6, mem_capacity_gb=16.0,
    mem_bus_width_bits=32, mem_type="LPDDR5", mem_data_rate_gtps=6.4,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=8.0,
    tier_family="Neutron-32-LP5",
    compute_util_factor=0.19,
    llm_prefill_util_factor=0.10,
    capability_levels=NEUTRON_INT8_ONLY_CAPABILITY,
    calibration_source=_IMX95_PRODUCTION,
    measured_vision_overrides={
        "yolov8n_trt_int8_coco128": {
            "1920x1080": {"ms_per_inference": 32.0, "fps": 31.25},
        },
    },
)


NPU_MID = Hardware(
    name="NPU Mid",
    peak_tops_bf16=0.0, peak_tops_int8=200.0, peak_tops_fp8=0.0,
    mem_bandwidth_gbs=134.4, mem_capacity_gb=24.0,
    mem_bus_width_bits=128, mem_type="LPDDR5X", mem_data_rate_gtps=8.4,
    compute_efficiency=0.65, bandwidth_efficiency=0.70,
    tdp_watts=25.0,
    tier_family="LP5X-8.4-128b",
    compute_util_factor=0.45,
    llm_prefill_util_factor=0.10,
    capability_levels=NEUTRON_INT8_ONLY_CAPABILITY,
    calibration_source=_BAKEOFF_MEASURED_2026_04,
    measured_decode_overrides={"qwen3_30b_a3b_moe": 37.85},
    measured_prefill_overrides={"qwen3_30b_a3b_moe": 2849.0},
)
"""200 eTOPS INT8-only on 128-bit LPDDR5X-8.4. Carries Skippy MoE Q4_K_M anchor."""


NPU_HIGH = Hardware(
    name="NPU High",
    peak_tops_bf16=200.0, peak_tops_int8=400.0, peak_tops_fp8=400.0,
    mem_bandwidth_gbs=134.4, mem_capacity_gb=32.0,
    mem_bus_width_bits=128, mem_type="LPDDR5X", mem_data_rate_gtps=8.4,
    compute_efficiency=0.70, bandwidth_efficiency=0.70,
    tdp_watts=40.0,
    tier_family="LP5X-8.4-128b",
    compute_util_factor=0.50,
    llm_prefill_util_factor=0.11,
    capability_levels=NPU_FULL_DTYPE_CAPABILITY,
    calibration_source=_BAKEOFF_MEASURED_2026_04,
)


RTX_5090_REFERENCE = Hardware(
    name="RTX 5090 (reference, measured)",
    peak_tops_bf16=209.0, peak_tops_int8=419.0, peak_tops_fp8=419.0,
    mem_bandwidth_gbs=1792.0, mem_capacity_gb=32.0,
    mem_bus_width_bits=512, mem_type="GDDR7", mem_data_rate_gtps=28.0,
    compute_efficiency=0.70, bandwidth_efficiency=0.85,
    tdp_watts=575.0,
    tier_family="GDDR7-28",
    compute_util_factor=0.85,
    llm_prefill_util_factor=0.10,
    compute_overhead_ms=0.3,
    npu_share_default=1.0,
    capability_levels=SM120_BLACKWELL_CAPABILITY,
    calibration_source=_BAKEOFF_MEASURED_2026_04,
    measured_llm={},  # Populated at import time by surface bundle loaders
    measured_vision_overrides={},
)


# ─── The registry dict ──────────────────────────────────────────────

TIERS: dict[str, Hardware] = {
    t.name: t for t in (
        NPU_LOW_LP4,
        NPU_LOW_LP5_32BIT,
        NPU_LOW_LP5_64BIT,
        NPU_LOW_LP5X,
        IMX95_MEASURED,
        NPU_MID,
        NPU_HIGH,
        RTX_5090_REFERENCE,
    )
}
