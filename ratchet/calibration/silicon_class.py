"""Silicon-class default calibration tables backing make_custom_tier().

Default compute_util_factor=0.45 (NPU Mid) would over-project a 2-TOPS Neutron
chip by ~2.5×. The silicon_class parameter selects family-appropriate defaults
so custom tiers don't silently inherit optimistic calibration.
"""
from typing import Any

from ratchet.precision.capability import (
    NEUTRON_INT8_ONLY_CAPABILITY,
    NPU_FULL_DTYPE_CAPABILITY,
    SM120_BLACKWELL_CAPABILITY,
)

SILICON_CLASS_DEFAULTS: dict[str, dict[str, Any]] = {
    "neutron": {
        "compute_efficiency": 0.60,
        "bandwidth_efficiency": 0.70,
        "tier_family": "Neutron-custom",
        "compute_util_factor": 0.19,
        "llm_prefill_util_factor": 0.10,
        "llm_decode_bw_realization": 1.0,
        "compute_overhead_ms": 1.0,
        "npu_share_default": 0.75,
        "capability_levels": NEUTRON_INT8_ONLY_CAPABILITY,
    },
    "lp5x_64": {
        "compute_efficiency": 0.60,
        "bandwidth_efficiency": 0.70,
        "tier_family": "LP5X-custom-64b",
        "compute_util_factor": 0.19,
        "llm_prefill_util_factor": 0.10,
        "llm_decode_bw_realization": 1.0,
        "compute_overhead_ms": 1.0,
        "npu_share_default": 0.75,
        "capability_levels": NPU_FULL_DTYPE_CAPABILITY,
    },
    "lp5x_128": {
        "compute_efficiency": 0.70,
        "bandwidth_efficiency": 0.70,
        "tier_family": "LP5X-custom-128b",
        "compute_util_factor": 0.50,
        "llm_prefill_util_factor": 0.11,
        "llm_decode_bw_realization": 1.0,
        "compute_overhead_ms": 1.0,
        "npu_share_default": 0.75,
        "capability_levels": NPU_FULL_DTYPE_CAPABILITY,
    },
    "lp5x_128_int8": {
        "compute_efficiency": 0.65,
        "bandwidth_efficiency": 0.70,
        "tier_family": "LP5X-custom-128b-int8",
        "compute_util_factor": 0.45,
        "llm_prefill_util_factor": 0.10,
        "llm_decode_bw_realization": 1.0,
        "compute_overhead_ms": 1.0,
        "npu_share_default": 0.75,
        "capability_levels": NEUTRON_INT8_ONLY_CAPABILITY,
    },
    "gddr_class": {
        "compute_efficiency": 0.70,
        "bandwidth_efficiency": 0.85,
        "tier_family": "GDDR-custom",
        "compute_util_factor": 0.85,
        "llm_prefill_util_factor": 0.10,
        "llm_decode_bw_realization": 1.0,
        "compute_overhead_ms": 0.3,
        "npu_share_default": 1.0,
        "capability_levels": SM120_BLACKWELL_CAPABILITY,
    },
    "unknown": {
        "compute_efficiency": 0.65,
        "bandwidth_efficiency": 0.70,
        "tier_family": "unknown-custom",
        "compute_util_factor": 0.45,
        "llm_prefill_util_factor": 0.10,
        "llm_decode_bw_realization": 1.0,
        "compute_overhead_ms": 1.0,
        "npu_share_default": 0.75,
        "capability_levels": None,
    },
}
