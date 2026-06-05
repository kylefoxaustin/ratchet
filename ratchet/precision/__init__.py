"""Capability taxonomy + dtype dispatch."""
from ratchet.precision.capability import (
    NEUTRON_INT8_ONLY_CAPABILITY,
    NPU_FP4_CAPABILITY,
    NPU_FULL_DTYPE_CAPABILITY,
    PRECISION_SET_CAPABILITY,
    SM120_BLACKWELL_CAPABILITY,
    CapabilityInfo,
    CapabilityLevel,
)
from ratchet.precision.deployment_path import (
    WorkloadKernelSource,
    deployment_path_for_tier,
)
from ratchet.precision.dtype_map import (
    DTYPE_ATTR_MAP,
    PRECISION_SET_COMPUTE_DTYPE,
    FP4RuntimeMaturity,
    NpuPrecisionSet,
    effective_compute_dtype,
    hw_peak_tops_for_dtype,
    hw_supports_dtype,
    hw_supports_dtype_via_key,
    is_fp4_compute_dtype,
    quant_scheme_capability_key,
    resolve_floor_dtype,
)

__all__ = [
    "CapabilityLevel", "CapabilityInfo",
    "NEUTRON_INT8_ONLY_CAPABILITY", "NPU_FULL_DTYPE_CAPABILITY",
    "SM120_BLACKWELL_CAPABILITY", "NPU_FP4_CAPABILITY", "PRECISION_SET_CAPABILITY",
    "DTYPE_ATTR_MAP",
    "hw_supports_dtype", "hw_supports_dtype_via_key",
    "hw_peak_tops_for_dtype", "quant_scheme_capability_key",
    "deployment_path_for_tier", "WorkloadKernelSource",
    "FP4RuntimeMaturity", "effective_compute_dtype", "is_fp4_compute_dtype",
    "NpuPrecisionSet", "PRECISION_SET_COMPUTE_DTYPE", "resolve_floor_dtype",
]
