"""Deployment-path classifier.

Exists because tensor_compat means different things to different workload
categories — pre-compiled workloads (TRT engines) see a fast path, fresh-compile
workloads (vLLM CUTLASS) see a blocker.
"""
from typing import Literal

from ratchet.precision.capability import CapabilityLevel
from ratchet.precision.dtype_map import hw_supports_dtype
from ratchet.tiers.hardware import Hardware

WorkloadKernelSource = Literal["precompiled", "fresh_compile"]


def deployment_path_for_tier(
    hw: Hardware,
    dtype: str,
    workload_kernel_source: WorkloadKernelSource,
) -> str:
    """Classify deployment path for (tier, dtype, workload-kernel-source).

    Returns one of: 'native_fast' | 'compat_fast' | 'compat_blocked' |
    'cuda_core_fallback' | 'unsupported'."""
    cap = hw_supports_dtype(hw, dtype)

    if cap is CapabilityLevel.TENSOR_NATIVE:
        return "native_fast"
    if cap is CapabilityLevel.TENSOR_COMPAT:
        return (
            "compat_fast"
            if workload_kernel_source == "precompiled"
            else "compat_blocked"
        )
    if cap is CapabilityLevel.CUDA_CORE:
        return "cuda_core_fallback"
    return "unsupported"
