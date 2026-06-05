"""Deployment-path classifier.

Exists because tensor_compat means different things to different workload
categories — pre-compiled workloads (TRT engines) see a fast path, fresh-compile
workloads (vLLM CUTLASS) see a blocker.

ADR 016 adds an orthogonal axis for FP4: even on silicon with native FP4 cores
(tensor_native), the throughput win is realized only on a runtime with mature FP4
kernels. An immature FP4 runtime (llama.cpp today) yields no win. This is NOT the
same as workload_kernel_source — there, fresh-compile (vLLM) is the *blocked*
path; for FP4, vLLM is the *winning* path while llama.cpp loses. Hence a separate
axis, not a reuse of WorkloadKernelSource.
"""
from typing import Literal

from ratchet.precision.capability import CapabilityLevel
from ratchet.precision.dtype_map import (
    FP4RuntimeMaturity,
    hw_supports_dtype,
    is_fp4_compute_dtype,
)
from ratchet.tiers.hardware import Hardware

WorkloadKernelSource = Literal["precompiled", "fresh_compile"]


def deployment_path_for_tier(
    hw: Hardware,
    dtype: str,
    workload_kernel_source: WorkloadKernelSource,
    fp4_runtime_maturity: FP4RuntimeMaturity = "mature",
) -> str:
    """Classify deployment path for (tier, dtype, workload-kernel-source).

    Returns one of: 'native_fast' | 'compat_fast' | 'compat_blocked' |
    'cuda_core_fallback' | 'fp4_runtime_immature' | 'unsupported'.

    fp4_runtime_maturity (ADR 016): when 'immature' and dtype is FP4, the native
    FP4 cores exist but the runtime cannot realize the win → 'fp4_runtime_immature'
    (the model behaves like INT4 weight-only). Defaults to 'mature' (non-breaking)."""
    cap = hw_supports_dtype(hw, dtype)

    if cap is CapabilityLevel.TENSOR_NATIVE:
        if fp4_runtime_maturity == "immature" and is_fp4_compute_dtype(dtype):
            return "fp4_runtime_immature"
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
