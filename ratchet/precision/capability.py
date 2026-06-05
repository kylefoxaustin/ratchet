"""The 4-level capability taxonomy.

Answers two distinct questions about an NPU/GPU tier and a dtype:
  1. Does this tier have *any* way to execute this dtype? (binary)
  2. *How well* does it execute it? (4-level)

The classic split is consumer Blackwell (SM120) on INT8: the silicon has INT8
tensor cores via sm80 IMMA binary compatibility (so question 1 is yes), but
vLLM's CUTLASS fresh-compile path lacks SM120 templates, so for fresh-compile
workloads the answer to question 2 is tensor_compat, not tensor_native.
"""
from dataclasses import dataclass
from enum import Enum


class CapabilityLevel(Enum):
    TENSOR_NATIVE = "tensor_native"
    """Native tensor-core execution. The silicon has dedicated hardware for
    this precision (e.g., Hopper FP8, Neutron INT8)."""

    TENSOR_COMPAT = "tensor_compat"
    """Tensor-core execution via binary compatibility. The silicon's tensor
    cores run kernels compiled for an earlier architecture (e.g., sm80 IMMA
    INT8 on sm120 Blackwell). Works for pre-compiled workloads (TRT engines)
    but blocked for fresh-compile (vLLM CUTLASS)."""

    CUDA_CORE = "cuda_core"
    """General-purpose compute execution (DP4A or equivalent). Works but
    significantly slower than tensor cores. Currently unused by any canonical
    tier; reserved for future silicon classes."""

    UNSUPPORTED = "unsupported"
    """Cannot execute this precision at all."""

    def __bool__(self) -> bool:
        return self is not CapabilityLevel.UNSUPPORTED


@dataclass(frozen=True)
class CapabilityInfo:
    """Per-(tier, dtype) capability with provenance.

    level: how fast (the 4-level enum)
    reason: surface-rendering tooltip-grade explanation
    """

    level: CapabilityLevel
    reason: str

    def __bool__(self) -> bool:
        return bool(self.level)


# ─── Canonical capability tables ────────────────────────────────────

NEUTRON_INT8_ONLY_CAPABILITY: dict[str, CapabilityInfo] = {
    "int8":      CapabilityInfo(CapabilityLevel.TENSOR_NATIVE,
                                "INT8 tensor cores native to Neutron NPU"),
    "fp8":       CapabilityInfo(CapabilityLevel.UNSUPPORTED,
                                "Neutron NPU has no FP path"),
    "bf16/fp16": CapabilityInfo(CapabilityLevel.UNSUPPORTED,
                                "Neutron NPU has no FP path"),
    "q4_km":     CapabilityInfo(CapabilityLevel.TENSOR_NATIVE,
                                "Q4_K_M weight-only quant runs via INT8 dequant path"),
    "nvfp4":     CapabilityInfo(CapabilityLevel.UNSUPPORTED,
                                "Neutron NPU has no FP4 path"),
}


NPU_FULL_DTYPE_CAPABILITY: dict[str, CapabilityInfo] = {
    "int8":      CapabilityInfo(CapabilityLevel.TENSOR_NATIVE, "INT8 tensor cores"),
    "fp8":       CapabilityInfo(CapabilityLevel.TENSOR_NATIVE, "FP8 tensor cores"),
    "bf16/fp16": CapabilityInfo(CapabilityLevel.TENSOR_NATIVE, "BF16/FP16 tensor cores"),
    "q4_km":     CapabilityInfo(CapabilityLevel.TENSOR_NATIVE,
                                "Q4_K_M weight-only quant runs via FP16 dequant path"),
    "nvfp4":     CapabilityInfo(CapabilityLevel.UNSUPPORTED,
                                "no native FP4 tensor-core path (FP4 is Blackwell-class today)"),
}


SM120_BLACKWELL_CAPABILITY: dict[str, CapabilityInfo] = {
    "int8":      CapabilityInfo(CapabilityLevel.TENSOR_COMPAT,
                                "sm80 IMMA via binary compat; vLLM CUTLASS fresh-compile blocked"),
    "fp8":       CapabilityInfo(CapabilityLevel.TENSOR_NATIVE, "FP8 native to SM120"),
    "bf16/fp16": CapabilityInfo(CapabilityLevel.TENSOR_NATIVE, "BF16/FP16 native to SM120"),
    "q4_km":     CapabilityInfo(CapabilityLevel.TENSOR_NATIVE,
                                "Q4_K_M weight-only quant runs via FP16 dequant path"),
    "nvfp4":     CapabilityInfo(CapabilityLevel.TENSOR_NATIVE,
                                "NVFP4/MXFP4 native to SM120 (5th-gen FP4 tensor cores) - "
                                "memory + compute format. Compute win is RUNTIME-CONDITIONAL "
                                "(ADR 016): mature FP4 kernels (vLLM>=0.22 FlashInfer/CUTLASS, "
                                "TensorRT-LLM) -> ~3.6x BF16 prefill / ~2.2x decode; immature "
                                "runtimes (llama.cpp today) realize no win (~INT4 weight-only)"),
}


# ADR 017: a forward-looking "hypothetical FP4-capable edge NPU" — NPU_FULL plus
# native FP4. v0.2.5 deliberately set nvfp4=UNSUPPORTED on every real NPU class
# (no edge NPU ships FP4 today); this dict is the OPT-IN assertion for what an
# FP4-capable edge NPU would look like. ZERO silicon anchors — confidence is low,
# and the realized FP4 throughput is still governed by fp4_runtime_maturity
# (which for edge should be 'immature', ADR 016).
NPU_FP4_CAPABILITY: dict[str, CapabilityInfo] = {
    **NPU_FULL_DTYPE_CAPABILITY,
    "nvfp4": CapabilityInfo(CapabilityLevel.TENSOR_NATIVE,
                            "FP4 tensor cores on a forward-looking FP4-capable edge "
                            "NPU (ADR 017) - modeled, ZERO silicon anchors, confidence "
                            "low; realized win is runtime-conditional (ADR 016)"),
}


# ADR 017: the escalating precision ladder for make_custom_tier(npu_precision_set=...).
# No real silicon ships FP4 datapaths without FP8 (FP4 MACs build on the FP8 ones),
# so the valid states are a 3-rung radio, not free checkboxes. Each rung selects the
# capability dict that names which dtypes the tensor engine executes natively.
PRECISION_SET_CAPABILITY: dict[str, dict[str, CapabilityInfo]] = {
    "int8":          NEUTRON_INT8_ONLY_CAPABILITY,
    "int8_fp8":      NPU_FULL_DTYPE_CAPABILITY,
    "int8_fp8_fp4":  NPU_FP4_CAPABILITY,
}
