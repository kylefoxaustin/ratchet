"""Dtype dispatch — route dtype/quant strings to capability and raw-peak-TOPS.

Two related-but-distinct concerns live here:

  * DTYPE_ATTR_MAP routes a compute dtype to the Hardware peak-TOPS field for
    raw-peak lookups (LLM compute floor, per ADR 015).
  * The capability key helpers answer "can this tier execute this workload?"

AMENDMENT 1 (2026-05-19): the executability of a quantized LLM is governed by
its *quant scheme*, not its compute dtype. A Q4_K_M model has compute_dtype
'fp16' but runs on INT8-only Neutron silicon via the INT8 dequant path. Gating
on compute_dtype alone wrongly rejects such models (and would skip a measured
anchor that exists for exactly that tier+model). quant_scheme_capability_key()
and hw_supports_dtype_via_key() resolve this; project_llm gates on them.
"""
from typing import TYPE_CHECKING, Literal, Optional

from ratchet.precision.capability import CapabilityLevel

if TYPE_CHECKING:
    from ratchet.tiers.hardware import Hardware


# ADR 016: FP4's compute win is realized only on a runtime with mature FP4 GEMM
# kernels (vLLM>=0.22 FlashInfer/CUTLASS NVFP4, TensorRT-LLM). On an immature
# runtime (llama.cpp today) the same silicon + weights yield no win — FP4 behaves
# like INT4 weight-only. This is a deployment/runtime property, NOT silicon, so it
# rides on the projection call rather than on Hardware.
FP4RuntimeMaturity = Literal["mature", "immature"]

# Canonical FP4 compute-dtype strings (all route to peak_tops_fp4).
_FP4_COMPUTE_DTYPES: frozenset[str] = frozenset({"nvfp4", "fp4", "mxfp4"})


# ADR 017: the NPU precision-set ladder. Each rung names the dtype the tensor
# engine executes at, which OVERRIDES a model's nominal compute_dtype for the
# projection floor (a weight-only Q4 model nominally carries compute_dtype='fp16'
# -> reads peak_tops_bf16; on an 'int8_fp8' tier it actually executes in fp8 ->
# reads peak_tops_fp8). Without this override the precision selector is inert.
NpuPrecisionSet = Literal["int8", "int8_fp8", "int8_fp8_fp4"]

PRECISION_SET_COMPUTE_DTYPE: dict[str, str] = {
    "int8":         "int8",
    "int8_fp8":     "fp8",
    "int8_fp8_fp4": "nvfp4",
}

# peak_tops_* fields to ZERO for each rung (the dtypes ABOVE the rung that the
# tier's capability dict declares UNSUPPORTED) — keeps a Hardware object coherent
# (it can't advertise FP4 TOPS while declaring FP4 unsupported).
PRECISION_SET_ZEROED_PEAK_TOPS: dict[str, tuple[str, ...]] = {
    "int8":         ("peak_tops_fp8", "peak_tops_fp4", "peak_tops_bf16"),
    "int8_fp8":     ("peak_tops_fp4",),
    "int8_fp8_fp4": (),
}


DTYPE_ATTR_MAP: dict[str, str] = {
    "int8":  "peak_tops_int8",
    "fp8":   "peak_tops_fp8",
    "bf16":  "peak_tops_bf16",
    "fp16":  "peak_tops_bf16",  # fp16 conflates to bf16 field
    "nvfp4": "peak_tops_fp4",   # NVFP4/MXFP4 — native FP4 compute path (Blackwell sm_120/sm_100)
    "fp4":   "peak_tops_fp4",   # alias
    "mxfp4": "peak_tops_fp4",   # alias
}


# AMENDMENT 1: quant scheme → canonical capability key.
# NOTE: NVFP4/MXFP4 (FP4) is a COMPUTE format → gates on 'nvfp4'. INT4 weight-only
# (AWQ/GPTQ) is a MEMORY format with no FP4 compute path → gates on 'bf16/fp16'
# (it dequantizes to bf16; prefill stays on the bf16 floor — measured on the 5090).
_QUANT_SCHEME_CAPABILITY_KEY: dict[str, str] = {
    "Q4_K_M": "q4_km",
    "Q5_K_M": "q4_km",
    "Q8_0":   "q4_km",
    "INT8_W8A8": "int8",
    "FP8":    "fp8",
    "FP16":   "bf16/fp16",
    "BF16":   "bf16/fp16",
    "NVFP4":  "nvfp4",
    "MXFP4":  "nvfp4",
    "INT4_AWQ": "bf16/fp16",  # weight-only INT4: memory format, dequant-to-bf16 compute path
    "INT4_GPTQ": "bf16/fp16",
}


def hw_peak_tops_for_dtype(hw: "Hardware", dtype: str) -> float:
    """Raw peak TOPS for a dtype, without compute_efficiency multiplier.

    LLM cross-class compute floor uses this against llm_prefill_util_factor
    (calibrated against raw peak per ADR 015). Vision uses effective_tops()."""
    attr = DTYPE_ATTR_MAP.get(dtype.lower())
    if attr is None:
        return 0.0
    return float(getattr(hw, attr, 0.0))


def is_fp4_compute_dtype(dtype: str) -> bool:
    """True if dtype is an FP4 compute format (nvfp4/fp4/mxfp4)."""
    return dtype.lower() in _FP4_COMPUTE_DTYPES


def effective_compute_dtype(
    dtype: str, fp4_runtime_maturity: FP4RuntimeMaturity = "mature"
) -> str:
    """Compute dtype to use for the raw-peak-TOPS floor lookup (ADR 016).

    On an immature FP4 runtime the FP4 GEMM win is unrealizable, so an FP4 model
    is modeled as INT4 weight-only: the compute floor falls to the bf16 floor
    (decode stays BW-bound by the ~4-bit weight bytes, which the caller handles
    separately). For every other (dtype, maturity) combination this is identity."""
    if fp4_runtime_maturity == "immature" and is_fp4_compute_dtype(dtype):
        return "bf16"
    return dtype


def resolve_floor_dtype(
    npu_precision_set: Optional[str],
    model_compute_dtype: str,
    fp4_runtime_maturity: FP4RuntimeMaturity = "mature",
) -> str:
    """The dtype the projection compute floor should read (ADR 016 + 017).

    A tier's `npu_precision_set` (when set) names the dtype the tensor engine
    executes at, overriding the model's nominal compute_dtype; then the ADR-016
    maturity derate applies (immature FP4 -> bf16 floor). When the tier has no
    precision set (every canonical tier), this is just the model's compute_dtype
    plus the maturity derate — i.e. v0.2.6 behavior."""
    base = PRECISION_SET_COMPUTE_DTYPE.get(npu_precision_set or "", model_compute_dtype)
    return effective_compute_dtype(base, fp4_runtime_maturity)


def _dtype_capability_key(dtype: str) -> str:
    """Map a raw compute-dtype string to its canonical capability-table key."""
    dt = dtype.lower()
    return "bf16/fp16" if dt in ("bf16", "fp16") else dt


def quant_scheme_capability_key(quant_scheme: str) -> str:
    """Map an LLMModel quant_scheme to the canonical capability-table key.

    AMENDMENT 1: this is the correct executability key for a quantized LLM.
    Weight-only quants (Q4_K_M / Q5_K_M / Q8_0) gate on 'q4_km'; INT8_W8A8 on
    'int8'; FP8 on 'fp8'; FP16/BF16 on 'bf16/fp16'. Unknown schemes fall back
    to treating the string as a raw dtype."""
    key = _QUANT_SCHEME_CAPABILITY_KEY.get(quant_scheme)
    if key is not None:
        return key
    return _dtype_capability_key(quant_scheme)


def hw_supports_dtype_via_key(hw: "Hardware", cap_key: str) -> CapabilityLevel:
    """Capability level for an already-resolved canonical capability key.

    Reads hw.capability_levels when populated. Falls back to the peak-TOPS
    heuristic when None: 'q4_km' maps onto whichever FP/INT path the silicon
    has (int8 if present, else bf16), all others map directly."""
    if hw.capability_levels is not None:
        info = hw.capability_levels.get(cap_key)
        return info.level if info is not None else CapabilityLevel.UNSUPPORTED

    # Heuristic fallback (custom tiers with capability_levels=None).
    if cap_key == "bf16/fp16":
        probe = "bf16"
    elif cap_key == "q4_km":
        probe = "int8" if hw.peak_tops_int8 > 0.0 else "bf16"
    else:
        probe = cap_key
    return (
        CapabilityLevel.TENSOR_NATIVE
        if hw_peak_tops_for_dtype(hw, probe) > 0.0
        else CapabilityLevel.UNSUPPORTED
    )


def hw_supports_dtype(hw: "Hardware", dtype: str) -> CapabilityLevel:
    """Capability level for a raw compute dtype on a Hardware tier.

    Reads from hw.capability_levels when populated. Falls back to the peak-TOPS
    heuristic when None. For quantized LLMs prefer hw_supports_dtype_via_key()
    with quant_scheme_capability_key() (AMENDMENT 1)."""
    return hw_supports_dtype_via_key(hw, _dtype_capability_key(dtype))
