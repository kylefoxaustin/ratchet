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
from typing import TYPE_CHECKING, Optional

from ratchet.precision.capability import CapabilityLevel

if TYPE_CHECKING:
    from ratchet.tiers.hardware import Hardware


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
