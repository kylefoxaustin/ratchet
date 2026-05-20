"""Parameterized post-projection overlay for private silicon anchors.

AMENDMENT 2 (2026-05-19): overlay_llm_anchor takes the LLMModel explicitly. The
design's _dtype_for_model(result.model_key) helper is removed — a Projected only
carries a model_key string, with no path to the compute dtype. Surfaces pass the
model. Spec-cell routing keys off the model's quant-scheme capability key (the
same AMENDMENT 1 reasoning: a Q4_K_M model routes via 'q4_km', not its 'fp16'
compute dtype — otherwise the NPU Mid INT8 cell would never resolve).
"""
import dataclasses
from typing import Callable, Optional

from ratchet.anchors.loader import load_llm_anchor
from ratchet.anchors.spec_routing import hw_to_anchor_tier_dtype
from ratchet.catalog.model import LLMModel
from ratchet.precision.dtype_map import quant_scheme_capability_key
from ratchet.projection.result import Projected
from ratchet.tiers.hardware import Hardware


def overlay_llm_anchor(
    result: Projected,
    hw: Hardware,
    model: LLMModel,
    catalog_to_spec_key: Callable[[str], Optional[str]],
    *,
    workload_multiplier: float = 1.0,
) -> Projected:
    """Hot-swap a Projected result with a private silicon anchor when available.

    Returns the same result unchanged when: no anchor secrets are loaded, the
    tier has no spec cell, the tier is a memory-upgrade clone (bw_projected), or
    the model has no spec-key mapping.

    catalog_to_spec_key: surface-supplied function mapping catalog model_keys to
        canonical (snake_case) spec model_keys.
    workload_multiplier: optional workload-pattern multiplier. Keyhole-sizer
        passes non-1.0 values; PAI sizer passes 1.0."""
    routing_key = quant_scheme_capability_key(model.quant_scheme)
    tier_dtype = hw_to_anchor_tier_dtype(hw, routing_key)
    if tier_dtype is None:
        return result

    spec_key = catalog_to_spec_key(result.model_key)
    if spec_key is None:
        return result

    anchor = load_llm_anchor(tier_dtype, spec_key)
    if anchor is None:
        return result

    new_decode = anchor.tokps * workload_multiplier
    new_ttft = (
        anchor.ms_per_inference / 1000.0
        if anchor.ms_per_inference
        else result.ttft_s
    )

    return dataclasses.replace(
        result,
        decode_tok_s=round(new_decode, 2),
        ttft_s=round(new_ttft, 4),
        source="measured_silicon_anchor",
        silicon_anchor_meta={
            "anchor_source": anchor.source,
            "measured_date": anchor.measured_date,
            "tier_dtype": tier_dtype,
            "spec_model_key": spec_key,
        },
    )
