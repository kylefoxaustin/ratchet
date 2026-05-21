"""Parameterized post-projection overlay for private silicon anchors.

Both PAI sizer and keyhole-sizer apply private anchors as an overlay on a
projection result (verified during the phase-2 retrofit — outcome C). This is
the canonical home for that pattern.

AMENDMENT 2 (2026-05-19): overlay_llm_anchor takes the LLMModel explicitly (a
Projected carries only a model_key string).

AMENDMENT 3 (2026-05-20): reshaped against the real loader (3-arg, rich anchor).
Two multipliers — decode_mult and ttft_mult, both default 1.0 — unify PAI's
workload-invariant overlay (defaults) and keyhole's workload-scaled overlay
(non-default values from its workload-pattern table). TTFT is recomputed from
anchor.prefill_tokps only when ttft_mult != 1.0 AND prefill_tokps > 0; otherwise
the projection's TTFT is preserved (the anchor doesn't always carry prefill).
"""
import dataclasses
from typing import Callable, Optional

from ratchet.anchors.loader import load_llm_anchor
from ratchet.anchors.spec_routing import hw_to_anchor_tier_precision
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
    decode_mult: float = 1.0,
    ttft_mult: float = 1.0,
) -> Projected:
    """Hot-swap a Projected result with a private silicon anchor when available.

    Returns the same result unchanged when: no anchor secret is loaded, the
    anchor isn't a 'measured' source, the tier has no spec cell, the tier is a
    memory-upgrade clone (bw_projected), or the model has no spec-key mapping.

    catalog_to_spec_key: surface-supplied function mapping catalog model_keys to
        canonical (snake_case) spec model_keys.
    decode_mult / ttft_mult: workload-pattern multipliers. PAI passes the 1.0
        defaults (workload-invariant); keyhole passes measured per-workload
        values."""
    routing = hw_to_anchor_tier_precision(
        hw, quant_scheme_capability_key(model.quant_scheme)
    )
    if routing is None:
        return result
    tier, precision = routing

    spec_key = catalog_to_spec_key(result.model_key)
    if spec_key is None:
        return result

    anchor = load_llm_anchor(tier, precision, spec_key)
    if anchor is None or anchor.source != "measured" or anchor.tokps <= 0:
        return result

    new_decode = anchor.tokps * decode_mult

    # Memory-upgrade clones BW-scale the anchor's decode (decode is BW-bound on
    # active-param weight streaming); TTFT is held at stock (prefill is
    # compute-bound). Stock tiers have ratio 1.0. (ADR 011 Amendment 5 — prior
    # versions dropped the anchor on any upgrade, causing a measured→cross-class
    # discontinuity where the first upgrade tier could read *lower* than stock.)
    if hw.bw_projected and hw.stock_mem_bandwidth_gbs:
        new_decode *= hw.mem_bandwidth_gbs / hw.stock_mem_bandwidth_gbs

    if ttft_mult != 1.0 and anchor.prefill_tokps > 0:
        new_ttft = (1024.0 / anchor.prefill_tokps) * ttft_mult
    else:
        new_ttft = result.ttft_s  # preserve projection's TTFT (held at stock)

    decode_s = result.decode_tokens / max(new_decode, 1e-6)
    total_s = new_ttft + decode_s

    return dataclasses.replace(
        result,
        decode_tok_s=round(new_decode, 2),
        ttft_s=round(new_ttft, 4),
        decode_s=round(decode_s, 3),
        total_s=round(total_s, 3),
        source="measured_silicon_anchor",
        silicon_anchor_meta={
            "measured_date": anchor.measured_date,
            "spec_tier_precision": f"{tier}_{precision}",
            "spec_model_key": spec_key,
            "bw_projected": hw.bw_projected,
        },
    )
