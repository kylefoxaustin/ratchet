"""Spec-cell routing — map a Hardware tier + model precision to anchor keys.

The anchor-secrets schema keys cells by `f"{tier}_{precision}"` with
tier in {'mid','high'} and precision in {'int8','fp'}. This routes a Hardware
tier and a model's capability key to the (tier, precision) pair the loader's
3-arg signature expects.

Routing keys off hw.tier_lookup_name so memory-upgrade clones route to their
stock identity; bw_projected clones have no anchors of their own.
"""
from typing import Optional

from ratchet.tiers.hardware import Hardware


def hw_to_anchor_tier_precision(
    hw: Hardware, capability_key: str
) -> Optional[tuple[str, str]]:
    """Map (Hardware, capability key) to an anchor (tier, precision) pair.

    capability_key is the model's quant-scheme capability key
    ('int8' | 'fp8' | 'bf16/fp16' | 'q4_km') — see
    ratchet.precision.dtype_map.quant_scheme_capability_key.

    Returns None for tier/precision combinations with no spec cell.

    Memory-upgrade clones (bw_projected) route via their STOCK identity
    (tier_lookup_name) to the stock tier's anchor cell — the overlay then
    BW-scales the anchor's decode for the upgraded bandwidth (decode is
    BW-bound), holding TTFT at stock. (Prior versions returned None here, which
    dropped the anchor entirely on any memory upgrade and let the cell fall to
    cross-class — see ADR 011 Amendment 5.)

    Precision bucketing mirrors the silicon's execution path:
      - NPU Mid is INT8-only → INT8 and Q4_K_M (INT8 dequant) route to 'int8'.
      - NPU High runs INT8 natively, and FP / Q4_K_M (FP16 dequant) via 'fp'.
    """
    name = hw.tier_lookup_name
    key = capability_key.lower()

    if name == "NPU Mid":
        return ("mid", "int8") if key in ("int8", "q4_km") else None
    if name == "NPU High":
        if key == "int8":
            return ("high", "int8")
        if key in ("bf16/fp16", "fp8", "q4_km"):
            return ("high", "fp")
        return None
    return None
