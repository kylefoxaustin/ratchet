"""Spec-cell routing — map (Hardware, model dtype) to an anchor-secrets key.

The canonical spec defines three tier_dtype cells: 'mid_int8', 'high_int8',
'high_fp'. Routing keys off hw.tier_lookup_name so memory-upgrade clones route
to their stock identity; bw_projected clones have no anchors of their own.
"""
from typing import Optional

from ratchet.tiers.hardware import Hardware


def hw_to_anchor_tier_dtype(hw: Hardware, model_compute_dtype: str) -> Optional[str]:
    """Map (Hardware, model dtype) to anchor-secrets tier_dtype key.

    Returns None for bw_projected tiers (memory-upgrade clones don't have their
    own anchors) and for tier/dtype combinations with no spec cell."""
    if hw.bw_projected:
        return None

    name = hw.tier_lookup_name
    dt = model_compute_dtype.lower()

    if name == "NPU Mid":
        return "mid_int8" if dt in ("int8", "q4_km") else None
    if name == "NPU High":
        if dt == "int8":
            return "high_int8"
        if dt in ("bf16", "fp16", "fp8", "q4_km"):
            return "high_fp"
        return None
    return None
