"""Memory-upgrade overlays — produce BW-scaled tier variants.

Encodes one architectural insight structurally: LLM decode is BW-bound, so
memory-bandwidth scaling translates directly to decode throughput. Prefill is
compute-bound; upgrading memory doesn't help it.
"""
import dataclasses
from typing import Optional

from ratchet.tiers.hardware import Hardware


def hw_with_memory(
    hw: Hardware,
    mem_type: str,
    mem_data_rate_gtps: float,
    name_suffix: Optional[str] = None,
) -> Hardware:
    """Return a Hardware variant with the memory subsystem swapped.

    Recomputes mem_bandwidth_gbs from the new data rate against the same bus
    width. Sets bw_projected=True and captures stock identity (stock_name,
    stock_mem_bandwidth_gbs) so silicon-intrinsic lookups still resolve.

    BW-scales measured_decode_overrides linearly. Holds
    measured_prefill_overrides at stock (prefill is compute-bound)."""
    new_bw = hw.mem_bus_width_bits * mem_data_rate_gtps / 8.0
    bw_ratio = new_bw / hw.mem_bandwidth_gbs

    new_name = (
        f"{hw.name} ({name_suffix})"
        if name_suffix
        else f"{hw.name} ({mem_type} @ {mem_data_rate_gtps} GT/s)"
    )

    new_decode_overrides = (
        {k: v * bw_ratio for k, v in hw.measured_decode_overrides.items()}
        if hw.measured_decode_overrides
        else None
    )

    return dataclasses.replace(
        hw,
        name=new_name,
        mem_type=mem_type,
        mem_data_rate_gtps=mem_data_rate_gtps,
        mem_bandwidth_gbs=new_bw,
        bw_projected=True,
        stock_name=hw.stock_name if hw.stock_name else hw.name,
        # Preserve the *original* stock BW across re-clones, mirroring the
        # stock_name handling above (a clone-of-a-clone still points at stock).
        stock_mem_bandwidth_gbs=(
            hw.stock_mem_bandwidth_gbs
            if hw.stock_mem_bandwidth_gbs is not None
            else hw.mem_bandwidth_gbs
        ),
        measured_decode_overrides=new_decode_overrides,
        # measured_prefill_overrides inherited unchanged (compute-bound).
    )


MEMORY_UPGRADE_OPTIONS: list[tuple[str, str, float]] = [
    ("LPDDR5T @ 11.2 GT/s", "LPDDR5T", 11.2),
    ("LPDDR6 @ 12 GT/s", "LPDDR6", 12.0),
    ("LPDDR6 @ 14 GT/s", "LPDDR6", 14.0),
]
