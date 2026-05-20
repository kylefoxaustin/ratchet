"""The custom-tier factory — construct Hardware for non-canonical silicon.

Requires a silicon_class so calibration defaults match the silicon family.
calibration_source is always method='default', confidence='low' so surfaces
render an appropriate "not calibrated for this specific chip" warning.
"""
from typing import Literal, Optional

from ratchet.calibration.silicon_class import SILICON_CLASS_DEFAULTS
from ratchet.calibration.source import CalibrationSource
from ratchet.tiers.hardware import Hardware

SiliconClass = Literal[
    "neutron",        # INT8-only Neutron-class
    "lp5x_64",        # Full-dtype on LPDDR5X 64-bit
    "lp5x_128",       # Full-dtype on LPDDR5X 128-bit
    "lp5x_128_int8",  # INT8-only on LPDDR5X 128-bit (NPU Mid)
    "gddr_class",     # Discrete GPU with dedicated VRAM
    "unknown",        # No analogous silicon class; engine defaults
]


def make_custom_tier(
    name: str,
    *,
    silicon_class: SiliconClass,
    peak_tops_int8: float,
    mem_bandwidth_gbs: float,
    mem_capacity_gb: float,
    mem_bus_width_bits: int,
    mem_type: str,
    mem_data_rate_gtps: float,
    peak_tops_bf16: float = 0.0,
    peak_tops_fp8: float = 0.0,
    compute_efficiency: Optional[float] = None,
    bandwidth_efficiency: Optional[float] = None,
    tdp_watts: float = 0.0,
) -> Hardware:
    """Construct a custom Hardware tier with silicon-class-defaulted calibration.

    The silicon_class parameter selects defaults for compute_util_factor,
    llm_prefill_util_factor, tier_family, and capability_levels. Defaults are
    appropriate for the silicon family but NOT calibrated for the specific
    user-defined chip. calibration_source is set to method='default' with
    confidence='low' so surfaces render appropriate warnings."""
    if silicon_class not in SILICON_CLASS_DEFAULTS:
        raise ValueError(
            f"make_custom_tier: unknown silicon_class={silicon_class!r}. "
            f"Valid: {sorted(SILICON_CLASS_DEFAULTS)}"
        )
    defaults = SILICON_CLASS_DEFAULTS[silicon_class]

    return Hardware(
        name=name,
        peak_tops_bf16=peak_tops_bf16,
        peak_tops_int8=peak_tops_int8,
        peak_tops_fp8=peak_tops_fp8,
        mem_bandwidth_gbs=mem_bandwidth_gbs,
        mem_capacity_gb=mem_capacity_gb,
        mem_bus_width_bits=mem_bus_width_bits,
        mem_type=mem_type,
        mem_data_rate_gtps=mem_data_rate_gtps,
        compute_efficiency=(
            compute_efficiency if compute_efficiency is not None
            else defaults["compute_efficiency"]
        ),
        bandwidth_efficiency=(
            bandwidth_efficiency if bandwidth_efficiency is not None
            else defaults["bandwidth_efficiency"]
        ),
        tdp_watts=tdp_watts,
        tier_family=defaults["tier_family"],
        compute_util_factor=defaults["compute_util_factor"],
        llm_prefill_util_factor=defaults["llm_prefill_util_factor"],
        llm_decode_bw_realization=defaults["llm_decode_bw_realization"],
        compute_overhead_ms=defaults["compute_overhead_ms"],
        npu_share_default=defaults["npu_share_default"],
        capability_levels=defaults["capability_levels"],
        calibration_source=CalibrationSource(
            method="default",
            reference=f"silicon_class={silicon_class}, engine defaults",
            confidence="low",
        ),
    )
