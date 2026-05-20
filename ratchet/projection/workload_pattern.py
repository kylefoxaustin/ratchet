"""Optional workload-pattern multiplier overlay.

PAI sizer never calls this (workload-invariant decode model). Keyhole-sizer
calls it after each projection with its measured workload-category multipliers.
"""
import dataclasses

from ratchet.projection.result import Projected


@dataclasses.dataclass(frozen=True)
class WorkloadPatternMultipliers:
    decode_p50_mult: float = 1.0
    decode_p95_mult: float = 1.0
    ttft_p50_mult: float = 1.0


def apply_workload_pattern(
    result: Projected,
    multipliers: WorkloadPatternMultipliers,
) -> Projected:
    """Apply workload-pattern multipliers to a Projected result. Returns a new
    Projected with base_decode_pre_multiplier preserved for diagnostics."""
    new_decode = result.decode_tok_s * multipliers.decode_p50_mult
    new_ttft = result.ttft_s * multipliers.ttft_p50_mult
    return dataclasses.replace(
        result,
        decode_tok_s=round(new_decode, 2),
        ttft_s=round(new_ttft, 4),
        base_decode_pre_multiplier=result.decode_tok_s,
    )
