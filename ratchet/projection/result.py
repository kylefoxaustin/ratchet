"""Structured projection result types.

The result-type union lives here (separate from the algorithm) because surfaces
consume these types directly in their UI rendering via match/case.
"""
from dataclasses import dataclass
from typing import Literal, Optional, Union

SourceLabel = Literal[
    "measured", "measured_anchor", "same_class_anchor",
    "cross_class", "measured_silicon_anchor",
]

Regime = Literal["bw_bound", "compute_bound"]


@dataclass(frozen=True)
class Projected:
    """A successful projection result."""

    # Headline numbers
    decode_tok_s: float
    prefill_tok_s: float
    ttft_s: float

    # Workload context
    decode_tokens: int
    prompt_tokens: int
    decode_s: float
    prefill_s: float
    total_s: float
    host_ms: float

    # Classification
    source: SourceLabel
    regime: Regime

    # Tier and model context
    hw_name: str
    model_key: str
    workload_id: str

    # Optional diagnostic fields
    decode_ceiling_tok_s: Optional[float] = None
    base_decode_pre_multiplier: Optional[float] = None
    silicon_anchor_meta: Optional[dict] = None


@dataclass(frozen=True)
class WontFit:
    """Model cannot fit on hardware at this context length."""
    hw_name: str
    model_key: str
    workload_id: str
    required_gb: float
    available_gb: float
    headroom_gb: float
    breakdown: dict
    prompt_tokens: int
    decode_tokens: int


@dataclass(frozen=True)
class DtypeMismatch:
    """Model requires a compute dtype this tier doesn't support."""
    hw_name: str
    model_key: str
    workload_id: str
    required_dtype: str
    tier_capability: str
    retargeting_hint: Optional[str] = None


ProjectionResult = Union[Projected, WontFit, DtypeMismatch]
