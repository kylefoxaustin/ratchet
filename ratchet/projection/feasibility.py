"""Memory feasibility — precondition that runs before any projection.

AMENDMENT 1 (2026-05-19): weights size derives from model.gguf_size_gb (GB), not
a model.gguf_bytes field. weights_bytes = gguf_size_gb * 1e9.
"""
from dataclasses import dataclass
from typing import Literal

from ratchet.catalog.model import LLMModel
from ratchet.tiers.hardware import Hardware

RUNTIME_OVERHEAD_BYTES = 1_000_000_000

FeasibilityVerdict = Literal["fits", "tight", "wont_fit"]


@dataclass(frozen=True)
class FeasibilityCheck:
    verdict: FeasibilityVerdict
    required_gb: float
    available_gb: float
    headroom_gb: float
    breakdown: dict


def kv_cache_bytes_per_token(model: LLMModel, dtype_bytes: int = 2) -> float:
    """KV cache bytes per token, from transformer geometry. Uses GQA ratio."""
    kv_heads = model.num_kv_heads or model.num_attention_heads or 1
    attn_heads = max(model.num_attention_heads or 1, 1)
    gqa_ratio = kv_heads / attn_heads
    return model.num_layers * 2 * model.hidden_dim * gqa_ratio * dtype_bytes


def memory_feasibility(
    model: LLMModel,
    hw: Hardware,
    context_tokens: int,
) -> FeasibilityCheck:
    weights_bytes = model.gguf_size_gb * 1_000_000_000  # AMENDMENT 1
    kv_bytes = kv_cache_bytes_per_token(model) * context_tokens
    total_required = weights_bytes + kv_bytes + RUNTIME_OVERHEAD_BYTES
    available_bytes = hw.mem_capacity_gb * 1_000_000_000
    headroom = available_bytes - total_required

    if headroom < 0:
        verdict: FeasibilityVerdict = "wont_fit"
    elif headroom < available_bytes * 0.15:
        verdict = "tight"
    else:
        verdict = "fits"

    return FeasibilityCheck(
        verdict=verdict,
        required_gb=total_required / 1e9,
        available_gb=available_bytes / 1e9,
        headroom_gb=headroom / 1e9,
        breakdown={
            "weights_gb": weights_bytes / 1e9,
            "kv_cache_gb": kv_bytes / 1e9,
            "overhead_gb": RUNTIME_OVERHEAD_BYTES / 1e9,
        },
    )
