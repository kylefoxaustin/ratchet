"""Subsystem demand primitives.

A ``SubsystemDemand`` is the engine-level representation of "what one logical
block needs from the chip" — TOPS, CPU cores, memory bandwidth/capacity, p99
latency. Sites compute these for their domain-specific subsystems
(perception/VIO/radar for drone, decode/transcode for video, etc.) and pass
the resulting list to the KPI evaluator.

This module also provides the LLM demand calculator, because LLM inference
math is generic across all sizer use cases — see ADR 004 (LLM workload
modeled as memory-bandwidth bound).
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SubsystemDemand:
    """What one subsystem requires from the chip."""
    name: str
    target_engine: str           # npu | cpu | gpu | vpu | isp | dsp | rt_core | ...
    tops_required: float = 0.0
    cpu_cores_required: float = 0.0
    memory_bw_gbps: float = 0.0
    memory_capacity_mb: float = 0.0
    latency_ms_p99: float = 0.0
    notes: list[str] = field(default_factory=list)


def eff_tops(profile_npu: dict, precision: str) -> float:
    """Effective NPU TOPS at the given precision after the efficiency factor.

    Reads ``profile_npu["tops_<precision>"]`` (falling back to ``tops_bf16``)
    and multiplies by ``profile_npu["efficiency_factor"]`` (default 0.55).
    See ADR 005 for the rationale on the default.
    """
    peak = profile_npu.get(f"tops_{precision}", profile_npu.get("tops_bf16", 0))
    eff = profile_npu.get("efficiency_factor", 0.55)
    return peak * eff


def llm_demand(profile: dict, workload: dict) -> SubsystemDemand:
    """LLM inference is memory-bandwidth bound at edge token rates.

    Bytes/token ≈ params × bytes_per_param. INT8 = 1 byte/param, INT4 = 0.5,
    FP16/BF16 = 2. The compute side (~2 × params × tokens/sec ops) is small
    compared to the bandwidth requirement.

    Expects ``workload["llm"]`` with keys: ``active``, ``params_b``,
    ``tokens_per_sec``, ``precision``. Returns zeroed demand when inactive.
    """
    l = workload["llm"]
    if not l.get("active"):
        return SubsystemDemand(name="llm", target_engine="npu")

    bpp = {"int4": 0.5, "int8": 1, "fp16": 2, "bf16": 2}.get(l["precision"], 1)
    bytes_per_token = l["params_b"] * 1e9 * bpp
    bw = (bytes_per_token * l["tokens_per_sec"]) / 1e9
    # ~2 × params × tokens/sec ops, then convert to TOPS (1e12)
    required = (2 * l["params_b"] * l["tokens_per_sec"]) / 1000.0
    return SubsystemDemand(
        name="llm",
        target_engine="npu",
        tops_required=required,
        memory_bw_gbps=bw,
        memory_capacity_mb=l["params_b"] * 1000 * bpp,    # weights resident
        latency_ms_p99=1000.0 / l["tokens_per_sec"],     # inverse of throughput
        notes=["LLM is memory-BW bound — bandwidth, not TOPS, is the constraint"],
    )
