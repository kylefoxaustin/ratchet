"""Perception measurement-attachment shapes (ADR 018, GAP A mechanism).

The existing measurement attachments (measured_decode_overrides, measured_llm,
measured_vision_overrides — ADR 009) are flat float dicts. Perception (SLAM/VIO)
measurements are richer: they are DISTRIBUTIONS (median / p95 / max — a real-time
verdict must size on the tail, not the median, per the drone-sizer spec) and, for
optimizer-based workloads, a SOLVER-CONVERGENCE anchor that decomposes into a
hardware-invariant iteration count and a hardware-dependent per-iteration cost.

This module defines the SHAPE only — ratchet ships the canonical data structure
and the projection math (ADR 019 / R2). It does NOT ship drone perception NUMBERS:
those are drone-sizer's, attached surface-side at import onto the tiers it uses
(the measured_llm precedent — engine canonical, surface supplies content). Each
metric carries its own calibration_source because, within one workload, latency
can be `measured` while BW/occupancy are `projected` (drone-sizer amendment 3:
only OpenVINS BW/occupancy was measured; ORB/VINS are projected).

Keep this a leaf module (CalibrationSource import only) so Hardware can import it
without an import cycle.
"""
from dataclasses import dataclass
from typing import Optional

from ratchet.calibration.source import CalibrationSource


@dataclass(frozen=True)
class LatencyDistribution:
    """A latency distribution in milliseconds. Real-time verdicts size on the
    tail (p95 / max), not the median (drone-sizer amendment 13)."""

    median_ms: float
    p95_ms: float
    max_ms: Optional[float] = None


@dataclass(frozen=True)
class SolverConvergenceAnchor:
    """An optimizer's to-convergence cost, decomposed for honest projection.

    `iters_*` is a HARDWARE-INVARIANT problem property: the trust-region iteration
    count to reach function tolerance is deterministic float math, identical on
    A55 / A720 / x86. It does NOT scale with the per-core ratio.

    `solve_ms` is the to-convergence solve-time distribution measured on the anchor
    silicon; it is HARDWARE-DEPENDENT and scales by the per-core ratio (since iters
    are invariant and only per-iteration cost moves). The projection (R2b) divides
    solve_ms by the ratio and carries iters through unchanged.
    """

    iters_median: float
    iters_p95: float
    solve_ms: LatencyDistribution
    iters_max: Optional[float] = None


@dataclass(frozen=True)
class PerceptionAnchor:
    """A measured (or projected) perception-workload anchor on one tier.

    Attached via Hardware.measured_perception[workload_class]. Per-metric
    calibration_source: latency may be `measured` while bw/cores are `projected`.
    The intrinsic workload properties (growing-vs-bounded memory, threading
    character, the ORB growing-map growth model) live in the drone-sizer catalog,
    NOT here — this is the per-(tier, workload) measurement only.
    """

    workload_class: str
    """e.g. 'orb_slam3' / 'vins_fusion' / 'openvins'."""

    latency: LatencyDistribution
    """Per-frame latency distribution (R2a baseline)."""

    latency_source: CalibrationSource

    solver: Optional[SolverConvergenceAnchor] = None
    """To-convergence anchor for optimizer-based workloads (R2b). None for
    filter-based workloads with no optimizer (steady-state OpenVINS)."""

    solver_source: Optional[CalibrationSource] = None

    bw_gbs: Optional[float] = None
    """Working-set memory bandwidth, GB/s (R3 — queryable for the shared-bus sum).
    Measured for OpenVINS; projected for ORB/VINS (drone-sizer amendment 3)."""

    bw_source: Optional[CalibrationSource] = None

    cores: Optional[float] = None
    """CPU core occupancy (fractional). Measured for OpenVINS; projected else."""

    cores_source: Optional[CalibrationSource] = None
