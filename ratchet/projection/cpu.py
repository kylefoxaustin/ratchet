"""CPU-workload (perception / SLAM-VIO) projection — ADR 019, R2.

Two INDEPENDENT projections that the surface combines; do NOT collapse them:

  R2a — frame-latency projection. A measured baseline latency DISTRIBUTION
        (median / p95 / max) scaled by the target complex's per-core speedup →
        projected distribution. The real-time question.

  R2b — solver-convergence projection (optimizer-based workloads only). The
        iteration count to converge is HARDWARE-INVARIANT (deterministic
        trust-region float math — identical on A55 / A720 / x86), so it is carried
        through UNCHANGED. Only the per-iteration cost is hardware-dependent, so
        the to-convergence SOLVE TIME divides by the speedup. The quality question.

A workload can be real-time-feasible but non-converging, or converge robustly yet
miss its frame budget (VINS-Fusion on A720 is the latter). The two outputs stay
separate so the surface can tell those failure modes apart.

These functions return projected NUMBERS, never verdicts. The vs-budget checks
(real-time, convergence-within-budget, tail-frame-drop) need the workload-bundle
budgets (perception_fps, solver time), which are surface-side inputs — so the
verdict is surface work (the Cut A boundary). Likewise the per-(workload, core)
speedup ratio is a surface-supplied parameter (calibration that gets replaced when
real A720 silicon is measured at sub-phase 0.3b), not an engine constant.
"""
from dataclasses import dataclass
from typing import Optional

from ratchet.tiers.perception import (
    LatencyDistribution,
    SolverConvergenceAnchor,
)

# A projection is 'measured' when no scaling is applied (speedup == 1.0, i.e. the
# target IS the anchor silicon); otherwise it is 'projected'.
PerceptionSource = str


@dataclass(frozen=True)
class CpuLatencyProjection:
    """Projected per-frame latency distribution (R2a). The surface compares these
    against 1000/perception_fps (on p95, per amendment 13) to render the
    real-time verdict."""

    median_ms: float
    p95_ms: float
    max_ms: Optional[float]
    speedup: float
    source: PerceptionSource


@dataclass(frozen=True)
class SolverConvergenceProjection:
    """Projected to-convergence cost (R2b). iters_* are carried through unchanged
    (hardware-invariant); solve_* is the anchor solve-time / speedup. The surface
    compares solve_* against the solver time budget (on p95)."""

    iters_median: float
    iters_p95: float
    iters_max: Optional[float]
    solve_median_ms: float
    solve_p95_ms: float
    solve_max_ms: Optional[float]
    speedup: float
    source: PerceptionSource


def _source_for(speedup: float) -> PerceptionSource:
    return "measured" if speedup == 1.0 else "projected"


def project_frame_latency(
    baseline: LatencyDistribution, speedup: float
) -> CpuLatencyProjection:
    """R2a. Project a measured baseline latency distribution onto a target CPU
    complex via a surface-supplied per-core speedup (>1 = target is faster).

    speedup folds the per-(workload, microarch) IPC ratio and any clock ratio; the
    surface computes it (the drone tiers are iso-clock @ 2.0 GHz, so it is just the
    IPC ratio: 2.3× ORB / 1.7× VINS / 2.3× OpenVINS). The whole distribution scales
    by the same factor — the tail (p95/max) is preserved, which is what the
    distributional real-time verdict needs."""
    if speedup <= 0:
        raise ValueError(f"speedup must be > 0, got {speedup}")
    return CpuLatencyProjection(
        median_ms=baseline.median_ms / speedup,
        p95_ms=baseline.p95_ms / speedup,
        max_ms=(baseline.max_ms / speedup) if baseline.max_ms is not None else None,
        speedup=speedup,
        source=_source_for(speedup),
    )


def project_solver_convergence(
    anchor: SolverConvergenceAnchor, speedup: float
) -> SolverConvergenceProjection:
    """R2b. Project an optimizer's to-convergence cost. iters are hardware-
    invariant (carried through); the solve-time distribution divides by speedup
    (iters fixed, only per-iteration cost moves). Distributional — the fat tail
    (solve p95/max) is what drops frames even when the median clears."""
    if speedup <= 0:
        raise ValueError(f"speedup must be > 0, got {speedup}")
    solve = anchor.solve_ms
    return SolverConvergenceProjection(
        iters_median=anchor.iters_median,
        iters_p95=anchor.iters_p95,
        iters_max=anchor.iters_max,
        solve_median_ms=solve.median_ms / speedup,
        solve_p95_ms=solve.p95_ms / speedup,
        solve_max_ms=(solve.max_ms / speedup) if solve.max_ms is not None else None,
        speedup=speedup,
        source=_source_for(speedup),
    )
