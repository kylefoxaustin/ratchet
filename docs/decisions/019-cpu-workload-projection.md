# ADR 019: CPU-workload (perception) projection — two independent projections

**Status:** Accepted
**Date:** 2026-06-06
**Origin:** drone-sizer Track 1, R2 (DRONE_SIZER_0.5_DESIGN_SPEC_v0.2, amendments
1/2/13 + the grounded R2 re-run §A). Read-and-acknowledge cleared 2026-06-06.

## Context

drone-sizer projects a measured SLAM/VIO baseline (i.MX 95 6×A55) onto target CPU
complexes (notably 8×A720). Two questions are genuinely distinct and must not be
collapsed into one "latency" number:

- **Throughput / real-time:** does a frame finish within `1000/perception_fps`?
- **Convergence quality:** does the optimizer converge, and does the converged
  solve fit its time budget? A workload can be real-time-feasible but
  non-converging, or converge robustly yet miss its budget (VINS-Fusion on A720 is
  the latter: median 4 iters — robust — but ~57 ms converged solve vs a 33 ms
  30 fps budget → ~8 fps).

## Decision

`ratchet.projection.cpu` provides two functions returning projected **numbers**
(never verdicts):

1. **R2a `project_frame_latency(baseline: LatencyDistribution, speedup)`** →
   `CpuLatencyProjection`. Divides the whole distribution (median/p95/max) by the
   speedup, preserving the tail (the real-time verdict sizes on p95, amendment 13).

2. **R2b `project_solver_convergence(anchor: SolverConvergenceAnchor, speedup)`** →
   `SolverConvergenceProjection`. **Iteration count is hardware-invariant**
   (deterministic trust-region float math — identical on A55/A720/x86) and is
   carried through unchanged; only the per-iteration cost moves, so the
   to-convergence **solve-time distribution divides by the speedup**. Distributional
   (carries solve p95/max) because the fat tail drops frames where the median clears.

Two design seams, both confirmed by the reviewer:

- **The engine returns numbers; the surface renders the verdict.** The budgets
  (`perception_fps`, solver time) are workload-bundle inputs (surface-side), so the
  vs-budget checks (real-time, convergence-within-budget, tail-frame-drop) are
  surface work — the Cut A boundary. R2 does not take a budget.

- **The per-workload ratio is a surface-supplied parameter, not an engine table.**
  `speedup` folds the per-(workload, microarch) IPC ratio (2.3× ORB / 1.7× VINS /
  2.3× OpenVINS) and any clock ratio (the drone tiers are iso-clock @ 2 GHz, so it
  is just the IPC ratio). These ratios are *projected* A720 calibration that gets
  *replaced* when 0.3b silicon is measured (v1.0); keeping them surface-side means
  that refresh is a surface data update, never an engine re-tag (rule-of-three: one
  consumer). ratchet owns the reusable projection math; drone-sizer owns the numbers.

## Consequences

- R2 is additive: a new module + result dataclasses, entirely separate from the LLM
  `ProjectionResult` union and `project_llm` (untouched — existing surfaces
  unaffected).
- The two outputs stay separate so the surface can distinguish the two failure
  modes (sub-real-time vs non-convergent) that classical SLAM/VIO sizing needs and
  pure-throughput sizing misses.

## Open recommendation carried to the surface (spec §14)

The per-workload ratio parameterization home is **surface-side** (above). If a
future second consumer needs CPU projection, the per-workload ratio mechanism is
the natural thing to lift into the engine then (not now).
