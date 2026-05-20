# ADR 012: Workload-pattern multipliers as optional overlay

**Status:** Accepted
**Date:** 2026-05-19

## Context

PAI sizer treats decode tok/s as intrinsic to (model, hardware) — workload-
invariant. Keyhole-sizer measured that decode rate and TTFT vary substantially
by workload category (plain chat vs RAG QA) and models that with multipliers.
Both views are correct for their domain; ratchet must accommodate both without
imposing either.

## Decision

`project_llm()` produces a workload-invariant `Projected`. Workload-pattern
scaling is a separate, optional overlay: `apply_workload_pattern(result,
WorkloadPatternMultipliers(...))` returns a new `Projected` with scaled decode
and TTFT, preserving the pre-multiplier decode rate in
`base_decode_pre_multiplier` for diagnostics.

PAI never calls it. Keyhole calls it after each projection with its measured
category multipliers. The model carries `llm_invariant_decode` (default True) so
surfaces can decide per-model whether multipliers apply.

## Consequences

- The two surface philosophies coexist without a branch in the engine.
- Multipliers compose with the anchor overlay (ADR 011); keyhole passes its
  multiplier into `overlay_llm_anchor` as well, so private anchors get the same
  workload scaling.
