# ADR 013: Custom tier factory with silicon-class defaults

**Status:** Accepted
**Date:** 2026-05-19

## Context

Users want to size against silicon not in the canonical registry. A naive
custom-tier constructor would inherit `Hardware`'s field defaults — including
`compute_util_factor=0.45` (calibrated for NPU Mid), which over-projects a
2-TOPS Neutron chip by ~2.5×. Optimistic-by-default calibration is a trap.

## Decision

`make_custom_tier()` requires a `silicon_class` keyword
(`neutron` | `lp5x_64` | `lp5x_128` | `lp5x_128_int8` | `gddr_class` |
`unknown`). The class selects family-appropriate defaults for
`compute_util_factor`, `llm_prefill_util_factor`, `tier_family`, capability
levels, and the other calibration constants from `SILICON_CLASS_DEFAULTS`.
Callers may still override `compute_efficiency` / `bandwidth_efficiency`
explicitly.

Every custom tier gets `calibration_source = CalibrationSource(method="default",
confidence="low", ...)` (ADR 014), so surfaces render a "not calibrated for
this specific chip" warning. `unknown` uses neutral defaults with
`capability_levels=None`, falling back to the peak-TOPS heuristic.

## Consequences

- The compute-floor over-projection trap is closed: picking the right class
  picks the right utilization factor; picking `unknown` yields an explicit
  low-confidence warning rather than a silently wrong number.
- Custom tiers are visibly distinguishable from canonical ones via their
  calibration provenance.
