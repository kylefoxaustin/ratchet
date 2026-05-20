# ADR 014: Calibration provenance

**Status:** Accepted
**Date:** 2026-05-19

## Context

Users ask of every projected number: "how much should I trust this?" A
37.85 tok/s figure from measured silicon means something different than the
same figure from a custom-tier projection on default constants. The engine
should encode the answer structurally, not leave surfaces to guess.

## Decision

Every `Hardware` tier carries a `CalibrationSource(method, reference,
confidence)`:

- `method`: `measured` | `interpolated` | `vendor_spec` | `default`.
- `confidence`: `high` | `medium` | `low`.
- `reference`: human-readable provenance string.

A light invariant: `method="default"` requires `confidence="low"` (defaults are
not calibrated). Canonical tiers carry measured/vendor_spec high-confidence
sources; custom tiers carry default/low (ADR 013). `medium` is reserved for
partial/disputed vendor data.

Ratchet does not impose UI: it ships the classification; surfaces choose colors,
wording, and banners.

A related provenance note lives in the catalog: `BYTES_PER_PARAM` reconciles a
PAI/keyhole disagreement (Q5_K_M, Q8_0) at keyhole's values pending empirical
investigation; both surfaces converge by importing the canonical table.

## Consequences

- Confidence is a first-class, queryable property of every projection input.
- Surfaces can render trust signals consistently without re-deriving provenance.
