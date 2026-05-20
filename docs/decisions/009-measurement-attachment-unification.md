# ADR 009: Measurement attachment unification

**Status:** Accepted
**Date:** 2026-05-19

## Context

Real-silicon performance data attaches to tiers in several shapes: workload-
agnostic per-model decode/prefill overrides, per-(pipeline, resolution) vision
measurements, and full per-(model, workload) LLM cells. PAI and keyhole
encoded these differently (keyhole used flat `measured_llm_q4_decode_tok_s`
fields; PAI used nested dicts). The projection cascade needs one discipline.

## Decision

`Hardware` carries three flat dicts plus one nested dict:

- `measured_decode_overrides: {model_key: tok_s}` — workload-agnostic, used by
  the 🟢 measured_anchor path.
- `measured_prefill_overrides: {model_key: tok_s}` — held at stock under
  memory upgrades (prefill is compute-bound).
- `measured_vision_overrides: {pipeline_key: {resolution: {field: float}}}`.
- `measured_llm: {model_key: {workload_id: {decode_tok_s, ...}}}` — per-cell,
  highest-priority 🟢 measured path; populated only on the reference tier by
  surface bundle loaders at import.

Flat named fields (not one discriminated dict) for type clarity/autocomplete,
because the code paths are genuinely independent, and because future workload
classes add fields (`measured_audio_overrides`) rather than dict keys.

Private silicon measurements are NOT a Hardware field — they overlay
post-projection from gitignored secrets (ADR 011). Surfaces never extend
Hardware with their own measurement fields; if data doesn't fit these four,
the move is to propose extending ratchet, store off-Hardware, or use the
anchor overlay.

## Consequences

- One canonical attachment contract across surfaces.
- The asymmetry of memory-upgrade behavior (decode BW-scaled, prefill/vision
  held) is encoded structurally in `hw_with_memory()`.
- Note: the section-3 type annotation for `measured_vision_overrides` was
  two-level; the implementation uses the three-level shape its own docstring
  and the i.MX 95 registry entry require.
