# ADR 011: Anchor-secrets as post-projection overlay

**Status:** Accepted
**Date:** 2026-05-19

## Context

Some silicon measurements are private (specific measured decode tok/s, TTFT,
power; whose silicon was measured). They must never enter source, git history,
or chat transcripts — but they must still override projection at runtime. PAI
and keyhole already solved this with byte-identical `npu_anchors.py` modules
reading `st.secrets`; the loader just needs one canonical home.

## Decision

Private measurements live only in a gitignored `.streamlit/secrets.toml` under
a public, documented schema (KEY-not-VALUE discipline: field names are public,
values are credentials). `load_llm_anchor()` / `load_cnn_anchor()` read them at
runtime and return `None` on any failure (no Streamlit, absent secret,
malformed value) — never raising.

Anchors are never stored on `Hardware`. They apply as a post-projection
overlay: `overlay_llm_anchor()` takes a `Projected` result and hot-swaps the
headline numbers, setting `source="measured_silicon_anchor"` and recording
provenance in `silicon_anchor_meta`. The overlay short-circuits unchanged when
no secret is present, the tier has no spec cell, the tier is a memory-upgrade
clone, or the surface can't map the model key.

A surface-supplied `catalog_to_spec_key` callable translates local catalog keys
to canonical snake_case spec keys, and a `workload_multiplier` parameter lets
keyhole apply its workload-pattern scaling (PAI passes 1.0).

**AMENDMENT 2 (2026-05-19):** `overlay_llm_anchor` takes the `LLMModel`
explicitly — `overlay_llm_anchor(result, hw, model, catalog_to_spec_key, *,
workload_multiplier=1.0)`. The design's `_dtype_for_model(result.model_key)`
helper is removed: a `Projected` carries only a `model_key` string, with no path
to the compute dtype. Spec-cell routing keys off the model's quant-scheme
capability key (consistent with AMENDMENT 1), so a Q4_K_M model routes via
`q4_km` and resolves the NPU Mid INT8 cell.

## Consequences

- Public/private separation is structural, not procedural. The open-source
  loader never sees values except at runtime from local secrets.
- The overlay composes cleanly after the projection cascade and after any
  workload-pattern overlay (ADR 012).
