# CLAUDE.md — ratchet engine

Generic edge-SoC sizing engine. The shared foundation for an ecosystem of sizer
"surfaces" (PAI sizer, keyhole-sizer, keyhole backend, Skippy framework,
nightjar, future drone-sizer). v0.2.0 is the engine-consolidation release.

## Architecture

Surfaces import the public API from `ratchet` directly, never from submodules.
~57 public names re-exported by `ratchet/__init__.py`.

- `tiers/` — `Hardware` dataclass (the central engine object: silicon facts,
  calibration constants, measurement attachments, clone-tracking) +
  `TIERS` canonical registry (8 named tiers) + `hw_with_memory()` overlay +
  `make_custom_tier()` factory.
- `precision/` — `CapabilityLevel` 4-level taxonomy + `CapabilityInfo` +
  capability tables + `DTYPE_ATTR_MAP` dispatch + `deployment_path_for_tier()`.
- `projection/` — `project_llm()` (4-path cascade), `Projected | WontFit |
  DtypeMismatch` result types, `memory_feasibility()`, workload-pattern overlay.
- `anchors/` — `load_llm_anchor()` / `overlay_llm_anchor()`: private silicon
  measurements loaded at runtime from gitignored secrets, applied as a
  post-projection overlay. Never stored on `Hardware`, never in source.
- `catalog/` — `LLMModel` schema + quant byte tables. Content is per-surface;
  ratchet ships only the schema + a small reference catalog.
- `calibration/` — `CalibrationSource` provenance + silicon-class defaults.
- `engine/`, `whatif/`, `probes/`, `schemas/` — carried forward from v0.1.0
  unchanged in contract.

## Key invariants

- **Tiers are canonical; catalogs are per-surface.** Surfaces select visible
  ladders from `TIERS`; they never define new `Hardware` (except via
  `make_custom_tier()`). They ship their own MODELS catalog.
- **Dual-calibration convention (ADR 015).** LLM compute floor uses *raw peak*
  TOPS × `llm_prefill_util_factor`. Vision compute floor uses `effective_tops()`
  × `compute_util_factor`. Do not mix them — it double-discounts.
- **Quant-scheme gating (ADR 015 / AMENDMENT 1).** A quantized LLM's
  executability is governed by its `quant_scheme`, not its `compute_dtype`.
  `project_llm` gates Step 0b via `quant_scheme_capability_key()` +
  `hw_supports_dtype_via_key()`. A Q4_K_M model (compute_dtype fp16) runs on
  INT8-only Neutron silicon via the INT8 dequant path.
- **Anchor overlay takes the model (ADR 011 / AMENDMENT 2).**
  `overlay_llm_anchor(result, hw, model, catalog_to_spec_key, *,
  workload_multiplier=1.0)`. Routing keys off the model's quant-scheme key.
- **Hardware is mutated only at import time** (reference-tier measurement
  attachment). Treated as immutable at runtime; variants via
  `dataclasses.replace` / `hw_with_memory()` / `make_custom_tier()`.
- **Anchor-secrets KEY-not-VALUE discipline.** Field names are public; values
  are credentials in gitignored `.streamlit/secrets.toml`. Never put anchor
  *values* in source, commits, or chat.

## Projection cascade (first hit wins)

1. 🟢 `measured` — per-cell `hw.measured_llm[key][workload_id]` (alias-aware)
2. 🟢 `measured_anchor` — tier-level `hw.measured_decode_overrides[key]`
3. 🟡 `same_class_anchor` — sibling tier in `tier_family`, BW-scaled
4. 🔴 `cross_class` — first-principles `max(BW-floor, compute-floor)`

Anchor-secrets overlay runs *after* the cascade, separately.

## Dev

```bash
pip install -e ".[dev,gpu]"
pytest tests/ -q          # 213 tests (76 carried from v0.1.0 + 137 new)
```

Tests: `tests/unit/` per-submodule, `tests/integration/` end-to-end. Carried
v0.1.0 tests remain flat in `tests/`.

## Discipline

This engine is consumed by pinned versions across surfaces. Rule-of-three: no
engine addition until ≥2 surfaces demonstrate the need. Engine changes happen in
dedicated ratchet sessions, never bundled with surface work. Don't edit accepted
ADRs in place — supersede them. Surfaces pin `ratchet>=0.2.0,<0.3.0`.
