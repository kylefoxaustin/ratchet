# ADR 015: DTYPE attribute dispatch and dual-calibration convention

**Status:** Accepted
**Date:** 2026-05-19

## Context

Two distinct dtype-related lookups are easy to conflate:

1. *Which Hardware TOPS field do I read for this dtype?* (`peak_tops_int8` vs
   `peak_tops_bf16` vs `peak_tops_fp8`).
2. *What effective compute do I use in the floor calculation?*

And there is a calibration subtlety: `llm_prefill_util_factor` was calibrated
against *raw peak* TOPS, while the vision `compute_util_factor` multiplies
`effective_tops()` (raw peak × `compute_efficiency`). Using `effective_tops()`
for the LLM floor would double-discount.

## Decision

`DTYPE_ATTR_MAP` routes a dtype string to a Hardware peak-TOPS field;
`hw_peak_tops_for_dtype()` returns raw peak (no efficiency multiplier). The dual
convention is explicit:

- **LLM compute floor** uses raw peak (`hw_peak_tops_for_dtype`) ×
  `llm_prefill_util_factor`.
- **Vision compute floor** uses `effective_tops()` × `compute_util_factor`.

Note an intentional asymmetry on unknown dtypes: `effective_tops()` falls back
to `bf16`, while `hw_peak_tops_for_dtype()` returns 0.0.

**AMENDMENT 1 (2026-05-19):** capability/executability for a quantized LLM is
governed by its *quant scheme*, not its raw compute dtype. A Q4_K_M model has
`compute_dtype="fp16"` but runs on INT8-only Neutron silicon via the INT8
dequant path; gating `project_llm` on `compute_dtype` returned `DtypeMismatch`
and made the measured NPU Mid anchor unreachable. The fix:
`quant_scheme_capability_key()` maps Q4_K_M/Q5_K_M/Q8_0 → `q4_km`, INT8_W8A8 →
`int8`, FP8 → `fp8`, FP16/BF16 → `bf16/fp16`; `hw_supports_dtype_via_key()`
gates on that key. `project_llm` Step 0b uses these. The compute-floor TOPS-field
lookup still uses the raw `compute_dtype` (capability ≠ which TOPS field to
read). The same quant-scheme key drives anchor spec routing (ADR 011).

## Consequences

- The dual-calibration convention is preserved and documented; mixing the two
  floors is now a visible error rather than a silent ~1.5× discount.
- Quantized models resolve to the correct execution path and reach their
  measured anchors. The previously dead `q4_km` branches in spec routing are
  now live.
