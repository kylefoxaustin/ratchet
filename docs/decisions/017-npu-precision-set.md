# ADR 017: NPU precision-set override + forward-looking FP4-capable NPU

**Status:** Accepted
**Date:** 2026-06-05
**Origin:** cross-surface spec from [docs] (`personal-ai-framework/docs/npu-precision-set-selector-spec.md`); gates PAI-sizer's precision selector. Reviewed + cleared by [docs] 2026-06-05 (Blocker B resolution folded in below).

## Context

PAI-sizer wants users to pick an NPU tier (Mid/High) and *independently* select
its precision capability as an escalating ladder — `int8 → int8+fp8 →
int8+fp8+fp4` — to show the benefit of FP-capable edge silicon. Two engine facts
block this today:

1. `make_custom_tier` **hard-locks** `capability_levels` to the silicon_class
   default; a caller cannot say "Mid memory class, but FP4-capable engine."
2. v0.2.5 deliberately set `nvfp4 = UNSUPPORTED` on every NPU capability dict
   (no edge NPU ships FP4), so `project_llm` returns `DtypeMismatch` for an FP4
   model on any NPU class — there is no FP4-capable-NPU object to construct.

Precision is genuinely orthogonal to memory class, but ratchet currently
conflates them inside `silicon_class`. This ADR separates them at the factory
level (the minimal, additive change) rather than refactoring the silicon_class
taxonomy or the canonical `TIERS` (deferred — rule-of-three).

## Decision

1. **New capability dict `NPU_FP4_CAPABILITY`** = `NPU_FULL_DTYPE_CAPABILITY`
   with `nvfp4 = TENSOR_NATIVE`. Its `reason` marks it **forward-looking**: a
   hypothetical FP4-capable edge NPU, **zero silicon anchors**, confidence-low,
   and (per ADR 016) realized throughput governed by `fp4_runtime_maturity` —
   which for edge should be `immature`. `tensor_native` is correct because
   capability describes *silicon*, not realization (ADR 008/016); the
   over-promise is fenced off by the maturity axis, not by downgrading the level.

2. **New optional `npu_precision_set` on `make_custom_tier`:**
   `Literal["int8", "int8_fp8", "int8_fp8_fp4"] | None = None`.
   - `None` → unchanged (silicon_class default). **Non-breaking.**
   - When set, it overrides `capability_levels` via a new
     `PRECISION_SET_CAPABILITY` map: `int8 → NEUTRON_INT8_ONLY_CAPABILITY`,
     `int8_fp8 → NPU_FULL_DTYPE_CAPABILITY`, `int8_fp8_fp4 → NPU_FP4_CAPABILITY`
     (reuses two existing dicts + the new one).
   - **Gates peak_tops:** zeros `peak_tops_*` fields *above* the selected rung
     (`int8` zeros fp8/fp4/bf16; `int8_fp8` zeros fp4) so a tier object can't
     advertise FP4 TOPS while declaring FP4 unsupported. The capability gate is
     the real enforcement; zeroing keeps the Hardware internally coherent.
   - It also stamps the chosen set onto the new `Hardware.npu_precision_set`
     field (see #3); silicon_class still supplies calibration + memory defaults.

3. **The rung sets the effective compute dtype (Blocker B resolution).** A
   weight-only-quant model (Q4_K_M) carries `compute_dtype="fp16"`, so its
   compute floor reads `peak_tops_bf16`. Overriding capability alone would leave
   the FP8 rung reading bf16 → **the selector would show zero benefit (inert).**
   So the rung must also drive the *execution* dtype: `int8 → "int8"`,
   `int8_fp8 → "fp8"`, `int8_fp8_fp4 → "nvfp4"`. This is modeled by a new
   `Hardware.npu_precision_set` field (defaulted `None`); `project_llm` derives
   the effective compute dtype from it (overriding `model.compute_dtype`) before
   applying the ADR-016 maturity derate.
   - **Owner: `make_custom_tier` / `Hardware`, not the call site.** This makes the
     rung *atomic* — building the tier with a rung makes the projection correct,
     eliminating the inert-selector foot-gun. A `project_llm(compute_dtype_override=…)`
     alternative was rejected: it splits the rung across two calls.
   - **Why a new field and not derived from `capability_levels`:** NPU High
     already binds `NPU_FULL` (fp8-native). Deriving the execution dtype from the
     capability dict would silently switch existing High projections fp16→fp8 and
     **break v0.2.x numbers.** An explicit opt-in field (`None` on every canonical
     tier) is required for non-breaking behavior.
   - **Invariant fit:** `npu_precision_set` is a *silicon* fact (which dtypes the
     tensor datapath executes) — same category as `peak_tops_*` / `capability_levels`
     already on `Hardware`. This is distinct from `fp4_runtime_maturity`, which is
     a *runtime* property and stays a `project_llm` param (ADR 016). The two
     compose: the rung picks the dtype; maturity decides whether the FP4 win is
     realized. Edge `immature` default remains a PAI-sizer policy (spec §6/§8),
     not an engine default (engine default stays `"mature"`).

4. **Tests:** the 6 (Mid/High × 3 precision-sets) capability combos, the
   peak_tops gating, the corrected per-tier prefill anchors (High naive=333 /
   INT8=175 / FP8=175 / FP4-mature=87 / FP4-immature=333 ms; Mid INT8=FP8=351 ms,
   given the §4 TOPS ladder), and the immature-FP4 floor collapse composing with
   an `int8_fp8_fp4` custom tier.

## Review outcome (resolved with [docs] 2026-06-05)

- **§4 ladder vs §9 anchors** — not a real conflict: `800 = 4× bf16(200) = 2×
  fp8(400)`; §4 stated the ratio vs bf16, §9 vs the fp8 path. [docs] made both
  tables name their baseline; `800` TOPS stands (datapath doubling, confidence
  low, zero NPU anchors).
- **Q4_K_M prefill TOPS field** — confirmed against `dtype_map.py`: Q4 →
  `compute_dtype="fp16"` → `peak_tops_bf16`. This *was* a real spec error (naive
  Q4 on High = 333 ms, not 175). The fix became this ADR's #3: the rung sets the
  effective compute dtype. Corrected anchors are in the spec and the tests.
- **Confidence labeling** — FP4-on-NPU has no anchors; `make_custom_tier` already
  stamps `confidence='low'`. Surfaces must render the "modeled, unproven on edge
  silicon" caveat for any `int8_fp8_fp4` tier.

## Consequences

- PAI-sizer (and later keyhole-sizer) can construct `Mid/High × {int8, +fp8,
  +fp4}` tiers and A/B/C them. The FP8 rung is accuracy-only (same TOPS); the FP4
  rung's speed/RAM win is honest-by-default because edge maturity is `immature`.
- No change to canonical `TIERS`, to ADR 016, or to existing `make_custom_tier`
  callers (param is optional). The broader silicon_class ↔ precision
  orthogonalization remains a possible future ADR.
