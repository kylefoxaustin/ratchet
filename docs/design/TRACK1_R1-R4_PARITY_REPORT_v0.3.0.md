# Track 1 — ratchet v0.3.0 (R1–R4) parity report

**Date:** 2026-06-06
**From:** ratchet-side implementation session
**For:** design reviewer (drone-sizer / ratchet program)
**Companion:** `DRONE_SIZER_0.5_DESIGN_SPEC_v0.2.md`, Track 1 kickoff
**ADRs:** `docs/decisions/018-cpu-complex-and-perception-anchors.md`,
`docs/decisions/019-cpu-workload-projection.md`
**Tag:** `v0.3.0` (`origin/main` `02a152e`, tag `de1b677`) — pushed, resolvable.

---

## 0. Outcome

R1–R4 implemented, ADR'd, tested, and shipped. Cut A boundary held (no
composition / contention / verdict / camera-ingest logic entered the engine).
Additive and non-breaking — both production surfaces pass against the extended
engine. v0.3.0 tagged and pushed after verification.

**Read-and-acknowledge cleared 2026-06-06**; all three load-bearing gaps + two
minor calls confirmed by the reviewer in the recommended direction. The unifying
principle: **engine = mechanism + math + silicon facts; surface = drone numbers +
ratios + verdicts.**

---

## 1. What R1–R4 became

### R1 — CPU-complex modeling (ADR 018)
`CpuComplex` — a small frozen dataclass (`cores: int`, `microarch: str`,
`clock_ghz: float`) — added as `Hardware.cpu: Optional[CpuComplex] = None`.

- `microarch` is a **free string** ('A55', 'A720', 'x86_64', …), not an enum: the
  projection keys off a surface-supplied ratio, not off this string, so there is no
  enum to maintain when a surface names a new core.
- Defaulted `None` ⇒ every pre-v0.3.0 consumer (PAI sizer, keyhole-sizer) is
  untouched. Populated only on the drone-brain tiers.

### GAP A + R3 — perception measurement-attachment shape (ADR 018)
The existing attachments (`measured_decode_overrides`, `measured_llm`,
`measured_vision_overrides`, ADR 009) are flat `dict[str, float]` — too thin for
perception, which needs **distributions** (median/p95/max) and a **solver anchor**,
with measured-vs-projected varying *per metric within one workload*.

New frozen shapes:
- `LatencyDistribution(median_ms, p95_ms, max_ms?)`
- `SolverConvergenceAnchor(iters_median, iters_p95, solve_ms: LatencyDistribution,
  iters_max?)`
- `PerceptionAnchor(workload_class, latency, latency_source, solver?, solver_source?,
  bw_gbs?, bw_source?, cores?, cores_source?)` — **per-metric `calibration_source`**,
  so latency can be `measured` while `bw`/`cores` are `projected` (amendment 3).

Attached via `Hardware.measured_perception: Optional[dict[str, PerceptionAnchor]]`;
exposed via `Hardware.get_perception_anchor(workload_class)`. **R3** is satisfied by
the BW being queryable off the anchor (`.bw_gbs` + `.bw_source`); the surface sums
and runs the shared-bus check.

**Key resolution (the spec's R4-vs-§5 contradiction):** the engine ships the
*shape*; drone-sizer attaches the *numbers* surface-side at import — the
`measured_llm` precedent. The canonical registry stays free of drone-domain
numbers, and the A720 v1.0 calibration refresh (sub-phase 0.3b) becomes a surface
data update, never an engine re-tag.

### R2 — CPU-workload projection, two independent projections (ADR 019)
New module `ratchet/projection/cpu.py`. Two functions, deliberately **not
collapsed**, returning projected **numbers** (never verdicts):

- `project_frame_latency(baseline: LatencyDistribution, speedup) ->
  CpuLatencyProjection` — divides the whole distribution by the speedup; the tail
  (p95/max) is preserved for the distributional real-time verdict (amendment 13).
- `project_solver_convergence(anchor: SolverConvergenceAnchor, speedup) ->
  SolverConvergenceProjection` — **iterations are hardware-invariant** (carried
  through unchanged); only the per-iteration cost moves, so the to-convergence
  **solve-time distribution divides by the speedup**.

`speedup == 1.0` ⇒ `source="measured"` (passthrough); otherwise `"projected"`.
Invalid (`<= 0`) speedup raises.

### R4 — registry
- **i.MX 93** added as a net-new canonical tier (2×A55 @ 2 GHz, measured floor).
  Registry **8 → 9 tiers**.
- `cpu` added to **i.MX 95** (6×A55 @ 2 GHz), **NPU Mid / NPU High** (8×A720 @ 2 GHz).
- `tdp_watts` already existed on `Hardware` (amendment 10 correct) — values present.
- i.MX 95 production silicon facts left intact (Amendment-4/6); only `cpu` added.

---

## 2. R2 parameterization-home recommendation (spec §14)

**Recommendation, now implemented: the per-(workload, microarch) ratio is a
surface-supplied parameter; ratchet owns the projection math.**

`project_frame_latency` / `project_solver_convergence` take `speedup` as an
argument. The surface computes it from the per-workload IPC ratio (2.3× ORB / 1.7×
VINS / 2.3× OpenVINS) × any clock ratio (the drone tiers are iso-clock @ 2.0 GHz,
so it is just the IPC ratio). Rationale:

1. **Rule-of-three** — one consumer today; the ratios are drone-empirical.
2. **The A720 ratios are projected and get replaced** when 0.3b silicon is
   measured (v1.0). Surface-side ⇒ that refresh never re-tags the engine.
3. Keeps the canonical engine free of drone workload classes.

If a second CPU-projection consumer appears, the per-workload ratio mechanism is
the natural thing to lift into the engine *then* (not now).

---

## 3. Amendments surfaced during implementation

1. **R4-vs-§5 anchor-home contradiction.** The spec said the i.MX 93/95 perception
   anchors are both "attached to tiers" (R4) and "carried in the catalog" (§5).
   Resolved per the `measured_llm` precedent + Cut A: the engine ships the
   attachment *shape*, drone-sizer ships the *numbers* surface-side. (Reviewer-
   confirmed during read-and-acknowledge.)

2. **i.MX 93 DRAM specs are not in the empirical artifacts** — the perf artifacts
   carry CPU + DRAM-capacity but not the memory bus/rate/type. Shipped datasheet
   values (16-bit LPDDR4X-3733 ⇒ 7.466 GB/s, 2 GB, 3 W TDP) with
   `calibration_source(method="measured", confidence="medium")` whose reference
   string explicitly states *"CPU complex is measured truth; DRAM specs are
   datasheet pending farm-board confirmation from orb_slam."* Flagged rather than
   silently asserted as measured. **Open item → v0.3.1 patch** once orb_slam
   confirms the farm board's actual LPDDR4X width/rate/capacity + i.MX 93 TDP.

3. **`microarch` as free string, not enum** (R1 representation, §14 open Q) — the
   projection keys off a surface ratio, not the string, so an enum would be
   maintenance with no benefit.

No other contract reshaping was required.

---

## 4. Parity verification (§5 safety rail)

| Suite | Result | Notes |
|---|---|---|
| ratchet (own) | **259 / 259** | +14 new R1–R4 tests; the one canonical-count test updated 8→9 |
| keyhole-sizer | **38 / 38** | against the editable-installed extended engine; its precision-selector matrix reproduces the **exact** v0.2.7 numbers (behavior unchanged) |
| PAI sizer | import + build clean | imports ratchet 0.3.0, builds all 9 tiers; ships no test suite |

Clean editable install verified (`pip install -e .` → `import ratchet` → `0.3.0`).
The changes are additive (new defaulted fields, new module, new tier); existing
behavior is unchanged. Surfaces re-pin to `>=0.3.0,<0.4.0` on **their** schedule —
v0.3.0 does not force it; they keep working at v0.2.x-compatible behavior.

---

## 5. New public API

Tiers: `CpuComplex`, `LatencyDistribution`, `SolverConvergenceAnchor`,
`PerceptionAnchor`, `IMX93_MEASURED`, `Hardware.cpu`, `Hardware.measured_perception`,
`Hardware.get_perception_anchor()`.
Projection: `project_frame_latency`, `project_solver_convergence`,
`CpuLatencyProjection`, `SolverConvergenceProjection`.

---

## 6. Open items for the reviewer

1. **i.MX 93 DRAM specs + TDP** — confirm/correct the farm-board values (amendment
   2); I patch as v0.3.1. Until then they are datasheet, `confidence="medium"`.
2. **Track 2 can start** — the drone-sizer surface session builds against v0.3.0:
   workload bundle, composition module (extractable), camera-ingest model,
   distributional verdict matrix, loud calibration display. It attaches the
   perception anchor *numbers* (with per-metric `calibration_source`) onto the
   tiers via `measured_perception`, and supplies the per-workload `speedup` ratios.

---

*Verify before claiming: every number above was produced by a run in this session,
not asserted from memory.*
