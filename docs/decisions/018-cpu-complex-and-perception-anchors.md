# ADR 018: CPU-complex modeling + the perception measurement-attachment shape

**Status:** Accepted
**Date:** 2026-06-06
**Origin:** drone-sizer Track 1, R1 + R3 + R4 (DRONE_SIZER_0.5_DESIGN_SPEC_v0.2).
Read-and-acknowledge cleared by the reviewer 2026-06-06.

## Context

Until v0.3.0, `Hardware` modeled only the NPU/GPU — LLM and vision workloads run
on the NPU. drone-sizer's perception workloads (visual / visual-inertial SLAM)
run on the **Cortex-A CPU complex**, which is the binding constraint (the spine
finding: drone-brain perception is CPU-compute-bound, and core *speed* beats core
*count* — the critical paths don't parallelize). So the engine needs (R1) CPU
silicon facts and (R3) a place to hold per-workload perception measurements that
the surface can query.

The existing measurement attachments (`measured_decode_overrides`, `measured_llm`,
`measured_vision_overrides`, ADR 009) are flat `dict[str, float]`. Perception data
is richer: latency **distributions** (median / p95 / max — a real-time verdict
sizes on the tail, not the median) and, for optimizer workloads, a **solver-
convergence** anchor; and within one workload, latency may be `measured` while
BW/occupancy are `projected` (only OpenVINS BW was measured). The flat dicts can't
carry that.

## Decision

1. **R1 — `CpuComplex` field on `Hardware`.** A small frozen dataclass
   (`cores`, `microarch` free-string, `clock_ghz`), as `Hardware.cpu: Optional =
   None`. Defaulted None → every pre-v0.3.0 consumer (PAI sizer, keyhole-sizer)
   is unaffected. `microarch` is a free string (not an enum) because the
   projection keys off a surface-supplied ratio, not off this string (ADR 019), so
   no enum maintenance is needed.

2. **GAP A / R3 — a structured perception attachment shape.** New frozen
   dataclasses `LatencyDistribution`, `SolverConvergenceAnchor`, `PerceptionAnchor`,
   attached via `Hardware.measured_perception: Optional[dict[str, PerceptionAnchor]]`
   (keyed by workload_class). Each metric carries its own `calibration_source`
   (latency / solver / bw / cores) so measured-vs-projected is never blurred within
   one anchor. `Hardware.get_perception_anchor()` exposes it; the surface reads
   `.bw_gbs` (with source) to sum the shared bus (R3).

3. **The engine ships the SHAPE, not the NUMBERS.** ratchet defines the dataclasses
   + the i.MX 93/95 + A720 **silicon facts** (R4); drone-sizer attaches the
   perception **measurements** surface-side at import — the `measured_llm`
   precedent (engine canonical, surface supplies content). This keeps the canonical
   registry free of drone-domain numbers and means the A720 v1.0 calibration
   refresh (when 0.3b silicon lands) is a surface data update, never an engine
   re-tag. (Resolves the spec's R4-vs-§5 self-contradiction in favor of the
   LLM precedent + Cut A.)

4. **R4 — registry.** i.MX 93 added as a net-new canonical tier (2×A55 @ 2 GHz,
   measured floor); i.MX 95 (6×A55), NPU Mid/High (8×A720) gain `cpu`. `tdp_watts`
   already existed (populated). Per the Amendment-4/6 rule, the production i.MX 95
   tier's existing silicon facts are left intact (only `cpu` added); i.MX 93 is
   drone-sole-authority, so its measured CPU facts stand — but its DRAM specs are
   datasheet (16-bit LPDDR4X-3733) and flagged `confidence='medium'` pending
   farm-board confirmation from orb_slam, rather than silently asserted as measured.

**v0.3.1 update (2026-06-06):** orb_slam's live farm-board probe resolved the open
item *and caught an error* — the i.MX 93 A55 runs at **1.7 GHz, not 2.0** (2.0 is
the i.MX 95; 1.7 is the i.MX 93 datasheet max, pinned, no DVFS). Corrected
`clock_ghz 2.0→1.7` and `tdp 3.0→2.0 W`. Clock + 2 GB capacity are now measured
(`confidence='high'`); DRAM type/width/rate are SKU-pinned (DDR clock is
SM/firmware-owned, not Linux-readable); TDP is datasheet. This is the flag-don't-
assert discipline (Amendment-4/6) working as intended: the provisional value was
labeled, then corrected by measurement rather than silently shipped as truth.

## Consequences

- Non-breaking: `cpu` and `measured_perception` default None; the registry grows
  8 → 9 tiers (the one canonical-count test updated). Existing surfaces unaffected.
- The richer attachment shape is canonical and reusable; it does not retrofit the
  flat ADR-009 attachments (those stay as-is for LLM/vision).
- drone-sizer owns the perception numbers + their per-metric calibration_source.
