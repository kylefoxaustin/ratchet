# ADR 007: Canonical tier registry

**Status:** Accepted
**Date:** 2026-05-19

## Context

Four ecosystem surfaces (PAI sizer, keyhole-sizer, keyhole backend, Skippy)
each carried their own Hardware/tier definitions. They stayed in sync through
maintainer discipline, not shared code — most visibly a byte-identical
`npu_anchors.py` in two repos. Tier silicon facts (NPU Mid is 200 eTOPS
INT8-only on 128-bit LPDDR5X-8.4) are invariant across the ecosystem; they
should be defined once.

## Decision

`ratchet.tiers.registry.TIERS` is the single source of truth: eight named
`Hardware` instances (NPU Low-LP4, Low-LP5-32bit, Low-LP5-64bit, Low-LP5X,
i.MX 95 measured, NPU Mid, NPU High, RTX 5090 reference), keyed by name.

Surfaces compose their *visible ladders* by selecting from `TIERS`; they do not
define new `Hardware` instances. The only runtime tier-construction path is
`make_custom_tier()` (ADR 013). If a surface needs a tier not in the registry,
the fix is to add it to ratchet, not to fork a local definition.

The MODELS catalog is the opposite case (per-surface content); see ADR 014 /
section 10. Tiers are canonical because facts are invariant; catalogs are
per-surface because content legitimately varies.

**Amendment (v0.2.2, 2026-05-21):** the initial registry specs (transcribed from
the design doc) diverged from the two production surfaces (PAI sizer,
keyhole-sizer), which agreed with each other. Discovered during the phase-2 PAI
retrofit. Corrected to production truth: NPU High TDP 35→40 W; NPU Low-LP4
3.2→4.266 GT/s (12.8→17.064 GB/s, LPDDR4X rate); NPU Low-LP5-32bit capacity
8→16 GB; per-tier TDPs LP4/LP5-32/LP5-64 → 10/15/20 W. Confirms the canonical
principle: where ratchet and a production surface disagree on an *invariant
silicon fact*, the production measurement wins and ratchet is corrected (the
surfaces don't fork around the engine).

## Consequences

- One definition of each silicon class; cross-surface drift becomes impossible
  by construction rather than by discipline.
- Surfaces show different ladders (PAI omits i.MX 95; keyhole omits the
  2-TOPS Low tiers) while sharing identical specs for shared silicon.
- Adding/altering a tier is an engine change in a dedicated ratchet session,
  governed by the rule-of-three discipline — never bundled with surface work.
