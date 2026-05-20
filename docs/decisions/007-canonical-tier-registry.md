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

## Consequences

- One definition of each silicon class; cross-surface drift becomes impossible
  by construction rather than by discipline.
- Surfaces show different ladders (PAI omits i.MX 95; keyhole omits the
  2-TOPS Low tiers) while sharing identical specs for shared silicon.
- Adding/altering a tier is an engine change in a dedicated ratchet session,
  governed by the rule-of-three discipline — never bundled with surface work.
