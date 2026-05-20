# ADR 010: Stock identity tracking on memory-upgrade clones

**Status:** Accepted
**Date:** 2026-05-19

## Context

`hw_with_memory()` produces tier variants with a swapped memory subsystem
(e.g. NPU Mid on LPDDR6 @ 14 GT/s). These variants share the silicon's compute
identity but differ in bandwidth. Silicon-intrinsic lookups — precision
capability, deployment path, anchor-secrets spec routing — must resolve against
the *stock* identity, not the variant. Keyhole previously detected variants via
a magic-constant bandwidth check, which is brittle.

## Decision

A memory-upgrade clone captures stock identity at construction:

- `bw_projected=True` flags it as a synthesized variant.
- `stock_name` snapshots the original tier name; `tier_lookup_name` returns it.
- `stock_mem_bandwidth_gbs` snapshots the original peak BW (preserved across
  re-clones, mirroring `stock_name`).

Silicon-intrinsic lookups key off `tier_lookup_name`. The anchor overlay
short-circuits when `bw_projected` is True (variants don't have their own
measured anchors). Decode overrides are BW-scaled; prefill overrides are held
at stock (prefill is compute-bound).

## Consequences

- Replaces keyhole's magic-constant check with explicit, inspectable state.
- Memory-upgrade projections route through stock for capability/anchor lookups
  while still reflecting upgraded bandwidth in decode throughput.
- Re-cloning a clone preserves the original stock identity rather than
  pointing at the intermediate variant.
