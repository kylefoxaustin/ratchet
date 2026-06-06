# ADR 020: `projected` + `derived_from_measured` calibration methods

**Status:** Accepted
**Date:** 2026-06-06
**Origin:** drone-sizer Track-2 read-and-acknowledge GAP 1 (reviewer + Kyle
adjudicated). Extends ADR 014 (calibration provenance) and ADR 018 (perception
attachment shape).

## Context

ADR 018 gave `PerceptionAnchor` **per-metric** `calibration_source` fields so that,
within one workload, latency can be `measured` while BW/occupancy are `projected`
(drone-sizer amendment 3), and the per-camera offload is `derived_from_measured`
(amendment 4). The drone-sizer spec §9 names the badge vocabulary as **measured /
vendor_spec / projected / derived_from_measured**, and `anchors/loader.py`'s
`BADGE_FOR_SOURCE` already maps `projected → 🟠`.

But `CalibrationMethod = Literal["measured", "interpolated", "vendor_spec",
"default"]` contained neither `projected` nor `derived_from_measured` — so the
per-metric source could not carry the label its own docstring intends. (Python's
Literal isn't runtime-enforced, so it silently "worked" in fixtures, but it was a
contract violation that a type-checker and the surface's badge rendering would both
trip over.)

## Decision

Add two members to `CalibrationMethod`:
- `projected` — computed from a model/ratio, not measured (e.g. the A720 IPC ratio).
- `derived_from_measured` — transformed from a measurement (e.g. per-camera offload
  scaling, HW-offload-at-N>1-cameras inference).

For consistency, add `derived_from_measured → 🔵` to `BADGE_FOR_SOURCE` (it was the
one §9 badge word the map lacked; `projected`/`vendor_spec`/`measured` were already
present). `interpolated`/`default` intentionally have no badge (they don't appear on
perception anchors).

## Consequences

- Purely additive: existing CalibrationSource construction is unchanged; the
  `__post_init__` rule (only `default` is special-cased → low confidence) is
  untouched, and no new validation rule is added (`projected` can be any confidence).
- One provenance vocabulary now spans both tier-level calibration and per-metric
  perception sources — no separate label type.
- Non-breaking for PAI sizer / keyhole-sizer (they use the existing four methods).
- Unblocks the drone-sizer surface to attach ORB/VINS `projected` BW and the
  `derived_from_measured` per-camera offload with honest labels.
