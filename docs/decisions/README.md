# Architectural Decision Records

These are short writeups (200-500 words each) capturing the *why* behind key
engine-level design choices in ratchet. They were originally formulated in the
rescue-bird/nightjar drone-sizing context but apply to any sizer consuming the
ratchet engine.

Format follows the lightweight ADR convention: each file describes one
decision, what alternatives were considered, what we chose, and why.

| #   | Decision                                                | Status   | Origin                       |
|-----|---------------------------------------------------------|----------|------------------------------|
| 001 | Two-source model: measured + projected                  | Accepted | rescue-bird ADR 001          |
| 002 | Per-subsystem process isolation for partition fidelity  | Accepted | rescue-bird ADR 002          |
| 003 | Native BF16 on NPU as competitive differentiator        | Accepted | rescue-bird ADR 007          |
| 004 | LLM workload modeled as memory-BW-bound                 | Accepted | rescue-bird ADR 008          |
| 005 | NPU efficiency factor of 0.55 default                   | Accepted | rescue-bird ADR 009          |
| 006 | Trajectory-driven test harness pattern                  | Accepted | rescue-bird ADR 011          |
| 007 | Canonical tier registry                                 | Accepted | ratchet v0.2.0               |
| 008 | 4-level capability taxonomy                             | Accepted | ratchet v0.2.0               |
| 009 | Measurement attachment unification                      | Accepted | ratchet v0.2.0               |
| 010 | Stock identity tracking on memory-upgrade clones        | Accepted | ratchet v0.2.0               |
| 011 | Anchor-secrets as post-projection overlay               | Accepted | ratchet v0.2.0               |
| 012 | Workload-pattern multipliers as optional overlay        | Accepted | ratchet v0.2.0               |
| 013 | Custom tier factory with silicon-class defaults         | Accepted | ratchet v0.2.0               |
| 014 | Calibration provenance                                  | Accepted | ratchet v0.2.0               |
| 015 | DTYPE attribute dispatch + dual-calibration convention  | Accepted | ratchet v0.2.0               |
| 016 | FP4 compute win is runtime-conditional                   | Accepted | ratchet v0.2.6               |
| 017 | NPU precision-set override + forward-looking FP4 NPU     | Accepted | ratchet v0.2.7               |

ADRs 007-015 were added in ratchet v0.2.0 (the engine-consolidation release).
Two design-doc amendments were folded in during implementation: AMENDMENT 1
(quant-scheme dtype gating, ADR 015) and AMENDMENT 2 (overlay takes the model,
ADR 011).

Add new ADRs by copying the format of an existing one. Don't edit accepted
ADRs in place — supersede them with a new one if a decision changes.
