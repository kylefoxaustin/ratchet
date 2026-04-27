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

Add new ADRs by copying the format of an existing one. Don't edit accepted
ADRs in place — supersede them with a new one if a decision changes.
