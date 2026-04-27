# ADR 002: Per-subsystem process isolation for partition fidelity

**Status:** Accepted
**Date:** 2026-04 (originally rescue-bird ADR 002)

## Context

The simulation could be structured as one monolithic process or as N
separate processes — each mapping to a logical block on the target SoC.
Monolithic is simpler to debug. Separate processes are more work but
mirror real silicon better.

## Decision

Each logical block that maps to a distinct silicon engine gets its own
process, in its own container. In the rescue-bird/nightjar context this
means per-subsystem ROS 2 nodes (`drone_perception`, `drone_vio`,
`drone_radar`, etc.). Other sizers using ratchet should follow the same
pattern with whatever IPC layer they prefer (gRPC, ZMQ, ROS 2, etc.).

## Alternatives considered

- **Monolithic Python process.** Rejected: process boundaries are how
  real silicon enforces engine isolation. Co-locating everything in one
  process makes the bandwidth measurements wrong (no DRAM round-trip for
  data passed in-process).
- **Threads in one process.** Rejected: same problem, plus the GIL hides
  realistic CPU contention.

## Consequences

- Every interface between subsystems goes through IPC, which is observable.
  The bandwidth section of the partition report wouldn't exist without
  this structure.
- Each container can be cgroup-isolated (CPU/memory limits) and
  GPU-pinned. Approximates engine isolation on real silicon.
- Per-process containers cost build time and memory. Acceptable trade
  because it's a development tool, not a deployed system.

## Worth knowing

The IPC hop is not free — ROS 2 DDS over loopback adds 100-500µs of
latency per hop on a typical Linux box. That's much higher than the
equivalent SoC NoC traversal would be. For tight latency budgets this
means the SITL latency *overestimates* what real silicon would do.
Subtract the IPC hop overhead before reporting silicon-side latency.
