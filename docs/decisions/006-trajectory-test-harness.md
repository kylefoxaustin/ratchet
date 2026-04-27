# ADR 006: Trajectory-driven test harness pattern

**Status:** Accepted
**Date:** 2026-04 (originally rescue-bird ADR 011)
**Decision drivers:** test reproducibility, separation of "is the chip
fast enough?" from "can a human operate this?", avoidance of human-in-
the-loop modeling complexity.

## Context

A sizer needs a way to systematically exercise the pipeline under known
input profiles. Three structural questions apply to any consumer-loop
use case (drone, video, agentic-AI):

1. **What drives the system's input in the test?** A real human, a
   simulated one, a programmatic profile generator, or a closed-loop
   controller?
2. **How is the human in the loop modeled?** Real (impossible in CI),
   recorded (acceptable but limited), simulated (rabbit hole), or
   measurement-only (loses some realism but gains tractability)?
3. **How many scenarios?** One scenario isn't enough; sweeping every
   possible input pattern is too many.

## Decision

**Trajectory-driven (programmatic profile generators), small fixed set
of scenarios, measurement-only consumer model.**

- Inputs are driven by deterministic generators (in nightjar:
  `instrumentation/trajectories/trajectories.py`; in keyhole/skippy:
  the equivalent for video clips and prompt streams). Each profile is
  pure math, deterministic given seed, produces a sequence of input
  samples at a configured rate.

- The end consumer (FPV pilot for drone, viewer for video, agent for
  skippy) is modeled as **measurement-only**. Given the measured end-to-
  end latency, the model outputs observations about consumer experience.
  It does NOT generate inputs that affect the system's motion.

- The consumer model lives in a separate node so future closed-loop
  variants can plug in via the same interface.

## Alternatives considered

- **Real human in the loop.** Rejected: not reproducible, can't run
  in CI, and the human is exactly what we're trying to abstract.

- **Recorded human replay.** Considered. Useful but requires non-trivial
  capture infrastructure. Hooked for future support but not implemented
  in v0.1.

- **Programmatic closed-loop consumer.** Rejected for now: modeling
  predictive human control under varying latency is its own research
  problem. The chip team's question is "can a human operate this?",
  not "what does an artificial-consumer policy look like?". Answer the
  former without solving the latter.

- **One mega-scenario.** Rejected: doesn't isolate workload regimes.
  Need separate "floor" and "ceiling" data points to size the chip's
  steady-state vs. peak budget.

## Pattern for site-specific scenarios

Sites cover the workload envelope with a small fixed set (typically
4 scenarios) ordered by intensity:

1. **floor** — minimum sustained workload (any chip that fails this
   fails everything)
2. **typical** — realistic operational baseline
3. **stress** — sustained heavy load (what the sustained-state budget
   has to handle)
4. **ceiling** — worst-case peak (binding case for latency/peak budgets)

## Consequences

- The test harness is reproducible (deterministic seeds, pure math).
  Same input → same output across machines and runs.

- Adding a new scenario is a one-function-plus-yaml change.

- Future closed-loop consumers can plug into the same interface by
  changing an env var without touching the rest of the stack.

- The KPI evaluator gets a new pass/fail axis (consumer-experience
  flyability/quality/usability). For consumer-facing sizers this is
  the most operationally relevant KPI in the entire model.

## Worth knowing

The consumer-model thresholds (e.g. drone FPV pilot's 100ms flyability
boundary) are calibrated to specific use cases. Sites should expose
them as sliders so per-scenario tuning is possible.
