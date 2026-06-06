"""CPU-complex silicon facts (ADR 018, R1).

ratchet's Hardware modeled only the NPU/GPU until v0.3.0, because LLM and vision
workloads run on the NPU. drone-sizer's perception workloads (visual / visual-
inertial SLAM) run on the Cortex-A CPU complex, which is the binding constraint
(the spine finding: drone-brain perception is CPU-compute-bound, and core *speed*
beats core *count* — the critical paths don't parallelize). Hardware therefore
needs the silicon facts of its CPU complex.

This is silicon facts only. The per-workload A55→A720 IPC ratios that PROJECT a
measured baseline onto a target complex are calibration, not silicon — they live
surface-side and are passed into the projection (ADR 019 / R2).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CpuComplex:
    """The application-processor CPU complex on an SoC.

    Minimal by design (R1): only what the CPU-workload projection (R2) consumes.
    `microarch` is a free string ('A55', 'A720', 'x86_64', …) rather than an enum
    so a surface can name a core type ratchet has never heard of; the projection
    keys off a surface-supplied per-(workload, microarch) ratio, not off this
    string, so no enum maintenance is required here.
    """

    cores: int
    """Number of cores of this type in the complex (e.g. 2 / 6 / 8)."""

    microarch: str
    """Core microarchitecture name, e.g. 'A55', 'A720', 'x86_64'. Free string."""

    clock_ghz: float
    """Per-core clock in GHz. The drone tiers are all modeled iso-clock @ 2.0 GHz,
    which is why the A720 projection is a pure IPC ratio (no frequency fudge)."""
