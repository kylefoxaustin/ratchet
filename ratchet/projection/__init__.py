"""The workload projection API."""
from ratchet.projection.cpu import (
    CpuLatencyProjection,
    SolverConvergenceProjection,
    project_frame_latency,
    project_solver_convergence,
)
from ratchet.projection.feasibility import (
    FeasibilityCheck,
    kv_cache_bytes_per_token,
    memory_feasibility,
)
from ratchet.projection.llm import project_llm
from ratchet.projection.result import (
    DtypeMismatch,
    Projected,
    ProjectionResult,
    WontFit,
)
from ratchet.projection.workload_pattern import (
    WorkloadPatternMultipliers,
    apply_workload_pattern,
)

__all__ = [
    "project_llm",
    "Projected", "WontFit", "DtypeMismatch", "ProjectionResult",
    "memory_feasibility", "kv_cache_bytes_per_token", "FeasibilityCheck",
    "WorkloadPatternMultipliers", "apply_workload_pattern",
    "project_frame_latency", "project_solver_convergence",
    "CpuLatencyProjection", "SolverConvergenceProjection",
]
