"""WorkloadRecord schema — single source of truth for probe Parquet output.

Site-specific subsystem and phase string constants (e.g. SUBSYSTEM_RADAR for
nightjar) live in the consuming site's package, not here. The record
dataclass itself carries optional fields for site-specific telemetry; sites
fill in only what they measure.
"""
from .workload_record import (
    WorkloadRecord,
    WORKLOAD_SCHEMA,
)

__all__ = [
    "WorkloadRecord",
    "WORKLOAD_SCHEMA",
]
