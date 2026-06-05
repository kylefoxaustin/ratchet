"""ratchet — generic SoC sizing engine.

The shared foundation for the edge-SoC sizing ecosystem. Surfaces (PAI sizer,
keyhole-sizer, keyhole, Skippy, nightjar, future drone-sizer) import the public
API from ``ratchet`` directly — not from submodules.

Submodules:

- ``ratchet.tiers``       — Hardware dataclass + canonical TIERS registry +
  memory-upgrade overlay + custom-tier factory.
- ``ratchet.precision``   — 4-level capability taxonomy + dtype dispatch +
  deployment-path classifier.
- ``ratchet.projection``  — LLM projection API (4-path cascade), structured
  result types, memory feasibility, workload-pattern overlay.
- ``ratchet.anchors``     — anchor-secrets loader + post-projection overlay for
  private silicon measurements (loaded at runtime, never in source).
- ``ratchet.catalog``     — LLMModel schema + quant byte tables (content is
  per-surface).
- ``ratchet.calibration`` — calibration provenance + silicon-class defaults.
- ``ratchet.engine``      — carried-forward primitives: Slider, SubsystemDemand,
  KpiResult, llm_demand math.
- ``ratchet.whatif``      — point/sweep/pareto runner that consumes the engine.
- ``ratchet.probes``      — Parquet writer + per-op / GPU / NVENC / g2g probes.
- ``ratchet.schemas``     — WorkloadRecord dataclass + PyArrow schema.

The MODELS catalog and per-surface tier ladders are NOT owned by ratchet — each
surface composes those from the canonical registry and ships its own catalog.
"""

__version__ = "0.2.5"

# ─── Tiers ──────────────────────────────────────────────────────────
from ratchet.tiers import (
    IMX95_MEASURED,
    MEMORY_UPGRADE_OPTIONS,
    NPU_HIGH,
    NPU_LOW_LP4,
    NPU_LOW_LP5_32BIT,
    NPU_LOW_LP5_64BIT,
    NPU_LOW_LP5X,
    NPU_MID,
    RTX_5090_REFERENCE,
    TIERS,
    Hardware,
    SiliconClass,
    hw_with_memory,
    make_custom_tier,
)

# ─── Precision / capability ─────────────────────────────────────────
from ratchet.precision import (
    DTYPE_ATTR_MAP,
    NEUTRON_INT8_ONLY_CAPABILITY,
    NPU_FULL_DTYPE_CAPABILITY,
    SM120_BLACKWELL_CAPABILITY,
    CapabilityInfo,
    CapabilityLevel,
    deployment_path_for_tier,
    hw_peak_tops_for_dtype,
    hw_supports_dtype,
    hw_supports_dtype_via_key,
    quant_scheme_capability_key,
)

# ─── Projection ─────────────────────────────────────────────────────
from ratchet.projection import (
    DtypeMismatch,
    FeasibilityCheck,
    Projected,
    ProjectionResult,
    WontFit,
    WorkloadPatternMultipliers,
    apply_workload_pattern,
    kv_cache_bytes_per_token,
    memory_feasibility,
    project_llm,
)

# ─── Anchors ────────────────────────────────────────────────────────
from ratchet.anchors import (
    CNNAnchor,
    LLMAnchor,
    hw_to_anchor_tier_precision,
    load_cnn_anchor,
    load_llm_anchor,
    overlay_llm_anchor,
)

# ─── Catalog ────────────────────────────────────────────────────────
from ratchet.catalog import (
    ACTIVE_PARAMS,
    BYTES_PER_PARAM,
    GGUF_SIZE_GB,
    LLMModel,
    lookup_model,
    resolve_measurement_key,
)

# ─── Calibration ────────────────────────────────────────────────────
from ratchet.calibration import CalibrationSource

# ─── Carried forward from v0.1.0 ────────────────────────────────────
from ratchet.engine import (
    KpiResult,
    Slider,
    SubsystemDemand,
    apply_sliders,
    default_values,
)
from ratchet.schemas import WORKLOAD_SCHEMA, WorkloadRecord
from ratchet.whatif import WhatifRunner

__all__ = [
    "__version__",
    # tiers
    "Hardware", "TIERS",
    "NPU_LOW_LP4", "NPU_LOW_LP5_32BIT", "NPU_LOW_LP5_64BIT", "NPU_LOW_LP5X",
    "IMX95_MEASURED", "NPU_MID", "NPU_HIGH", "RTX_5090_REFERENCE",
    "hw_with_memory", "MEMORY_UPGRADE_OPTIONS",
    "make_custom_tier", "SiliconClass",
    # precision
    "CapabilityLevel", "CapabilityInfo",
    "NEUTRON_INT8_ONLY_CAPABILITY", "NPU_FULL_DTYPE_CAPABILITY",
    "SM120_BLACKWELL_CAPABILITY",
    "hw_supports_dtype", "hw_supports_dtype_via_key", "hw_peak_tops_for_dtype",
    "quant_scheme_capability_key", "DTYPE_ATTR_MAP", "deployment_path_for_tier",
    # projection
    "project_llm", "Projected", "WontFit", "DtypeMismatch", "ProjectionResult",
    "memory_feasibility", "kv_cache_bytes_per_token", "FeasibilityCheck",
    "WorkloadPatternMultipliers", "apply_workload_pattern",
    # anchors
    "LLMAnchor", "CNNAnchor", "load_llm_anchor", "load_cnn_anchor",
    "overlay_llm_anchor", "hw_to_anchor_tier_precision",
    # catalog
    "LLMModel", "BYTES_PER_PARAM", "GGUF_SIZE_GB", "ACTIVE_PARAMS",
    "lookup_model", "resolve_measurement_key",
    # calibration
    "CalibrationSource",
    # carried from v0.1.0
    "WorkloadRecord", "WORKLOAD_SCHEMA",
    "Slider", "SubsystemDemand", "KpiResult", "apply_sliders", "default_values",
    "WhatifRunner",
]
