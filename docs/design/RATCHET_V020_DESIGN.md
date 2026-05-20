# RATCHET v0.2.0 — Design specification

**Status:** Draft, 2026-05-19
**Author:** Reviewer synthesis with input from PAI sizer (commit `c1d56f9`) and keyhole-sizer (commit `21debd2`)
**Target:** Claude Code briefing for phase 1 (ratchet v0.2.0 implementation)
**Predecessor:** ratchet v0.1.0 (commit `481d559`, the engine-extraction-from-nightjar release)

---

## Section 1 — Executive summary

### What's being built

Ratchet v0.2.0 is an expansion of the existing ratchet engine library to become the shared foundation across five surfaces of an edge-SoC sizing ecosystem. The current ratchet (v0.1.0) provides workload modeling and KPI evaluation primitives extracted from nightjar. The new ratchet absorbs the broader architectural patterns that the Skippy ecosystem evolved independently — NPU canonical tier specifications, an anchor-secrets system for private silicon measurements, a 4-level dtype capability taxonomy, projection math with 4-path resolution, and an LLM catalog schema.

The expansion is driven by a specific finding from phase 0 inventory: four surfaces (Skippy framework, PAI sizer, keyhole backend, keyhole-sizer) reached production releases at v5.10.0 / v1.0.0 / v1.0.0 / v1.0.0 through *discipline-coordinated* synchronization rather than shared code. The cross-coordination produced specific evidence the abstractions exist — most notably the byte-identical `npu_anchors.py` module that appears verbatim in PAI sizer and keyhole-sizer. Discipline worked, but it is load-bearing on the maintainer's attention. Crystallizing the shared concerns into a library replaces attention with code.

### Scope

Ratchet v0.2.0 owns:

- The Hardware dataclass and the canonical tier registry (NPU Low-LP4 through RTX 5090, plus i.MX 95 with ground-truth measurement)
- The 4-level capability taxonomy (`tensor_native` / `tensor_compat` / `cuda_core` / `unsupported`)
- The dtype-to-Hardware-field attribute map and external helper functions
- The workload projection API (4-path resolution, structured result types)
- The anchor-secrets loader and overlay system (post-projection hot-swap)
- The LLM catalog schema and measurement-alias resolution
- The calibration source / provenance tracking
- The custom-tier factory with silicon-class-defaulted calibration
- The probe writers and WorkloadRecord schema (carried forward from v0.1.0)
- The what-if runner primitives (carried forward from v0.1.0)

### Non-scope

Ratchet v0.2.0 does *not* own:

- Surface-specific tier ladders. Each surface composes its own visible ladder from the registry.
- Surface-specific catalog-to-spec key maps. The anchor-secrets canonical naming convention is fixed by ratchet, but each surface owns the translation from its local catalog keys to canonical keys.
- The workload-pattern multipliers (keyhole's 5090-measured ratios). These are an optional overlay applied at the surface level; ratchet provides the helper function but doesn't impose the model.
- UI conventions (tab structure, color schemes, badge rendering, role taxonomy display). Source-state classification flows through ratchet's result types; how those classifications get rendered is per-surface.
- Vision pipeline schemas. Keyhole's 23 vision pipelines are keyhole's domain; ratchet has no analog.
- Heterogeneous-architecture modeling (drone-class concerns: Cortex-A vs Cortex-M, ISP/VPU/DSP/NPU on the same SoC, bus arbitration, real-time guarantees). This is deferred to a later ratchet milestone driven by the drone use case.
- Opaque architecture-docs ingest (NDA-bound MPU reference manuals). Also deferred to drone-driven work.

The deferred items will eventually need ratchet, but the discipline-coordinated baseline already covers the four current surfaces well. Building heterogeneous-architecture support speculatively risks abstractions that don't fit the actual drone needs. Defer until drone work begins.

### What changes for the four existing surfaces

Each surface retrofits onto the new ratchet in a separate Claude Code session, in this order:

1. **PAI sizer** — lightest retrofit. PAI's `sizer/npu_model.py` and `sizer/precision.py` largely consist of code that ratchet now owns. The retrofit deletes that code, imports from ratchet, and keeps PAI-specific concerns (the `_ANCHOR_LLM_MODEL_KEY_MAP` translation, the catalog naming, the UI layout).
2. **keyhole-sizer** — similar retrofit, deletes the byte-identical `npu_anchors.py` plus other shared code, imports from ratchet. Migrates from keyhole's legacy flat measurement fields to the unified attachment schema.
3. **keyhole backend** — larger surface area. Retrofit is principally about replacing keyhole's local Hardware/tier definitions with ratchet imports. Bake-off harnesses and FastAPI service are unchanged.
4. **Skippy framework** — possibly the smallest retrofit despite the framework's overall complexity. Skippy uses ratchet only for cross-surface canonical specs and sizer-bundle integration. Most Skippy code is unrelated to ratchet's domain.

After all four retrofits, ratchet v0.2.0 enters its long-term mode: the engine is stable, the surfaces depend on pinned versions, and changes follow a rule-of-three discipline (no engine addition until at least two surfaces have demonstrated the need; engine changes happen in dedicated sessions, never bundled with surface work).

### What changes for the drone work

Nightjar v0.3.0-dev is currently pinned against ratchet v0.1.0. After ratchet v0.2.0 ships, nightjar gets the option to upgrade. The decision is not automatic — nightjar's existing v0.1.0 pin works fine for current drone-engineering needs. Nightjar upgrades when:

- The drone surface (new repo, planned post-retrofits) is being built and benefits from the new shared engine
- A specific feature in v0.2.0 is needed for ongoing nightjar work

If nightjar upgrades, the post-extraction stable nightjar tag (currently `v0.3.0-dev`) gets bumped to a `v0.3.0` stable release pinned against new ratchet.

### The pre-flight check

Phase 1 (this work) requires:

- Working `~/Documents/GitHub/ratchet` repo clone with v0.1.0 tag (commit `481d559`) intact
- Stable v1.0.0 / v5.10.0 tags on the four ecosystem surfaces (already present)
- Active Claude Code session in the ratchet directory
- This design document as the briefing

The deliverable of phase 1 is ratchet at v0.2.0: tagged, pushed, with the expanded test suite green, with ADRs landed, and with each of the eight major design areas (sections 2-10 of this document) implemented per spec.

The deliverable does *not* include retrofitting any surface. That's phases 2-5, each a separate Claude Code session in the respective surface's repo.

### Rollback safety

Every phase ends with a tagged release. Before phase 1 starts, the four surface repos are at known-good stable tags (already done via the stabilization sweep). Ratchet's v0.1.0 is preserved. If phase 1 produces a broken ratchet v0.2.0, the recovery is `git checkout v0.1.0` on ratchet and no surface is affected because no surface depends on v0.2.0 yet.

After each surface retrofit phase, that surface gets a new tagged release. If a retrofit goes badly, that single surface rolls back independently — the others remain on whatever they were pointed at.

There is no global rollback in this plan because there is no global commit. Each repo is independent; each phase produces a self-contained tagged checkpoint.

---

## Section 2 — Module structure

### Package layout

Ratchet v0.2.0 ships as a single Python package with seven submodules. The layout reflects natural domain boundaries: each submodule owns one architectural concern, and the public API is what surfaces import.

```
ratchet/
├── pyproject.toml                # name="ratchet", version="0.2.0"
├── README.md
├── CLAUDE.md
├── docs/
│   └── decisions/               # ADRs (carries over from v0.1.0, expanded)
│       ├── 001_two_source_workload_model.md       # carried from v0.1.0
│       ├── 002_process_isolation_for_partition_fidelity.md  # carried
│       ├── 003_bf16_competitive_differentiator.md  # carried
│       ├── 004_llm_memory_bound_workload.md       # carried
│       ├── 005_npu_efficiency_factor_calibration.md  # carried
│       ├── 006_trajectory_driven_test_harness.md  # carried
│       ├── 007_canonical_tier_registry.md         # new in v0.2.0
│       ├── 008_capability_taxonomy_4level.md      # new
│       ├── 009_measurement_attachment_unification.md  # new
│       ├── 010_stock_identity_tracking.md         # new
│       ├── 011_anchor_secrets_post_projection_overlay.md  # new
│       ├── 012_workload_pattern_multipliers_optional.md  # new
│       ├── 013_custom_tier_factory.md             # new
│       ├── 014_calibration_provenance.md          # new
│       └── 015_dtype_attribute_dispatch.md        # new
├── ratchet/
│   ├── __init__.py              # public API: re-exports the main surface
│   │
│   ├── engine/                  # Existing v0.1.0 content + expansion
│   │   ├── __init__.py
│   │   ├── slider.py            # Slider, apply_sliders, default_values
│   │   │                        #   (carried from v0.1.0)
│   │   ├── demand.py            # SubsystemDemand, eff_tops, llm_demand
│   │   │                        #   (carried from v0.1.0, made public)
│   │   └── kpi.py               # KpiResult, evaluate_budget, npu_kpis, etc.
│   │                            #   (carried from v0.1.0, made public)
│   │
│   ├── tiers/                   # NEW: canonical tier registry
│   │   ├── __init__.py          # Re-exports the registry constants
│   │   ├── hardware.py          # Hardware dataclass + properties + methods
│   │   ├── registry.py          # TIERS dict + named tier constants
│   │   ├── memory_overlay.py    # hw_with_memory() and MEMORY_UPGRADE_OPTIONS
│   │   └── custom.py            # make_custom_tier() factory + silicon_class
│   │
│   ├── precision/               # NEW: capability taxonomy + dtype dispatch
│   │   ├── __init__.py
│   │   ├── capability.py        # CapabilityLevel enum + CapabilityInfo
│   │   ├── dtype_map.py         # DTYPE_ATTR_MAP + hw_supports_dtype +
│   │   │                        #   hw_peak_tops_for_dtype
│   │   └── deployment_path.py   # deployment_path_for_tier() with
│   │                            #   workload_kernel_source parameter
│   │
│   ├── projection/              # NEW: the workload projection API
│   │   ├── __init__.py
│   │   ├── result.py            # Projected | WontFit | DtypeMismatch types
│   │   ├── feasibility.py       # memory_feasibility, kv_cache calc
│   │   ├── llm.py               # project_llm with 4-path resolution
│   │   └── workload_pattern.py  # optional workload-multiplier overlay
│   │
│   ├── anchors/                 # NEW: anchor-secrets system
│   │   ├── __init__.py
│   │   ├── loader.py            # load_llm_anchor / load_cnn_anchor
│   │   │                        #   (byte-identical from existing surfaces)
│   │   ├── schemas.py           # LLMAnchor / CNNAnchor dataclasses
│   │   ├── overlay.py           # overlay_llm_anchor (parameterized)
│   │   └── spec_routing.py      # tier+dtype → spec cell key mapping
│   │
│   ├── catalog/                 # NEW: LLM catalog schema
│   │   ├── __init__.py
│   │   ├── model.py             # LLMModel @dataclass(frozen=True)
│   │   ├── constants.py         # BYTES_PER_PARAM, GGUF_SIZE_GB, etc.
│   │   └── alias.py             # measurement_alias resolution helper
│   │
│   ├── calibration/             # NEW: provenance + silicon-class defaults
│   │   ├── __init__.py
│   │   ├── source.py            # CalibrationSource dataclass
│   │   └── silicon_class.py     # _SILICON_CLASS_DEFAULTS mapping
│   │
│   ├── probes/                  # CARRIED from v0.1.0
│   │   ├── __init__.py
│   │   ├── probe_writer.py
│   │   ├── op_probe.py
│   │   ├── gpu_probe.py
│   │   ├── nvenc_probe.py
│   │   └── g2g_probe.py
│   │
│   ├── schemas/                 # CARRIED from v0.1.0
│   │   ├── __init__.py
│   │   └── workload_record.py
│   │
│   └── whatif/                  # CARRIED from v0.1.0
│       ├── __init__.py
│       └── runner.py            # WhatifRunner — point/sweep/pareto
│
└── tests/
    ├── conftest.py
    ├── unit/                    # Per-module unit tests
    │   ├── test_engine_slider.py
    │   ├── test_engine_demand.py
    │   ├── test_engine_kpi.py
    │   ├── test_tiers_hardware.py
    │   ├── test_tiers_registry.py
    │   ├── test_tiers_memory_overlay.py
    │   ├── test_tiers_custom.py
    │   ├── test_precision_capability.py
    │   ├── test_precision_dtype_map.py
    │   ├── test_projection_feasibility.py
    │   ├── test_projection_llm.py
    │   ├── test_projection_result_types.py
    │   ├── test_projection_workload_pattern.py
    │   ├── test_anchors_loader.py
    │   ├── test_anchors_overlay.py
    │   ├── test_catalog_model.py
    │   ├── test_catalog_alias.py
    │   ├── test_calibration_source.py
    │   ├── test_calibration_silicon_class.py
    │   └── test_probes_smoke.py     # carried from v0.1.0
    └── integration/
        ├── test_full_llm_projection.py     # end-to-end via WorkloadProjector
        ├── test_anchor_overlay_e2e.py      # loader + overlay + result
        ├── test_memory_upgrade_e2e.py      # tier → clone → projection
        └── test_workload_record_roundtrip.py
```

The eleven runtime submodules of `ratchet/` align with the eight design areas in this document (engine + tiers + precision + projection + anchors + catalog + calibration + probes + schemas + whatif). Two of those (probes, schemas, whatif) carry over from v0.1.0 with mostly cosmetic touch-ups; the rest are new in v0.2.0.

### Why this layout

**Submodule boundaries follow the architectural cuts, not the file sizes.** Each submodule answers one question. `tiers/` answers "what is this NPU?" `precision/` answers "what dtypes can it run?" `projection/` answers "given this NPU and this workload, what performance do I get?" `anchors/` answers "are there private measurements that override the projection?" `catalog/` answers "what LLMs do we know about?" Calling these out as separate submodules keeps the responsibility surface explicit.

**The `engine/` submodule's name is historical, not architectural.** It currently holds Slider, SubsystemDemand, and KpiResult — primitives lifted from nightjar's analysis layer. In v0.2.0 those primitives still live there because the existing API contract carries forward unchanged. If we were greenfielding ratchet today we might split them across `tiers/`, `projection/`, and `whatif/`. But since these primitives are imported by existing consumers (nightjar v0.3.0-dev), splitting them would break the import surface for no functional gain. They stay in `engine/`.

**The `precision/` submodule is the consolidation of PAI's `sizer/precision.py` plus keyhole's inline `CapabilityLevel` literals plus the `_DTYPE_ATTR` map.** Three sources in two repos converge into one canonical location.

**The `anchors/` submodule is the byte-identical loader plus parameterized overlay.** The PAI and keyhole-sizer copies of `sizer/npu_anchors.py` get replaced by `from ratchet.anchors import load_llm_anchor`. The overlay function gets parameterized so keyhole can pass its workload multiplier and PAI passes 1.0.

**`projection/` separates result types from algorithm.** The result-type union (`Projected | WontFit | DtypeMismatch`) lives in `projection/result.py` because surfaces consume those types in their UI rendering. The 4-path algorithm lives in `projection/llm.py`. The memory feasibility check is its own module because it's a precondition that runs before any projection.

### Public vs private API surface

The top-level `ratchet/__init__.py` re-exports the consumer-facing API. Surfaces import from `ratchet` directly, not from submodules:

```python
# Consumer import pattern (PAI sizer, keyhole-sizer, nightjar, future):
from ratchet import (
    Hardware,
    TIERS,
    NPU_LOW_LP4, NPU_LOW_LP5_32BIT, NPU_LOW_LP5_64BIT, NPU_LOW_LP5X,
    IMX95_MEASURED, NPU_MID, NPU_HIGH, RTX_5090_REFERENCE,
    hw_with_memory, MEMORY_UPGRADE_OPTIONS,
    make_custom_tier,
    CapabilityLevel, CapabilityInfo,
    hw_supports_dtype, hw_peak_tops_for_dtype, DTYPE_ATTR_MAP,
    deployment_path_for_tier,
    project_llm,
    Projected, WontFit, DtypeMismatch,    # result types
    memory_feasibility, kv_cache_bytes_per_token,
    workload_pattern_multiplier,           # optional overlay
    LLMAnchor, CNNAnchor,
    load_llm_anchor, load_cnn_anchor,
    overlay_llm_anchor, overlay_cnn_anchor,
    LLMModel,
    BYTES_PER_PARAM, GGUF_SIZE_GB, ACTIVE_PARAMS,
    CalibrationSource,
    # carried from v0.1.0:
    WorkloadRecord, WORKLOAD_SCHEMA,
    Slider, SubsystemDemand, KpiResult,
    apply_sliders, default_values,
    WhatifRunner,
)
```

That's ~40 public names. Anything under a submodule that isn't re-exported is implementation detail. The leading-underscore private convention applies *within* submodules (`_SILICON_CLASS_DEFAULTS`, `_DTYPE_PEAK_FALLBACK`); these don't get re-exported and surfaces shouldn't import them.

The MODELS catalog (the data, distinct from the LLMModel schema) is *not* exported by ratchet. Each surface owns its own catalog because catalog content varies per-surface — PAI's 20 entries differ from keyhole's 17, and that's deliberate. Ratchet provides the LLMModel schema; surfaces fill in the data.

The TIERS registry *is* exported by ratchet because tier content is canonical — NPU Mid means the same thing everywhere. Surfaces compose their visible ladders by selecting from TIERS; they don't define new tiers (the `make_custom_tier()` factory is the only runtime tier construction path).

### Versioning and compatibility

Ratchet v0.2.0 is a minor version bump from v0.1.0. The contract:

- All v0.1.0 imports continue to work in v0.2.0. The Slider, SubsystemDemand, KpiResult, WorkloadRecord, and what-if primitives keep their existing signatures.
- New surface area (tiers, precision, projection, anchors, catalog, calibration) is purely additive.
- The probe APIs are unchanged. Nightjar's import of `ratchet.probes` continues to work.

Surfaces depend on a specific ratchet version. Recommended pin pattern in surface pyproject.toml:

```toml
dependencies = [
    "ratchet>=0.2.0,<0.3.0",
    # ...
]
```

The `<0.3.0` upper bound is important. When ratchet eventually bumps to v0.3.0 (drone-driven heterogeneous-architecture work), surfaces don't auto-upgrade — each surface deliberately bumps its pin when ready.

### Dependencies

Ratchet v0.2.0's required dependencies stay minimal:

```toml
[project]
name = "ratchet"
version = "0.2.0"
requires-python = ">=3.10"

dependencies = [
    "pyarrow>=14.0",   # WorkloadRecord schema + probe writers
]

[project.optional-dependencies]
gpu = [
    "nvidia-ml-py>=12.535",  # for GpuProbe; not needed by surfaces
]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "mypy>=1.5",
]
```

Notably *absent*: streamlit, pandas, numpy, pyyaml, tabulate. The anchor-secrets loader uses `st.secrets` when running inside a Streamlit context, but the loader's fallback path works without Streamlit. Ratchet itself doesn't need Streamlit; surfaces that use it do.

This keeps ratchet installable in non-Streamlit environments — including the future drone surface, the nightjar SITL stack, and any headless analysis pipeline.

---

## Section 3 — The Hardware dataclass

### Overview

The Hardware dataclass is the central engine object. It represents an NPU or GPU tier with its silicon facts, calibration constants, measurement attachments, and clone-tracking metadata. Every projection call consumes a Hardware instance. Every anchor overlay reads from one. Every UI tier-render derives from one.

The v0.2.0 Hardware is a union of PAI's and keyhole's Hardware definitions, plus a small number of new fields driven by the design inputs.

### Full field specification

```python
# ratchet/tiers/hardware.py
from dataclasses import dataclass, field
from typing import Optional

from ratchet.precision.capability import CapabilityInfo
from ratchet.calibration.source import CalibrationSource


@dataclass
class Hardware:
    """Generic compute-and-bandwidth spec for any NPU or GPU.

    A Hardware instance is mutable in principle (plain @dataclass, not frozen)
    but is treated as immutable in practice. The only mutation paths are:
      1. ratchet.tiers.memory_overlay.hw_with_memory() — creates a variant
         via dataclasses.replace() (functional copy, not in-place mutation).
      2. Measurement attachment at import time, on the reference tier only
         (typically RTX_5090_REFERENCE) — populated by sizer-specific
         measurement loaders that read sizer_bundle.json or equivalent.

    Surfaces SHOULD NOT mutate Hardware instances post-construction.
    """

    # ─── Silicon-fact fields (required, immutable in practice) ───
    name: str
    """Display name. Rewritten by hw_with_memory() to add the variant suffix
    (e.g., 'NPU Mid (LPDDR6 @ 14 GT/s)'). For canonical identity lookups,
    use tier_lookup_name property instead."""

    peak_tops_bf16: float
    """Raw peak BF16/FP16 TOPS. 0.0 means tier has no FP16 path
    (e.g., NPU Mid is INT8-only). FP16 and BF16 conflate to this field
    per common-silicon convention."""

    peak_tops_int8: float
    """Raw peak INT8 TOPS."""

    peak_tops_fp8: float
    """Raw peak FP8 TOPS. 0.0 on tiers without FP8 path."""

    mem_bandwidth_gbs: float
    """Raw peak DRAM bandwidth in GB/s. Effective BW =
    this × bandwidth_efficiency × npu_share. Rewritten by hw_with_memory()
    via the formula: new_bw = mem_bus_width_bits × mem_data_rate_gtps / 8."""

    mem_capacity_gb: float
    """DRAM capacity in GB."""

    mem_bus_width_bits: int
    """Memory bus width (32 / 64 / 128 / 512). Used by hw_with_memory()
    to recompute BW under data-rate swaps."""

    mem_type: str
    """One of: 'LPDDR4', 'LPDDR5', 'LPDDR5X', 'LPDDR5T', 'LPDDR6',
    'LPDDR7', 'GDDR6', 'GDDR6X', 'GDDR7', 'HBM3'."""

    mem_data_rate_gtps: float
    """DRAM I/O data rate in GT/s. Used in the BW recomputation formula
    on memory-upgrade clones. NOT used for anchor-secrets routing (that
    routes through bw_projected + tier_lookup_name instead, per ADR 010)."""

    # ─── Calibration constants (per-tier defaults, mutable per-tier) ───
    compute_efficiency: float = 0.65
    """Real-workload effective TOPS multiplier. effective_tops() returns
    raw_peak × compute_efficiency. Scales with silicon maturity:
    0.60 NPU Low / 0.65 NPU Mid / 0.70 NPU High / 0.70 RTX 5090.
    USED FOR VISION compute floor only — LLM compute floor uses raw peak
    with llm_prefill_util_factor instead, per ADR 015."""

    bandwidth_efficiency: float = 0.70
    """Real-workload BW multiplier. effective_bandwidth_gbs property
    returns mem_bandwidth_gbs × bandwidth_efficiency. 0.70 NPU tiers,
    0.85 RTX 5090 (dedicated VRAM has lower contention overhead)."""

    tdp_watts: float = 0.0
    """Thermal envelope; informational only. Not consumed by projection.
    Surfaces may display this in tier cards."""

    tier_family: Optional[str] = None
    """Memory-class taxonomy for same-class anchor scaling. Tiers in the
    same family share enough silicon characteristics that anchors measured
    on one tier can be BW-scaled within the family (🟡 same_class_anchor
    in the projection cascade). Values like 'LP5X-8.4-128b',
    'Neutron-32-LP5', 'GDDR7-28'. None means 'no family lookup available;
    fall through to cross-class (🔴) in projection."""

    compute_util_factor: float = 0.45
    """VISION compute-floor utilization factor. Per-tier calibration:
    0.19 Neutron-class / 0.45 NPU Mid / 0.50 NPU High / 0.85 RTX 5090.
    Multiplied against effective_tops() in vision projection compute floor.
    Vision-only; LLM uses llm_prefill_util_factor instead.
    Default 0.45 corresponds to NPU Mid (middle of the calibrated range)."""

    llm_prefill_util_factor: float = 0.10
    """LLM PREFILL compute-floor utilization factor. Multiplied against
    raw peak TOPS (NOT effective_tops) in LLM projection compute floor.
    Calibrated against raw peak per ADR 015 to avoid double-discounting
    with compute_efficiency. Typical value 0.10 on NPU silicon; reflects
    that LLM prefill achieves 5-15% of vendor peak due to small per-layer
    matmuls, MoE expert routing, and KV cache writes."""

    llm_decode_bw_realization: float = 1.0
    """LLM decode BW realization fraction. Defaults to 1.0 (pure BW
    ceiling — active_params_GB streaming at full effective BW).
    Held at default across tiers because realization is model-class-
    specific; using a measured MoE value would over-pessimize dense
    projections. Per-model calibration is captured in the anchor itself."""

    compute_overhead_ms: float = 1.0
    """Per-inference kernel-launch + sync overhead. NPU default 1.0;
    RTX 5090 0.3 (less overhead on dedicated PCIe device)."""

    npu_share_default: float = 0.75
    """Default fraction of memory bandwidth available to the NPU on an
    SoC with a shared memory bus. 0.75 for NPU SoC tiers (shared with
    CPU, GPU, ISP), 1.0 for RTX 5090 (dedicated VRAM, no contention).
    Surfaces may expose a user override via UI slider."""

    # ─── Per-dtype capability taxonomy ───
    capability_levels: Optional[dict[str, CapabilityInfo]] = None
    """Per-dtype 4-level capability dict. Keys are precision strings
    ('int8', 'fp8', 'bf16/fp16', 'q4_km'). Values are CapabilityInfo
    with level (tensor_native/tensor_compat/cuda_core/unsupported)
    and reason string. See ratchet/precision/capability.py.

    When None, hw_supports_dtype() falls back to the peak-TOPS heuristic
    (peak_tops_<dtype> > 0 → supported). Canonical tiers MUST set this;
    custom tiers MAY leave it None and accept the heuristic fallback."""

    # ─── Calibration provenance ───
    calibration_source: Optional[CalibrationSource] = None
    """Provenance metadata for the calibration constants on this tier.
    Encodes (method, reference, confidence) so surfaces can render
    appropriate banners. Canonical tiers carry measured calibration;
    custom tiers carry 'default' with low confidence. See
    ratchet/calibration/source.py."""

    # ─── Stock-identity tracking for memory-upgrade clones ───
    bw_projected: bool = False
    """True iff this Hardware was synthesized via hw_with_memory().
    Surfaces use this to mark BW-scaled projections as '(BW-proj)' in
    the UI. Anchor-secrets overlay short-circuits when True (memory-
    upgrade variants don't have their own measured anchors)."""

    stock_mem_bandwidth_gbs: Optional[float] = None
    """Snapshot of stock peak BW captured by hw_with_memory(). Lets
    projection hold prefill at stock under memory-upgrade overlays
    (prefill is compute-bound, not BW-bound, so memory upgrades don't
    affect it). None on stock tiers."""

    stock_name: Optional[str] = None
    """Snapshot of stock tier name captured by hw_with_memory(). The
    tier_lookup_name property returns stock_name when set, else name.
    Silicon-intrinsic lookups (precision capability, deployment path)
    key off the stock identity regardless of memory variant."""

    # ─── Measurement attachment (three flat fields, per ADR 009) ───
    measured_decode_overrides: Optional[dict[str, float]] = None
    """LLM decode tok/s overrides keyed by model_key. Per-tier-and-model
    measurements that override projection. Currently populated on
    NPU_MID with the Skippy MoE Q4_K_M anchor (37.85 tok/s). Public
    source-tree data; NOT anchor-secrets."""

    measured_prefill_overrides: Optional[dict[str, float]] = None
    """LLM prefill tok/s overrides keyed by model_key. Held at stock
    under memory-upgrade clones (prefill is compute-bound)."""

    measured_vision_overrides: Optional[dict[str, dict[str, float]]] = None
    """Vision measurements keyed by pipeline_key, then resolution.
    Inner dict shape: {'ms_per_inference': float, 'fps': float, ...}.
    Populated on tiers with vision bake-off data (NPU_LOW_LP5X,
    IMX95_MEASURED, RTX_5090_REFERENCE). Surface uses this to bypass
    projection entirely for matching (pipeline, resolution) pairs."""

    measured_llm: Optional[dict[str, dict[str, dict[str, float]]]] = None
    """Per-cell LLM measurements with full workload-scoped granularity.
    Shape: {model_key: {workload_id: {'decode_tok_s', 'prefill_tok_s',
    'ttft_s', 'host_ms'}}}.

    Populated only on the reference tier (RTX_5090_REFERENCE typically)
    by surface-specific bundle loaders. Read by projection's measured-
    cell path (🟢 measured) as the highest-priority resolution.

    NOT populated on NPU tiers in this version — those use
    measured_decode_overrides / measured_prefill_overrides for now."""
```

That's 24 fields total: 9 silicon facts, 8 calibration constants, 1 capability dict, 1 calibration provenance, 3 stock-identity fields, 4 measurement attachments. Each field has a docstring documenting its role.

### Methods and properties

```python
    @property
    def effective_bandwidth_gbs(self) -> float:
        """Raw peak BW × bandwidth_efficiency. Does NOT include npu_share
        (that composes downstream in projection)."""
        return self.mem_bandwidth_gbs * self.bandwidth_efficiency

    @property
    def tier_lookup_name(self) -> str:
        """Canonical identity for silicon-intrinsic lookups. Returns
        stock_name if this is a memory-upgrade clone, else name. Used by:
          - Anchor-secrets spec-cell routing
          - Precision-capability lookup
          - Deployment-path lookup
        Silicon caps don't change with memory swaps, so these lookups
        must key off the stock identity."""
        return self.stock_name if self.stock_name is not None else self.name

    def effective_tops(self, dtype: str) -> float:
        """Effective TOPS for a dtype = raw peak × compute_efficiency.
        FP16 routes to peak_tops_bf16 per dtype-conflation convention.
        Unknown dtype falls back to bf16.

        USED FOR VISION compute floor. LLM compute floor uses raw peak
        via hw_peak_tops_for_dtype() because llm_prefill_util_factor
        was calibrated against raw peak (ADR 015)."""
        from ratchet.precision.dtype_map import DTYPE_ATTR_MAP
        attr = DTYPE_ATTR_MAP.get(dtype.lower(), "peak_tops_bf16")
        return getattr(self, attr) * self.compute_efficiency

    def get_measured_llm_cell(
        self, model_key: str, workload_id: str
    ) -> Optional[dict]:
        """Resolve a per-cell LLM measurement. Checks measured_llm
        directly, then falls back to measurement_alias resolution
        via the catalog (alias-aware lookup).

        Returns None if no measurement is available. Callers should
        then fall through to tier-level overrides
        (measured_decode_overrides), then to projection."""
        if not self.measured_llm:
            return None
        cell = self.measured_llm.get(model_key, {}).get(workload_id)
        if cell is not None:
            return cell
        # Alias resolution happens at the call site (catalog-aware);
        # this method only handles the direct lookup.
        return None
```

Four methods total: two properties, two regular methods. Deliberately minimal. The `effective_tops()` method routes through `DTYPE_ATTR_MAP` from the precision submodule — keeps the dtype-attribute mapping in one place.

The catalog-alias resolution does *not* live on Hardware. The Hardware class doesn't import from the catalog submodule. Alias resolution happens at the projection call site, where both the catalog and Hardware are visible:

```python
# Pattern at the projection layer (ratchet/projection/llm.py):
def project_llm(hw: Hardware, model: LLMModel, ...):
    cell = hw.get_measured_llm_cell(model.key, workload_id)
    if cell is None and model.measurement_alias:
        cell = hw.get_measured_llm_cell(model.measurement_alias, workload_id)
    # ... then proceed with projection cascade
```

This keeps Hardware decoupled from the catalog — important for the future drone surface, which might not use the LLM catalog at all but still uses Hardware.

### Mutability discipline

The dataclass is mutable (plain `@dataclass`, not `frozen=True`). The decision *not* to freeze it is deliberate, with reasons:

1. The reference tier needs `measured_llm` populated at import time. Frozen dataclasses can't be mutated after construction; the alternative would be passing `measured_llm={...}` into the constructor, which works but creates an awkward two-step pattern (define the tier, then load measurements).

2. Memory-upgrade clones use `dataclasses.replace()` which works on frozen dataclasses too — frozen wouldn't break this path.

3. Existing surfaces (PAI sizer, keyhole-sizer) treat Hardware as mutable in practice. Freezing it now would require coordinated changes during retrofit.

The discipline rule: **mutate Hardware only at module import time, never during runtime.** Specifically:

- Tier constants get defined at module top level
- `measured_llm` gets populated by surface measurement loaders (these run at import)
- Beyond import, Hardware instances are treated as immutable

Mid-conversation mutation would be a bug. Surfaces that need per-session "what-if" tier modifications should construct new tiers via `make_custom_tier()` or `hw_with_memory()`, not mutate existing ones.

### Post-init validation

Hardware has no `__post_init__`. Validation happens at use sites where appropriate — for example, projection rejects tiers that lack required dtype support, and `make_custom_tier()` validates its inputs at construction time.

The choice not to validate in `__post_init__` is intentional: it would slow tier-registration at import time (validation cost per tier × number of tiers), and the validation rules belong with the consumers that care about them (projection cares about dtype support, UI cares about display fields, the registry cares about uniqueness).

If validation needs become more complex in the future (e.g., constraint that `mem_bandwidth_gbs == mem_bus_width_bits * mem_data_rate_gtps / 8`), a dedicated `validate_tier(hw: Hardware) -> list[ValidationError]` function in `ratchet/tiers/validation.py` is the right place — not `__post_init__`.

### Field-level rationale and ADR cross-references

Each field's design rationale traces back to one or more ADRs:

| Field | Driven by ADR |
|---|---|
| `peak_tops_*`, `mem_*` (silicon facts) | ADR 007 (canonical tier registry) |
| `compute_efficiency`, `bandwidth_efficiency` | Carried from v0.1.0 (workload modeling) |
| `compute_util_factor` vs `llm_prefill_util_factor` | ADR 015 (dtype dispatch + workload-class calibration) |
| `tier_family` | ADR 007 + ADR 009 (same-class anchor scaling) |
| `capability_levels` | ADR 008 (4-level taxonomy) |
| `calibration_source` | ADR 014 (calibration provenance) |
| `bw_projected`, `stock_name`, `stock_mem_bandwidth_gbs` | ADR 010 (stock identity tracking) |
| `measured_decode_overrides`, `measured_prefill_overrides`, `measured_vision_overrides`, `measured_llm` | ADR 009 (measurement attachment unification) |

This mapping is enforced: every non-trivial field references the ADR that justifies its presence. Adding fields in the future requires adding (or extending) an ADR.

---

## Section 4 — The TIERS registry

### Overview

The TIERS registry is the canonical source of truth for NPU and GPU tier definitions. It holds eight named Hardware instances representing the silicon classes the ecosystem sizes against. Surfaces compose their visible ladders by selecting from the registry; they don't define new tiers (except through the explicit `make_custom_tier()` factory, covered in section 6).

This is the architectural cut that lets multiple surfaces share canonical specifications while exposing different ladders. PAI sizer doesn't show i.MX 95 (LLM-only sizer doesn't size against vision-only silicon). Keyhole-sizer doesn't show Low-LP4 (vision workloads don't run on 2-TOPS silicon). Both surfaces show Mid, High, and 5090 with identical specs because they're the same silicon. The registry encodes the silicon facts once; the ladders encode which silicon each surface cares about.

### The canonical tier list

```python
# ratchet/tiers/registry.py

from ratchet.tiers.hardware import Hardware
from ratchet.precision.capability import (
    NEUTRON_INT8_ONLY_CAPABILITY,
    NPU_FULL_DTYPE_CAPABILITY,
    SM120_BLACKWELL_CAPABILITY,
)
from ratchet.calibration.source import CalibrationSource


# Calibration source constants used throughout. Each tier's calibration
# is tagged with provenance so surfaces can render appropriate banners.

_CANONICAL_SPEC_2026_05_14 = CalibrationSource(
    method="vendor_spec",
    reference="PAI deck Slide 11 (canonical, locked 2026-05-14)",
    confidence="high",
)

_BAKEOFF_MEASURED_2026_04 = CalibrationSource(
    method="measured",
    reference="5090 bake-off + Mid silicon measurements, 2026-04",
    confidence="high",
)

_IMX95_PRODUCTION = CalibrationSource(
    method="measured",
    reference="i.MX 95 production silicon, Kyle's NXP eIQ measurement",
    confidence="high",
)


# ─── Tier definitions ───────────────────────────────────────────────

NPU_LOW_LP4 = Hardware(
    name="NPU Low-LP4",
    peak_tops_bf16=0.0, peak_tops_int8=2.0, peak_tops_fp8=0.0,
    mem_bandwidth_gbs=12.8, mem_capacity_gb=8.0,
    mem_bus_width_bits=32, mem_type="LPDDR4", mem_data_rate_gtps=3.2,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=5.0,
    tier_family="Neutron-32-LP4",
    compute_util_factor=0.19,
    llm_prefill_util_factor=0.10,
    capability_levels=NEUTRON_INT8_ONLY_CAPABILITY,
    calibration_source=_CANONICAL_SPEC_2026_05_14,
)
"""The cheapest edge NPU class: 2-TOPS INT8-only silicon on 32-bit LPDDR4."""


NPU_LOW_LP5_32BIT = Hardware(
    name="NPU Low-LP5-32bit",
    peak_tops_bf16=0.0, peak_tops_int8=2.0, peak_tops_fp8=0.0,
    mem_bandwidth_gbs=25.6, mem_capacity_gb=8.0,
    mem_bus_width_bits=32, mem_type="LPDDR5", mem_data_rate_gtps=6.4,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=5.0,
    tier_family="Neutron-32-LP5",
    compute_util_factor=0.19,
    llm_prefill_util_factor=0.10,
    capability_levels=NEUTRON_INT8_ONLY_CAPABILITY,
    calibration_source=_CANONICAL_SPEC_2026_05_14,
)


NPU_LOW_LP5_64BIT = Hardware(
    name="NPU Low-LP5-64bit",
    peak_tops_bf16=0.0, peak_tops_int8=2.0, peak_tops_fp8=0.0,
    mem_bandwidth_gbs=51.2, mem_capacity_gb=16.0,
    mem_bus_width_bits=64, mem_type="LPDDR5", mem_data_rate_gtps=6.4,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=6.0,
    tier_family="Neutron-64-LP5",
    compute_util_factor=0.19,
    llm_prefill_util_factor=0.10,
    capability_levels=NEUTRON_INT8_ONLY_CAPABILITY,
    calibration_source=_CANONICAL_SPEC_2026_05_14,
)


NPU_LOW_LP5X = Hardware(
    name="NPU Low-LP5X",
    peak_tops_bf16=50.0, peak_tops_int8=100.0, peak_tops_fp8=100.0,
    mem_bandwidth_gbs=67.2, mem_capacity_gb=16.0,
    mem_bus_width_bits=64, mem_type="LPDDR5X", mem_data_rate_gtps=8.4,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=10.0,
    tier_family="LP5X-8.4-64b",
    compute_util_factor=0.19,
    llm_prefill_util_factor=0.10,
    capability_levels=NPU_FULL_DTYPE_CAPABILITY,
    calibration_source=_CANONICAL_SPEC_2026_05_14,
)


IMX95_MEASURED = Hardware(
    name="NPU i.MX 95 (ground truth)",
    peak_tops_bf16=0.0, peak_tops_int8=2.0, peak_tops_fp8=0.0,
    mem_bandwidth_gbs=25.6, mem_capacity_gb=16.0,
    mem_bus_width_bits=32, mem_type="LPDDR5", mem_data_rate_gtps=6.4,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=8.0,
    tier_family="Neutron-32-LP5",
    compute_util_factor=0.19,
    llm_prefill_util_factor=0.10,
    capability_levels=NEUTRON_INT8_ONLY_CAPABILITY,
    calibration_source=_IMX95_PRODUCTION,
    measured_vision_overrides={
        "yolov8n_trt_int8_coco128": {
            "1920x1080": {"ms_per_inference": 32.0, "fps": 31.25},
        },
    },
)


NPU_MID = Hardware(
    name="NPU Mid",
    peak_tops_bf16=0.0, peak_tops_int8=200.0, peak_tops_fp8=0.0,
    mem_bandwidth_gbs=134.4, mem_capacity_gb=24.0,
    mem_bus_width_bits=128, mem_type="LPDDR5X", mem_data_rate_gtps=8.4,
    compute_efficiency=0.65, bandwidth_efficiency=0.70,
    tdp_watts=25.0,
    tier_family="LP5X-8.4-128b",
    compute_util_factor=0.45,
    llm_prefill_util_factor=0.10,
    capability_levels=NEUTRON_INT8_ONLY_CAPABILITY,
    calibration_source=_BAKEOFF_MEASURED_2026_04,
    measured_decode_overrides={"qwen3_30b_a3b_moe": 37.85},
    measured_prefill_overrides={"qwen3_30b_a3b_moe": 2849.0},
)
"""200 eTOPS INT8-only on 128-bit LPDDR5X-8.4. Carries Skippy MoE Q4_K_M anchor."""


NPU_HIGH = Hardware(
    name="NPU High",
    peak_tops_bf16=200.0, peak_tops_int8=400.0, peak_tops_fp8=400.0,
    mem_bandwidth_gbs=134.4, mem_capacity_gb=32.0,
    mem_bus_width_bits=128, mem_type="LPDDR5X", mem_data_rate_gtps=8.4,
    compute_efficiency=0.70, bandwidth_efficiency=0.70,
    tdp_watts=35.0,
    tier_family="LP5X-8.4-128b",
    compute_util_factor=0.50,
    llm_prefill_util_factor=0.11,
    capability_levels=NPU_FULL_DTYPE_CAPABILITY,
    calibration_source=_BAKEOFF_MEASURED_2026_04,
)


RTX_5090_REFERENCE = Hardware(
    name="RTX 5090 (reference, measured)",
    peak_tops_bf16=209.0, peak_tops_int8=419.0, peak_tops_fp8=419.0,
    mem_bandwidth_gbs=1792.0, mem_capacity_gb=32.0,
    mem_bus_width_bits=512, mem_type="GDDR7", mem_data_rate_gtps=28.0,
    compute_efficiency=0.70, bandwidth_efficiency=0.85,
    tdp_watts=575.0,
    tier_family="GDDR7-28",
    compute_util_factor=0.85,
    llm_prefill_util_factor=0.10,
    compute_overhead_ms=0.3,
    npu_share_default=1.0,
    capability_levels=SM120_BLACKWELL_CAPABILITY,
    calibration_source=_BAKEOFF_MEASURED_2026_04,
    measured_llm={},  # Populated at import time by surface bundle loaders
    measured_vision_overrides={},
)


# ─── The registry dict ──────────────────────────────────────────────

TIERS: dict[str, Hardware] = {
    t.name: t for t in (
        NPU_LOW_LP4,
        NPU_LOW_LP5_32BIT,
        NPU_LOW_LP5_64BIT,
        NPU_LOW_LP5X,
        IMX95_MEASURED,
        NPU_MID,
        NPU_HIGH,
        RTX_5090_REFERENCE,
    )
}
```

### Per-surface ladder composition

Surfaces compose visible ladders by selecting from TIERS:

```python
# personal-ai-assistant-sizer/sizer/ladder.py
from ratchet import TIERS

PAI_LADDER = [TIERS[n] for n in (
    "NPU Low-LP4",
    "NPU Low-LP5-32bit",
    "NPU Low-LP5-64bit",
    "NPU Low-LP5X",
    "NPU Mid",
    "NPU High",
    "RTX 5090 (reference, measured)",
)]
# PAI does NOT include 'NPU i.MX 95 (ground truth)' — LLM-only sizer.

# keyhole-sizer/sizer/ladder.py
from ratchet import TIERS

KEYHOLE_LADDER = [TIERS[n] for n in (
    "NPU i.MX 95 (ground truth)",
    "NPU Low-LP5-64bit",
    "NPU Low-LP5X",
    "NPU Mid",
    "NPU High",
    "RTX 5090 (reference, measured)",
)]
# Keyhole does NOT include the Low-LP4 / Low-LP5-32bit tiers.
```

If a surface needs a tier that doesn't exist in TIERS, the answer is *not* to define a local Hardware instance — it's to add the tier to ratchet's registry.

### What goes in the registry vs the catalog

Two distinct concepts:

**The TIERS registry** is canonical because tier facts are invariant. NPU Mid is 200 eTOPS INT8-only on 128-bit LPDDR5X-8.4 regardless of which surface is asking. Surfaces don't redefine this.

**The MODELS catalog** is per-surface because catalog content varies. PAI has 20 entries; keyhole has 17. Each surface owns its catalog. Ratchet provides the `LLMModel` schema, not the catalog data.

### Memory upgrade overlays

The `hw_with_memory()` function in `ratchet/tiers/memory_overlay.py` produces variant Hardware instances by swapping memory characteristics on a stock tier:

```python
def hw_with_memory(
    hw: Hardware,
    mem_type: str,
    mem_data_rate_gtps: float,
    name_suffix: Optional[str] = None,
) -> Hardware:
    """Return a Hardware variant with memory subsystem swapped.

    Recomputes mem_bandwidth_gbs from new data rate against same bus width.
    Sets bw_projected=True and captures stock identity (stock_name,
    stock_mem_bandwidth_gbs) so silicon-intrinsic lookups still resolve.

    BW-scales measured_decode_overrides linearly. Holds
    measured_prefill_overrides at stock (prefill is compute-bound)."""
    new_bw = hw.mem_bus_width_bits * mem_data_rate_gtps / 8.0
    bw_ratio = new_bw / hw.mem_bandwidth_gbs

    new_name = (
        f"{hw.name} ({name_suffix})"
        if name_suffix
        else f"{hw.name} ({mem_type} @ {mem_data_rate_gtps} GT/s)"
    )

    new_decode_overrides = (
        {k: v * bw_ratio for k, v in hw.measured_decode_overrides.items()}
        if hw.measured_decode_overrides
        else None
    )

    return dataclasses.replace(
        hw,
        name=new_name,
        mem_type=mem_type,
        mem_data_rate_gtps=mem_data_rate_gtps,
        mem_bandwidth_gbs=new_bw,
        bw_projected=True,
        stock_name=hw.stock_name if hw.stock_name else hw.name,
        stock_mem_bandwidth_gbs=hw.mem_bandwidth_gbs,
        measured_decode_overrides=new_decode_overrides,
        # measured_prefill_overrides inherited unchanged
    )


MEMORY_UPGRADE_OPTIONS: list[tuple[str, str, float]] = [
    ("LPDDR5T @ 11.2 GT/s", "LPDDR5T", 11.2),
    ("LPDDR6 @ 12 GT/s", "LPDDR6", 12.0),
    ("LPDDR6 @ 14 GT/s", "LPDDR6", 14.0),
]
```

The memory overlay encodes one specific architectural insight: on a memory-bound workload like LLM decode, BW scaling translates directly to throughput scaling. Prefill is compute-bound; upgrading memory doesn't help it. The function encodes this asymmetry structurally.

---

## Section 5 — The precision/capability model

### Overview

The precision/capability submodule answers two related but distinct questions about an NPU tier:

**Question 1:** Does this tier have *any* way to execute this dtype? (Binary: yes / no)

**Question 2:** *How well* does this tier execute this dtype? (4-level: native tensor cores / binary-compat tensor cores / general-purpose compute / not at all)

These are different questions with different answers in subtle cases. The classic example is consumer Blackwell (SM120) on INT8: the silicon has INT8 tensor cores accessible via sm80 IMMA binary compatibility, so question 1 is yes; but vLLM's CUTLASS LLM serving compiles fresh kernels per architecture and SM120 templates don't exist yet, so for fresh-compile workloads question 2 is `tensor_compat`.

Ratchet exposes both surfaces and lets consumers ask whichever question matches their need.

### The 4-level taxonomy

```python
# ratchet/precision/capability.py

from dataclasses import dataclass
from enum import Enum


class CapabilityLevel(Enum):
    TENSOR_NATIVE = "tensor_native"
    """Native tensor-core execution. The silicon has dedicated hardware
    for this precision (e.g., Hopper FP8, Neutron INT8)."""

    TENSOR_COMPAT = "tensor_compat"
    """Tensor-core execution via binary compatibility. The silicon's
    tensor cores run kernels compiled for an earlier architecture
    (e.g., sm80 IMMA INT8 on sm120 Blackwell). Works for pre-compiled
    workloads (TRT engines) but blocked for fresh-compile (vLLM CUTLASS)."""

    CUDA_CORE = "cuda_core"
    """General-purpose compute execution (DP4A or equivalent). Works
    but significantly slower than tensor cores. Currently unused by any
    canonical tier; reserved for future silicon classes."""

    UNSUPPORTED = "unsupported"
    """Cannot execute this precision at all."""

    def __bool__(self) -> bool:
        return self is not CapabilityLevel.UNSUPPORTED


@dataclass(frozen=True)
class CapabilityInfo:
    """Per-(tier, dtype) capability with provenance.

    level: how fast (the 4-level enum)
    reason: surface-rendering tooltip-grade explanation"""

    level: CapabilityLevel
    reason: str

    def __bool__(self) -> bool:
        return bool(self.level)
```

### The canonical capability tables

```python
NEUTRON_INT8_ONLY_CAPABILITY: dict[str, CapabilityInfo] = {
    "int8":      CapabilityInfo(CapabilityLevel.TENSOR_NATIVE,
                                "INT8 tensor cores native to Neutron NPU"),
    "fp8":       CapabilityInfo(CapabilityLevel.UNSUPPORTED,
                                "Neutron NPU has no FP path"),
    "bf16/fp16": CapabilityInfo(CapabilityLevel.UNSUPPORTED,
                                "Neutron NPU has no FP path"),
    "q4_km":     CapabilityInfo(CapabilityLevel.TENSOR_NATIVE,
                                "Q4_K_M weight-only quant runs via INT8 dequant path"),
}


NPU_FULL_DTYPE_CAPABILITY: dict[str, CapabilityInfo] = {
    "int8":      CapabilityInfo(CapabilityLevel.TENSOR_NATIVE, "INT8 tensor cores"),
    "fp8":       CapabilityInfo(CapabilityLevel.TENSOR_NATIVE, "FP8 tensor cores"),
    "bf16/fp16": CapabilityInfo(CapabilityLevel.TENSOR_NATIVE, "BF16/FP16 tensor cores"),
    "q4_km":     CapabilityInfo(CapabilityLevel.TENSOR_NATIVE,
                                "Q4_K_M weight-only quant runs via FP16 dequant path"),
}


SM120_BLACKWELL_CAPABILITY: dict[str, CapabilityInfo] = {
    "int8":      CapabilityInfo(CapabilityLevel.TENSOR_COMPAT,
                                "sm80 IMMA via binary compat; vLLM CUTLASS fresh-compile blocked"),
    "fp8":       CapabilityInfo(CapabilityLevel.TENSOR_NATIVE, "FP8 native to SM120"),
    "bf16/fp16": CapabilityInfo(CapabilityLevel.TENSOR_NATIVE, "BF16/FP16 native to SM120"),
    "q4_km":     CapabilityInfo(CapabilityLevel.TENSOR_NATIVE,
                                "Q4_K_M weight-only quant runs via FP16 dequant path"),
}
```

### The precision-key set

Four canonical precision keys: `int8`, `fp8`, `bf16/fp16`, `q4_km`. Not five — the `bf16/fp16` conflation is intentional because both precisions map to the same tensor-core class on every silicon in the canonical registry.

`q4_km` as a peer precision encodes the architectural choice that weight-only quants need their own capability tracking because their support depends on the underlying dequant path: INT8 dequant on Neutron silicon, FP16 dequant on FP-capable silicon.

### The dtype-attribute map

Distinct from but related to capability, the `DTYPE_ATTR_MAP` routes dtype strings to Hardware fields for raw-peak-TOPS lookups:

```python
# ratchet/precision/dtype_map.py

DTYPE_ATTR_MAP: dict[str, str] = {
    "int8": "peak_tops_int8",
    "fp8":  "peak_tops_fp8",
    "bf16": "peak_tops_bf16",
    "fp16": "peak_tops_bf16",  # fp16 conflates to bf16 field
}


def hw_peak_tops_for_dtype(hw: Hardware, dtype: str) -> float:
    """Raw peak TOPS for a dtype, without compute_efficiency multiplier.

    LLM cross-class compute floor uses this against llm_prefill_util_factor
    (calibrated against raw peak per ADR 015). Vision uses effective_tops()."""
    attr = DTYPE_ATTR_MAP.get(dtype.lower())
    if attr is None:
        return 0.0
    return float(getattr(hw, attr, 0.0))


def hw_supports_dtype(hw: Hardware, dtype: str) -> CapabilityLevel:
    """Capability level for a dtype on a Hardware tier.

    Reads from hw.capability_levels when populated. Falls back to peak-TOPS
    heuristic when None."""
    dt = dtype.lower()
    cap_key = "bf16/fp16" if dt in ("bf16", "fp16") else dt
    if hw.capability_levels is not None:
        info = hw.capability_levels.get(cap_key)
        if info is not None:
            return info.level
        return CapabilityLevel.UNSUPPORTED
    return (
        CapabilityLevel.TENSOR_NATIVE
        if hw_peak_tops_for_dtype(hw, dt) > 0.0
        else CapabilityLevel.UNSUPPORTED
    )
```

### The deployment-path classifier

```python
# ratchet/precision/deployment_path.py

WorkloadKernelSource = Literal["precompiled", "fresh_compile"]


def deployment_path_for_tier(
    hw: Hardware,
    dtype: str,
    workload_kernel_source: WorkloadKernelSource,
) -> str:
    """Classify deployment path for (tier, dtype, workload-kernel-source).

    Returns: 'native_fast' | 'compat_fast' | 'compat_blocked' |
             'cuda_core_fallback' | 'unsupported'

    Exists because tensor_compat means different things to different
    workload categories — pre-compiled workloads see a fast path,
    fresh-compile workloads see a blocker."""

    cap = hw_supports_dtype(hw, dtype)

    if cap is CapabilityLevel.TENSOR_NATIVE:
        return "native_fast"
    if cap is CapabilityLevel.TENSOR_COMPAT:
        return (
            "compat_fast"
            if workload_kernel_source == "precompiled"
            else "compat_blocked"
        )
    if cap is CapabilityLevel.CUDA_CORE:
        return "cuda_core_fallback"
    return "unsupported"
```

This is the function that surfaces consume for retargeting-cost lookups and UI categorization. PAI's existing deployment_path classifier reduces `tensor_compat` to "blocked" (LLM-focused, fresh-compile via vLLM); the parameterized version preserves PAI's behavior with `workload_kernel_source="fresh_compile"` and adds keyhole-friendly behavior for TRT vision pipelines.

---

## Section 6 — Calibration provenance and custom tiers

### Overview

Calibration provenance answers a question users will ask of every projected number: *"How much should I trust this?"* A 37.85 tok/s decode figure from measured silicon means something different than 37.85 tok/s from a custom-tier projection using default calibration constants. The provenance system encodes the answer structurally, so surfaces can render appropriate confidence banners.

Custom tier construction is the corollary need: when a user wants to size against silicon that isn't in the canonical registry, the engine must produce a Hardware instance with the right calibration defaults. Silently optimistic defaults would over-project a 2-TOPS Neutron chip by ~5×.

### The CalibrationSource dataclass

```python
# ratchet/calibration/source.py

from dataclasses import dataclass
from typing import Literal

CalibrationMethod = Literal[
    "measured",       # Calibrated against real silicon measurements
    "interpolated",   # Calibrated by interpolating between measured tiers
    "vendor_spec",    # Derived from vendor-published specs only
    "default",        # Engine defaults; not calibrated for the specific silicon
]


CalibrationConfidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class CalibrationSource:
    """Provenance metadata for a Hardware tier's calibration constants."""

    method: CalibrationMethod
    reference: str
    confidence: CalibrationConfidence

    def __post_init__(self):
        # Light validation: default-method tiers must have low confidence.
        if self.method == "default" and self.confidence != "low":
            raise ValueError(
                "CalibrationSource: method='default' requires confidence='low'. "
                "Calibration defaults are not calibrated."
            )
```

### Provenance values for canonical tiers

| Tier | Method | Confidence | Reference |
|---|---|---|---|
| `NPU_LOW_LP4` | `vendor_spec` | `high` | PAI deck Slide 11 |
| `NPU_LOW_LP5_32BIT` | `vendor_spec` | `high` | PAI deck Slide 11 |
| `NPU_LOW_LP5_64BIT` | `vendor_spec` | `high` | PAI deck Slide 11 |
| `NPU_LOW_LP5X` | `vendor_spec` | `high` | PAI deck Slide 11 |
| `IMX95_MEASURED` | `measured` | `high` | i.MX 95 production silicon |
| `NPU_MID` | `measured` | `high` | 5090 bake-off + Mid measurements, 2026-04 |
| `NPU_HIGH` | `measured` | `high` | 5090 bake-off + Mid measurements, 2026-04 |
| `RTX_5090_REFERENCE` | `measured` | `high` | 5090 bake-off + Mid measurements, 2026-04 |

`medium` confidence is reserved for tiers where vendor-spec data is partial or in dispute. `low` confidence is exclusively for custom tiers using engine defaults.

### The custom-tier factory

```python
# ratchet/tiers/custom.py

SiliconClass = Literal[
    "neutron",        # INT8-only Neutron-class
    "lp5x_64",        # Full-dtype on LPDDR5X 64-bit
    "lp5x_128",       # Full-dtype on LPDDR5X 128-bit
    "lp5x_128_int8",  # INT8-only on LPDDR5X 128-bit (NPU Mid)
    "gddr_class",     # Discrete GPU with dedicated VRAM
    "unknown",        # No analogous silicon class; engine defaults
]


def make_custom_tier(
    name: str,
    *,
    silicon_class: SiliconClass,
    peak_tops_int8: float,
    mem_bandwidth_gbs: float,
    mem_capacity_gb: float,
    mem_bus_width_bits: int,
    mem_type: str,
    mem_data_rate_gtps: float,
    peak_tops_bf16: float = 0.0,
    peak_tops_fp8: float = 0.0,
    compute_efficiency: Optional[float] = None,
    bandwidth_efficiency: Optional[float] = None,
    tdp_watts: float = 0.0,
) -> Hardware:
    """Construct a custom Hardware tier with silicon-class-defaulted calibration.

    The silicon_class parameter selects defaults for compute_util_factor,
    llm_prefill_util_factor, tier_family, and capability_levels. Defaults
    are appropriate for the silicon family but NOT calibrated for the
    specific user-defined chip. calibration_source is set to method='default'
    with confidence='low' so surfaces render appropriate warnings."""

    defaults = SILICON_CLASS_DEFAULTS[silicon_class]

    return Hardware(
        name=name,
        peak_tops_bf16=peak_tops_bf16,
        peak_tops_int8=peak_tops_int8,
        peak_tops_fp8=peak_tops_fp8,
        mem_bandwidth_gbs=mem_bandwidth_gbs,
        mem_capacity_gb=mem_capacity_gb,
        mem_bus_width_bits=mem_bus_width_bits,
        mem_type=mem_type,
        mem_data_rate_gtps=mem_data_rate_gtps,
        compute_efficiency=(
            compute_efficiency if compute_efficiency is not None
            else defaults["compute_efficiency"]
        ),
        bandwidth_efficiency=(
            bandwidth_efficiency if bandwidth_efficiency is not None
            else defaults["bandwidth_efficiency"]
        ),
        tdp_watts=tdp_watts,
        tier_family=defaults["tier_family"],
        compute_util_factor=defaults["compute_util_factor"],
        llm_prefill_util_factor=defaults["llm_prefill_util_factor"],
        llm_decode_bw_realization=defaults["llm_decode_bw_realization"],
        compute_overhead_ms=defaults["compute_overhead_ms"],
        npu_share_default=defaults["npu_share_default"],
        capability_levels=defaults["capability_levels"],
        calibration_source=CalibrationSource(
            method="default",
            reference=f"silicon_class={silicon_class}, engine defaults",
            confidence="low",
        ),
    )
```

### The silicon-class defaults table

```python
# ratchet/calibration/silicon_class.py

SILICON_CLASS_DEFAULTS: dict[str, dict[str, Any]] = {
    "neutron": {
        "compute_efficiency": 0.60,
        "bandwidth_efficiency": 0.70,
        "tier_family": "Neutron-custom",
        "compute_util_factor": 0.19,
        "llm_prefill_util_factor": 0.10,
        "llm_decode_bw_realization": 1.0,
        "compute_overhead_ms": 1.0,
        "npu_share_default": 0.75,
        "capability_levels": NEUTRON_INT8_ONLY_CAPABILITY,
    },
    "lp5x_64": {
        "compute_efficiency": 0.60,
        "bandwidth_efficiency": 0.70,
        "tier_family": "LP5X-custom-64b",
        "compute_util_factor": 0.19,
        "llm_prefill_util_factor": 0.10,
        "llm_decode_bw_realization": 1.0,
        "compute_overhead_ms": 1.0,
        "npu_share_default": 0.75,
        "capability_levels": NPU_FULL_DTYPE_CAPABILITY,
    },
    "lp5x_128": {
        "compute_efficiency": 0.70,
        "bandwidth_efficiency": 0.70,
        "tier_family": "LP5X-custom-128b",
        "compute_util_factor": 0.50,
        "llm_prefill_util_factor": 0.11,
        "llm_decode_bw_realization": 1.0,
        "compute_overhead_ms": 1.0,
        "npu_share_default": 0.75,
        "capability_levels": NPU_FULL_DTYPE_CAPABILITY,
    },
    "lp5x_128_int8": {
        "compute_efficiency": 0.65,
        "bandwidth_efficiency": 0.70,
        "tier_family": "LP5X-custom-128b-int8",
        "compute_util_factor": 0.45,
        "llm_prefill_util_factor": 0.10,
        "llm_decode_bw_realization": 1.0,
        "compute_overhead_ms": 1.0,
        "npu_share_default": 0.75,
        "capability_levels": NEUTRON_INT8_ONLY_CAPABILITY,
    },
    "gddr_class": {
        "compute_efficiency": 0.70,
        "bandwidth_efficiency": 0.85,
        "tier_family": "GDDR-custom",
        "compute_util_factor": 0.85,
        "llm_prefill_util_factor": 0.10,
        "llm_decode_bw_realization": 1.0,
        "compute_overhead_ms": 0.3,
        "npu_share_default": 1.0,
        "capability_levels": SM120_BLACKWELL_CAPABILITY,
    },
    "unknown": {
        "compute_efficiency": 0.65,
        "bandwidth_efficiency": 0.70,
        "tier_family": "unknown-custom",
        "compute_util_factor": 0.45,
        "llm_prefill_util_factor": 0.10,
        "llm_decode_bw_realization": 1.0,
        "compute_overhead_ms": 1.0,
        "npu_share_default": 0.75,
        "capability_levels": None,
    },
}
```

### The compute-floor over-projection risk

Default `compute_util_factor=0.45` calibrated against NPU Mid would over-project a 2-TOPS Neutron chip by 2.5× (correct value is 0.19). The `silicon_class` parameter prevents this — `silicon_class="neutron"` selects `compute_util_factor=0.19`. Users who pick the wrong class (or `silicon_class="unknown"`) get an explicit calibration warning via the low-confidence CalibrationSource.

---

## Section 7 — Measurement attachment

### Overview

Measurement attachment is how real-silicon performance data gets connected to a Hardware tier, overriding projection when present. Section 5's projection cascade defines four resolution paths:

1. **🟢 measured** — per-cell measurement on this exact tier and workload
2. **🟢 measured_anchor** — per-tier measurement on this tier (workload-agnostic)
3. **🟡 same_class_anchor** — measurement on a sibling tier in the same family, BW-scaled
4. **🔴 cross_class** — first-principles projection (no measurement available)

This section defines the data structures and discipline that make those paths work.

### The three measurement-attachment fields

Hardware carries three flat fields for source-tree-resident measurements:

```python
measured_decode_overrides: Optional[dict[str, float]] = None
"""LLM decode tok/s overrides keyed by model_key. Workload-agnostic.
Used by the 🟢 measured_anchor path."""

measured_prefill_overrides: Optional[dict[str, float]] = None
"""LLM prefill tok/s overrides keyed by model_key. Held at stock under
memory-upgrade clones (prefill is compute-bound). Used by 🟢 measured_anchor."""

measured_vision_overrides: Optional[dict[str, dict[str, float]]] = None
"""Vision pipeline measurements keyed by pipeline_key then resolution."""
```

And one nested-dict field for per-cell granularity:

```python
measured_llm: Optional[dict[str, dict[str, dict[str, float]]]] = None
"""Per-cell LLM measurements. Shape: {model_key: {workload_id:
{decode_tok_s, prefill_tok_s, ttft_s, host_ms}}}. Populated only on the
reference tier by surface-specific bundle loaders. Used by 🟢 measured."""
```

### Why four fields and not one unified dict

- **Type narrowing is cleaner with flat fields.** Named fields support IDE autocomplete; unified dict returns Any.
- **Code paths differ.** LLM decode override and vision measurements are accessed by completely different code. No shared consumer.
- **Future workload classes add fields, not dict keys.** If audio shows up, add `measured_audio_overrides`. Naming reads consistently.

### The measurement-data interchange schema

**`measured_decode_overrides` and `measured_prefill_overrides`:**
```
{model_key: float}  # tok/s number
```

**`measured_vision_overrides`:**
```
{pipeline_key: {resolution_key: {
    "ms_per_inference": float,   # required
    "fps": float,                 # derived
    "ms_per_inference_p95": float # optional
}}}
```

**`measured_llm`:**
```
{model_key: {workload_id: {
    "decode_tok_s": float,       # required for 🟢 measured
    "prefill_tok_s": float,      # optional
    "ttft_s": float,
    "host_ms": float
}}}
```

### The four canonical measurement paths

| Path | Privacy | Storage | Populated at | Used by |
|---|---|---|---|---|
| **1. Per-cell LLM bake-off** | Public | `measured_llm` field | Import time, surface-specific loader | 🟢 measured |
| **2. Tier-level LLM override** | Public | `measured_decode_overrides` + `measured_prefill_overrides` | Tier construction in registry | 🟢 measured_anchor |
| **3. Per-cell vision bake-off** | Public | `measured_vision_overrides` | Tier construction or import-time loader | Vision projection direct lookup |
| **4. Private silicon anchor** | **Private** | Result-dict overlay (NOT on Hardware) | Runtime, from `.streamlit/secrets.toml` | 🟢 measured_silicon_anchor |

Paths 1-3 are source-tree-resident. Path 4 is runtime-loaded from gitignored secrets. The Hardware object never holds private anchor data.

### Population patterns

**Static population at registry construction.** Tiers declare measurements inline (NPU_MID's Skippy MoE Q4_K_M anchor).

**Surface-specific import-time loading.** RTX_5090_REFERENCE ships with empty `measured_llm={}` and `measured_vision_overrides={}`. Surfaces populate via measurement loaders at module import:

```python
# personal-ai-assistant-sizer/sizer/measured.py
from ratchet import RTX_5090_REFERENCE

def attach_measurements_to_reference():
    bundle = load_bundle()
    measured = {}
    for model_canonical, workloads in bundle["models"].items():
        for workload_id, cell in workloads.items():
            measured.setdefault(model_canonical, {})[workload_id] = cell
    RTX_5090_REFERENCE.measured_llm = measured

attach_measurements_to_reference()
```

This is the only place in the design where Hardware gets mutated after construction. Mutation happens once per surface at import; never during runtime.

### Memory-upgrade overlay semantics

| Field | Behavior under memory upgrade |
|---|---|
| `measured_decode_overrides` | BW-scaled by ratio (decode is BW-bound on MoE) |
| `measured_prefill_overrides` | **Held at stock** (prefill is compute-bound) |
| `measured_vision_overrides` | **Inherited unchanged** (vision overrides are silicon-intrinsic) |
| `measured_llm` | Per-cell: `decode_tok_s` BW-scaled, `prefill_tok_s` and `ttft_s` held |

This asymmetry encodes the architectural insight that decode is BW-bound and prefill is compute-bound, applied structurally.

### Anti-pattern: surface-specific measurement fields on Hardware

Surfaces don't extend Hardware with their own measurement fields. If measurements don't fit the four canonical fields, the right move is one of:

1. Propose extending Hardware in ratchet (normal release)
2. Store off-Hardware in surface state
3. Use the anchor-secrets overlay pattern

This discipline matters because Hardware is the central engine object. Letting it grow surface-specific extensions destroys the canonical contract.

---

## Section 8 — The projection API

### Overview

The projection API consolidates PAI sizer's `project_llm` and keyhole-sizer's equivalent, with structural refinements: 4-path resolution, memory feasibility as precondition, dtype compatibility check, workload-pattern multipliers as optional overlay, structured result types, compiler quality parameter, NPU share affecting BW-bound paths only.

### The result types

```python
# ratchet/projection/result.py

SourceLabel = Literal[
    "measured", "measured_anchor", "same_class_anchor",
    "cross_class", "measured_silicon_anchor",
]


Regime = Literal["bw_bound", "compute_bound"]


@dataclass(frozen=True)
class Projected:
    """A successful projection result."""

    # Headline numbers
    decode_tok_s: float
    prefill_tok_s: float
    ttft_s: float

    # Workload context
    decode_tokens: int
    prompt_tokens: int
    decode_s: float
    prefill_s: float
    total_s: float
    host_ms: float

    # Classification
    source: SourceLabel
    regime: Regime

    # Tier and model context
    hw_name: str
    model_key: str
    workload_id: str

    # Optional diagnostic fields
    decode_ceiling_tok_s: Optional[float] = None
    base_decode_pre_multiplier: Optional[float] = None
    silicon_anchor_meta: Optional[dict] = None


@dataclass(frozen=True)
class WontFit:
    """Model cannot fit on hardware at this context length."""
    hw_name: str
    model_key: str
    workload_id: str
    required_gb: float
    available_gb: float
    headroom_gb: float
    breakdown: dict
    prompt_tokens: int
    decode_tokens: int


@dataclass(frozen=True)
class DtypeMismatch:
    """Model requires a compute dtype this tier doesn't support."""
    hw_name: str
    model_key: str
    workload_id: str
    required_dtype: str
    tier_capability: str
    retargeting_hint: Optional[str] = None


ProjectionResult = Union[Projected, WontFit, DtypeMismatch]
```

### Memory feasibility precondition

```python
# ratchet/projection/feasibility.py

RUNTIME_OVERHEAD_BYTES = 1_000_000_000


FeasibilityVerdict = Literal["fits", "tight", "wont_fit"]


@dataclass(frozen=True)
class FeasibilityCheck:
    verdict: FeasibilityVerdict
    required_gb: float
    available_gb: float
    headroom_gb: float
    breakdown: dict


def kv_cache_bytes_per_token(model: LLMModel, dtype_bytes: int = 2) -> float:
    """KV cache bytes per token, computed from transformer geometry.
    Uses GQA ratio when available."""
    kv_heads = model.num_kv_heads or model.num_attention_heads or 1
    attn_heads = max(model.num_attention_heads or 1, 1)
    gqa_ratio = kv_heads / attn_heads
    return model.num_layers * 2 * model.hidden_dim * gqa_ratio * dtype_bytes


def memory_feasibility(
    model: LLMModel,
    hw: Hardware,
    context_tokens: int,
) -> FeasibilityCheck:
    weights_bytes = model.gguf_bytes
    kv_bytes = kv_cache_bytes_per_token(model) * context_tokens
    total_required = weights_bytes + kv_bytes + RUNTIME_OVERHEAD_BYTES
    available_bytes = hw.mem_capacity_gb * 1_000_000_000
    headroom = available_bytes - total_required

    if headroom < 0:
        verdict = "wont_fit"
    elif headroom < available_bytes * 0.15:
        verdict = "tight"
    else:
        verdict = "fits"

    return FeasibilityCheck(
        verdict=verdict,
        required_gb=total_required / 1e9,
        available_gb=available_bytes / 1e9,
        headroom_gb=headroom / 1e9,
        breakdown={
            "weights_gb": weights_bytes / 1e9,
            "kv_cache_gb": kv_bytes / 1e9,
            "overhead_gb": RUNTIME_OVERHEAD_BYTES / 1e9,
        },
    )
```

### The projection function

```python
# ratchet/projection/llm.py

def project_llm(
    model: LLMModel,
    hw: Hardware,
    workload_id: str,
    *,
    prompt_tokens: int = 500,
    decode_tokens: int = 200,
    host_ms: float = 0.0,
    compiler_quality: float = 1.0,
    npu_share: Optional[float] = None,
) -> ProjectionResult:
    """Project LLM performance for (model, hw, workload).

    Returns Projected | WontFit | DtypeMismatch.

    Resolution cascade (first hit wins):
      1. Per-cell measured (hw.measured_llm[model.key][workload_id])
      2. Tier-level anchor (hw.measured_decode_overrides[model.key])
      3. Same-family anchor (sibling tier in tier_family, BW-scaled)
      4. Cross-class first-principles (two-floor MAX(BW, compute))

    compiler_quality: Trust factor for vendor compiler stack.
    npu_share: Bandwidth fraction; scales BW-bound paths only.

    Does NOT apply anchor-secrets overlays — those run separately."""

    # Step 0a: Memory feasibility
    context = prompt_tokens + decode_tokens
    feas = memory_feasibility(model, hw, context)
    if feas.verdict == "wont_fit":
        return WontFit(...)

    # Step 0b: Dtype compatibility
    cap_level = hw_supports_dtype(hw, model.compute_dtype)
    if cap_level is CapabilityLevel.UNSUPPORTED:
        return DtypeMismatch(...)

    # Step 1: Per-cell measured
    cell = _resolve_per_cell_measurement(hw, model, workload_id)
    if cell is not None:
        return _build_projected_from_cell(cell, hw, model, ...)

    # Step 2: Tier-level anchor
    anchor = _resolve_tier_level_anchor(hw, model)
    if anchor is not None:
        return _build_projected_from_tier_anchor(anchor, hw, model, ...)

    # Step 3: Same-family anchor
    family_anchor = _find_same_family_anchor(hw, model)
    if family_anchor is not None:
        return _build_projected_from_family_anchor(family_anchor, hw, model, ...)

    # Step 4: Cross-class fallback
    return _build_projected_cross_class(hw, model, ...)
```

### The cross-class two-floor MAX math

```python
def _build_projected_cross_class(hw, model, workload_id, *, prompt_tokens,
                                  decode_tokens, compiler_quality,
                                  npu_share_actual, host_ms) -> Projected:
    """First-principles projection: max(BW_floor, compute_floor) + overhead."""

    active_params_gb = (model.active_params * model.bytes_per_param) / 1e9
    gops_per_token = 2 * (model.active_params / 1e9)

    # LLM compute floor uses RAW peak (calibrated against
    # llm_prefill_util_factor, not effective_tops — per ADR 015)
    peak_tops_llm = max(hw_peak_tops_for_dtype(hw, model.compute_dtype), 1e-9)

    # Decode-side per-token floors
    decode_bw_realized = (
        hw.effective_bandwidth_gbs
        * hw.llm_decode_bw_realization
        * npu_share_actual
    )
    bw_floor_ms_decode = (active_params_gb / max(decode_bw_realized, 1e-9)) * 1000.0
    compute_floor_ms_decode = gops_per_token / (peak_tops_llm * hw.llm_prefill_util_factor)

    per_token_ms = max(bw_floor_ms_decode, compute_floor_ms_decode)
    decode_tok_s = (1000.0 / max(per_token_ms, 1e-6)) * compiler_quality

    regime = "bw_bound" if bw_floor_ms_decode >= compute_floor_ms_decode else "compute_bound"

    # Prefill side: per-batch BW, per-token compute
    bw_floor_ms_prefill = (active_params_gb / hw.effective_bandwidth_gbs) * 1000.0
    compute_floor_ms_prefill = (
        gops_per_token * prompt_tokens / (peak_tops_llm * hw.llm_prefill_util_factor)
    )
    ttft_ms = max(bw_floor_ms_prefill, compute_floor_ms_prefill) + hw.compute_overhead_ms
    prefill_tok_s = prompt_tokens / max(ttft_ms / 1000.0, 1e-6) * compiler_quality

    decode_ceiling = 1000.0 / max(bw_floor_ms_decode, 1e-6)
    decode_s = decode_tokens / max(decode_tok_s, 1e-6)
    prefill_s = prompt_tokens / max(prefill_tok_s, 1e-6)
    host_s = (host_ms or hw.compute_overhead_ms) / 1000.0
    total_s = host_s + prefill_s + decode_s

    return Projected(
        decode_tok_s=round(decode_tok_s, 2),
        prefill_tok_s=round(prefill_tok_s, 2),
        ttft_s=round(ttft_ms / 1000.0, 4),
        decode_tokens=decode_tokens,
        prompt_tokens=prompt_tokens,
        decode_s=round(decode_s, 3),
        prefill_s=round(prefill_s, 3),
        total_s=round(total_s, 3),
        host_ms=round(host_s * 1000, 2),
        source="cross_class",
        regime=regime,
        hw_name=hw.name,
        model_key=model.key,
        workload_id=workload_id,
        decode_ceiling_tok_s=round(decode_ceiling, 2),
    )
```

Key invariants:
- Compute floor uses *raw peak TOPS*, not effective_tops()
- `npu_share_actual` applies to BW-bound paths only
- `compiler_quality` multiplies final rates, not floor calculations
- Regime classification is diagnostic

### Workload-pattern multipliers as optional overlay

```python
# ratchet/projection/workload_pattern.py

@dataclass(frozen=True)
class WorkloadPatternMultipliers:
    decode_p50_mult: float = 1.0
    decode_p95_mult: float = 1.0
    ttft_p50_mult: float = 1.0


def apply_workload_pattern(
    result: Projected,
    multipliers: WorkloadPatternMultipliers,
) -> Projected:
    """Apply workload-pattern multipliers to a Projected result.
    Returns new Projected with base_decode_pre_multiplier preserved."""
    new_decode = result.decode_tok_s * multipliers.decode_p50_mult
    new_ttft = result.ttft_s * multipliers.ttft_p50_mult
    return dataclasses.replace(
        result,
        decode_tok_s=round(new_decode, 2),
        ttft_s=round(new_ttft, 4),
        base_decode_pre_multiplier=result.decode_tok_s,
    )
```

PAI sizer never calls this. Keyhole-sizer calls it after each projection with its measured workload-category multipliers.

### Surface usage patterns

```python
# Pattern 1: Simple projection (PAI sizer style)
from ratchet import project_llm, Projected, WontFit, DtypeMismatch

result = project_llm(
    model=qwen_30b_a3b,
    hw=NPU_MID,
    workload_id="rag_qa",
    prompt_tokens=4800,
    decode_tokens=400,
)

match result:
    case Projected(decode_tok_s=t, source=s):
        st.metric("Decode rate", f"{t} tok/s ({s})")
    case WontFit(required_gb=r, available_gb=a):
        st.error(f"Won't fit: {r:.1f} GB needed, {a:.1f} GB available")
    case DtypeMismatch(retargeting_hint=h):
        st.warning(f"Dtype mismatch. {h}")


# Pattern 2: Projection + workload pattern (keyhole-sizer style)
result = project_llm(model=skippy_7b_v4, hw=NPU_HIGH, workload_id="plain_chat")

if isinstance(result, Projected):
    rag_multipliers = WorkloadPatternMultipliers(
        decode_p50_mult=0.073, ttft_p50_mult=130.5,
    )
    rag_result = apply_workload_pattern(result, rag_multipliers)
```

---

## Section 9 — The anchor-secrets system

### Overview

The anchor-secrets system is how private silicon measurements override projection results without those measurements ever entering the source tree, git history, or chat transcripts. Phase 0 found that PAI sizer and keyhole-sizer have byte-identical `npu_anchors.py` modules — the loader is already canonical, it just needs a single home.

### What's private vs public

**Public** (lives in source, git, chat):
- Tier specifications (NPU Mid has 134.4 GB/s peak BW, 200 eTOPS INT8)
- Calibration constants (`compute_efficiency=0.65`)
- Vendor-published bake-off claims
- The structure of the anchor-secrets schema itself

**Private** (lives only in gitignored `.streamlit/secrets.toml`):
- Specific measured decode tok/s values for NPU silicon
- Specific TTFT seconds for NPU silicon
- Power measurements not published by vendor
- Source attribution identifying whose silicon was measured

### The canonical schema

```toml
# .streamlit/secrets.toml (gitignored, KEY-not-VALUE discipline)

[npu_llm_anchors.mid_int8.qwen3_30b_a3b_moe]
tokps = <REDACTED>
ms_per_inference = <REDACTED>
peak_bw_gbps = <REDACTED>
bw_share_frac = <REDACTED>
bw_efficiency_frac = <REDACTED>
source = "measured"
measured_date = "2026-04-XX"

# ... 8 more LLM cells (3 models × 3 tier_dtype combinations)

[cnn_anchors.mid_int8.resnet50_w4]
ms_per_inference = <REDACTED>
fps = <REDACTED>
peak_bw_gbps = <REDACTED>
bw_share_frac = <REDACTED>
bw_efficiency_frac = <REDACTED>
source = "measured"
measured_date = "2026-04-XX"

# ... 5 more CNN cells
```

### The loader

```python
# ratchet/anchors/loader.py

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


@dataclass(frozen=True)
class LLMAnchor:
    tokps: float
    ms_per_inference: float
    peak_bw_gbps: float
    bw_share_frac: float = 0.75
    bw_efficiency_frac: float = 0.70
    source: str = "measured"
    measured_date: str = ""


@dataclass(frozen=True)
class CNNAnchor:
    ms_per_inference: float
    fps: float
    peak_bw_gbps: float
    bw_share_frac: float = 0.75
    bw_efficiency_frac: float = 0.70
    source: str = "measured"
    measured_date: str = ""


def load_llm_anchor(tier_dtype: str, model_key: str) -> Optional[LLMAnchor]:
    """Load an LLM anchor. Returns None when Streamlit unavailable,
    secrets absent, cell unpopulated, or value malformed."""
    if not _HAS_STREAMLIT:
        return None
    try:
        cells = st.secrets["npu_llm_anchors"][tier_dtype]
        cell = cells[model_key]
        return LLMAnchor(
            tokps=float(cell["tokps"]),
            ms_per_inference=float(cell["ms_per_inference"]),
            peak_bw_gbps=float(cell["peak_bw_gbps"]),
            bw_share_frac=float(_try_get(cell, "bw_share_frac", 0.75)),
            bw_efficiency_frac=float(_try_get(cell, "bw_efficiency_frac", 0.70)),
            source=str(_try_get(cell, "source", "measured")),
            measured_date=str(_try_get(cell, "measured_date", "")),
        )
    except (KeyError, ValueError, TypeError):
        return None


def load_cnn_anchor(tier_dtype: str, model_key: str) -> Optional[CNNAnchor]:
    """Load a CNN anchor. Same fallback semantics as load_llm_anchor."""
    # Parallel implementation
```

The loader is byte-identical to what currently lives in PAI sizer and keyhole-sizer.

### The spec-routing map

```python
# ratchet/anchors/spec_routing.py

def hw_to_anchor_tier_dtype(hw: Hardware, model_compute_dtype: str) -> Optional[str]:
    """Map (Hardware, model dtype) to anchor-secrets tier_dtype key.

    Canonical spec defines three tier_dtype cells:
      - 'mid_int8': NPU Mid running INT8 workload
      - 'high_int8': NPU High running INT8 workload
      - 'high_fp': NPU High running FP workload (BF16 or FP8)

    Uses hw.tier_lookup_name (memory-upgrade clones route to stock).
    Returns None for bw_projected tiers (memory-upgrade clones don't
    have their own anchors)."""

    if hw.bw_projected:
        return None

    name = hw.tier_lookup_name
    dt = model_compute_dtype.lower()

    if name == "NPU Mid":
        return "mid_int8" if dt in ("int8", "q4_km") else None
    if name == "NPU High":
        if dt == "int8":
            return "high_int8"
        if dt in ("bf16", "fp16", "fp8", "q4_km"):
            return "high_fp"
        return None
    return None
```

### The parameterized overlay

```python
# ratchet/anchors/overlay.py

def overlay_llm_anchor(
    result: Projected,
    hw: Hardware,
    catalog_to_spec_key: Callable[[str], Optional[str]],
    *,
    workload_multiplier: float = 1.0,
) -> Projected:
    """Hot-swap a Projected result with a private silicon anchor when available.

    Returns the same result unchanged when:
      - No anchor secrets are loaded
      - Hardware tier has no spec cell
      - Memory-upgrade clone (bw_projected=True)
      - Model has no spec-key mapping

    catalog_to_spec_key: Surface-supplied function mapping catalog model_keys
        to spec model_keys. Different surfaces have different catalog naming;
        the canonical spec uses snake_case.

    workload_multiplier: Optional workload-pattern multiplier. Keyhole-sizer
        passes non-1.0 values; PAI sizer passes 1.0."""

    tier_dtype = hw_to_anchor_tier_dtype(hw, _dtype_for_model(result.model_key))
    if tier_dtype is None:
        return result

    spec_key = catalog_to_spec_key(result.model_key)
    if spec_key is None:
        return result

    anchor = load_llm_anchor(tier_dtype, spec_key)
    if anchor is None:
        return result

    new_decode = anchor.tokps * workload_multiplier
    new_ttft = (anchor.ms_per_inference / 1000.0) if anchor.ms_per_inference else result.ttft_s

    return dataclasses.replace(
        result,
        decode_tok_s=round(new_decode, 2),
        ttft_s=round(new_ttft, 4),
        source="measured_silicon_anchor",
        silicon_anchor_meta={
            "anchor_source": anchor.source,
            "measured_date": anchor.measured_date,
            "tier_dtype": tier_dtype,
            "spec_model_key": spec_key,
        },
    )
```

The function takes `catalog_to_spec_key` as a callable so surfaces inject their own naming translation.

### Surface usage patterns

```python
# Pattern 1: PAI sizer
_PAI_ANCHOR_MAP = {
    "qwen3-30b-a3b-q4-moe": "qwen3_30b_a3b_moe",
    "qwen2.5-32b-q4-dense": "qwen25_32b_dense",
    "qwen2.5-7b-q4-dense": "qwen25_7b_dense",
}
_pai_translate = lambda k: _PAI_ANCHOR_MAP.get(k)

result = project_llm(model=qwen_30b, hw=NPU_MID, workload_id="rag_qa")
if isinstance(result, Projected):
    result = overlay_llm_anchor(result, NPU_MID, _pai_translate)


# Pattern 2: Keyhole-sizer with workload multiplier
_KEYHOLE_ANCHOR_MAP = {
    "skippy_7b_v4": "qwen25_7b_dense",
    "qwen3_30b_a3b_moe": "qwen3_30b_a3b_moe",
}
_keyhole_translate = lambda k: _KEYHOLE_ANCHOR_MAP.get(k, k)

result = project_llm(model=skippy_7b_v4, hw=NPU_HIGH, workload_id="rag_qa")
if isinstance(result, Projected):
    rag_mult = compute_workload_multiplier("rag_qa", reference="plain_chat")
    result = overlay_llm_anchor(
        result, NPU_HIGH, _keyhole_translate,
        workload_multiplier=rag_mult,
    )
```

### The discipline rules carry forward

- Anchor values never appear in source code
- Anchor values never appear in chat transcripts or commit messages
- The schema is public; the values are credentials
- `.streamlit/secrets.toml` is gitignored
- Field names in chat are public; field values are not

Ratchet's loader respects this — schema is documented, loader is open source, but the loader never has access to values except at runtime from local secrets.

---

## Section 10 — The LLM catalog

### Overview

The LLM catalog defines the schema for every LLM the engine knows about. Each surface ships its own catalog *content* (PAI has 20 entries, keyhole has 17), but ratchet owns the schema. PAI carries the richer transformer-architecture-aware schema; keyhole carries the cleaner dataclass-frozen encoding. The new ratchet is the synthesis: PAI's field coverage in keyhole's encoding style.

### The LLMModel dataclass

```python
# ratchet/catalog/model.py

ComputeDtype = Literal["fp16", "bf16", "fp8", "int8"]
QuantScheme = Literal["Q4_K_M", "Q5_K_M", "Q8_0", "FP16", "FP8", "INT8_W8A8"]


@dataclass(frozen=True)
class LLMModel:
    """Schema for a single LLM in a sizer catalog. Frozen for safety."""

    # ─── Core identity ───
    key: str
    """Unique catalog key. Snake_case canonical (matches anchor-secrets spec):
    'skippy_7b_v4', 'qwen3_30b_a3b_moe'. PAI sizer migrates from hyphenated
    keys during retrofit."""

    family: str
    """Model family for filtering and grouping."""

    base: str
    """Base architecture identifier. Distinct from family for fine-tunes."""

    # ─── Size and quantization ───
    total_params_b: float
    """Total parameter count in billions."""

    active_params_b: float
    """Active parameters per forward pass. For dense: == total_params_b."""

    quant_scheme: QuantScheme
    """Storage quantization."""

    bytes_per_param: float
    """Storage bytes per parameter. From quant lookup table."""

    gguf_size_gb: float
    """On-disk model size in GB."""

    size_gb_inflight: float
    """In-memory size during inference, in GB."""

    compute_dtype: ComputeDtype
    """Compute dtype for matmul."""

    # ─── Transformer architecture (required) ───
    num_layers: int
    hidden_dim: int
    num_attention_heads: int
    num_kv_heads: int
    """For GQA models: < num_attention_heads. For non-GQA: equals it."""

    # ─── MoE-specific (optional) ───
    is_moe: bool = False
    num_experts: Optional[int] = None
    experts_per_token: Optional[int] = None

    # ─── Context and vocabulary (optional) ───
    ctx_len_trained: Optional[int] = None
    vocab_size: Optional[int] = None

    # ─── Training and provenance (optional) ───
    training: Optional[str] = None
    training_recipe: Optional[str] = None
    fine_tune_version: Optional[str] = None

    # ─── Reference-only flag ───
    perf_reference_only: bool = False

    # ─── Measurement routing ───
    measurement_alias: Optional[str] = None
    """When set: measurements come from the named other model.
    Skippy 7B v4 → 'qwen25_7b_dense'."""

    # ─── Quality metrics (optional) ───
    pass_rate: Optional[float] = None
    pass_n_passes: Optional[int] = None
    pass_n_total: Optional[int] = None
    category_deltas: Optional[dict] = None

    # ─── Display fields ───
    accuracy_bullet: Optional[str] = None
    description: Optional[str] = None

    # ─── Workload-pattern behavior ───
    llm_invariant_decode: bool = True
    """True (default): decode tok/s intrinsic to (model, hardware).
    False: surfaces can apply workload-pattern multipliers."""


def lookup_model(key: str, catalog: dict[str, LLMModel]) -> Optional[LLMModel]:
    return catalog.get(key)


def resolve_measurement_key(model: LLMModel) -> str:
    """Return key for measurement lookup. Returns measurement_alias when set."""
    return model.measurement_alias if model.measurement_alias else model.key
```

### Quant byte tables

```python
# ratchet/catalog/constants.py

BYTES_PER_PARAM: dict[str, float] = {
    "Q4_K_M": 0.57,
    "Q5_K_M": 0.68,
    "Q8_0":   1.04,
    "FP16":   2.00,
    "BF16":   2.00,
    "FP8":    1.00,
    "INT8_W8A8": 1.00,
}
"""PAI and keyhole agreed on Q4_K_M=0.57, diverged on Q5_K_M and Q8_0.
Reconciled here at keyhole's values pending empirical investigation."""


GGUF_SIZE_GB: dict[str, float] = {
    "Q4_K_M": 18.6,
    "Q5_K_M": 21.7,
    "Q8_0":   32.5,
    "FP16":   60.0,
    "BF16":   60.0,
}


ACTIVE_PARAMS: int = 3_000_000_000
"""Reference active parameter count: Qwen3 30B-A3B MoE."""
```

### The canonical reference catalog

```python
# ratchet/catalog/reference.py

QWEN3_30B_A3B_MOE_Q4 = LLMModel(
    key="qwen3_30b_a3b_moe",
    family="Qwen3-MoE",
    base="Qwen3-30B-A3B",
    total_params_b=30.0,
    active_params_b=3.0,
    is_moe=True,
    num_experts=128,
    experts_per_token=8,
    quant_scheme="Q4_K_M",
    bytes_per_param=BYTES_PER_PARAM["Q4_K_M"],
    gguf_size_gb=18.6,
    size_gb_inflight=18.6,
    num_layers=48,
    hidden_dim=2048,
    num_attention_heads=32,
    num_kv_heads=4,
    compute_dtype="fp16",
    ctx_len_trained=131072,
    vocab_size=151936,
    description="Qwen3 30B-A3B MoE with 8/128 expert routing, Q4_K_M GGUF",
)

QWEN25_32B_DENSE_Q4 = LLMModel(
    key="qwen25_32b_dense",
    family="Qwen2.5",
    base="Qwen2.5-32B",
    total_params_b=32.5,
    active_params_b=32.5,
    is_moe=False,
    quant_scheme="Q4_K_M",
    bytes_per_param=BYTES_PER_PARAM["Q4_K_M"],
    gguf_size_gb=19.9,
    size_gb_inflight=19.9,
    num_layers=64,
    hidden_dim=5120,
    num_attention_heads=40,
    num_kv_heads=8,
    compute_dtype="fp16",
    ctx_len_trained=131072,
    vocab_size=152064,
    description="Qwen2.5 32B dense, Q4_K_M GGUF",
)

QWEN25_7B_DENSE_Q4 = LLMModel(
    key="qwen25_7b_dense",
    family="Qwen2.5",
    base="Qwen2.5-7B",
    total_params_b=7.6,
    active_params_b=7.6,
    is_moe=False,
    quant_scheme="Q4_K_M",
    bytes_per_param=BYTES_PER_PARAM["Q4_K_M"],
    gguf_size_gb=4.4,
    size_gb_inflight=4.4,
    num_layers=28,
    hidden_dim=3584,
    num_attention_heads=28,
    num_kv_heads=4,
    compute_dtype="fp16",
    ctx_len_trained=131072,
    vocab_size=152064,
    description="Qwen2.5 7B dense, Q4_K_M GGUF",
)


REFERENCE_MODELS: dict[str, LLMModel] = {
    m.key: m for m in (
        QWEN3_30B_A3B_MOE_Q4,
        QWEN25_32B_DENSE_Q4,
        QWEN25_7B_DENSE_Q4,
        # ... Q5 and Q8 variants follow
    )
}
```

### Surface catalog patterns

```python
# personal-ai-assistant-sizer/sizer/catalog.py
from ratchet import LLMModel, BYTES_PER_PARAM

SKIPPY_7B_V4 = LLMModel(
    key="skippy_7b_v4",
    family="Skippy",
    base="Qwen2.5-7B",
    training="skippy_v4_fft",
    training_recipe="skippy/recipes/v4_fft.md",
    fine_tune_version="v4",
    measurement_alias="qwen25_7b_dense",  # measurements come from stock base
    total_params_b=7.6,
    active_params_b=7.6,
    quant_scheme="Q4_K_M",
    bytes_per_param=BYTES_PER_PARAM["Q4_K_M"],
    gguf_size_gb=4.4,
    size_gb_inflight=4.4,
    num_layers=28,
    hidden_dim=3584,
    num_attention_heads=28,
    num_kv_heads=4,
    compute_dtype="fp16",
    ctx_len_trained=131072,
    pass_rate=0.99,
    pass_n_passes=99,
    pass_n_total=100,
    description="Skippy 7B v4 — Qwen2.5-7B fully fine-tuned on Skippy golden set",
    accuracy_bullet="99/100 pass rate on Skippy v4 golden set",
)

MODELS: dict[str, LLMModel] = {m.key: m for m in (SKIPPY_7B_V4, ...)}
```

### Role taxonomy: still surface-side

```python
# Surface helper (not in ratchet)
def model_role(model: LLMModel) -> str:
    """Derive PROD / FT / BASE / PERF role from model fields."""
    if model.key == PRODUCTION_REFERENCE_KEY:
        return "PROD"
    if model.training is not None:
        return "FT"
    if model.perf_reference_only:
        return "PERF"
    return "BASE"
```

### Quantization disagreement resolution

`BYTES_PER_PARAM` currently disagrees: PAI (Q5_K_M=0.70, Q8_0=1.06) vs keyhole (Q5_K_M=0.68, Q8_0=1.04). Ratchet ships with keyhole's values pending investigation. Both surfaces converge automatically when importing from ratchet. If empirical comparison reveals one set is correct, update canonical values and bump ratchet version.

---

## Section 11 — ADRs

The new ratchet design produces 9 new ADRs covering the architectural decisions in sections 2-10. Combined with the 6 ADRs carried forward from v0.1.0, ratchet v0.2.0 ships with 15 ADRs total.

### Carried from ratchet v0.1.0

**ADR 001 — Two-source workload model.** Workloads carry both slider-based and measurement-based override paths. Projection cascades through overrides before first-principles.

**ADR 002 — Process-isolation for partition fidelity.** Subsystem workload modeling uses separate process boundaries for measurement isolation.

**ADR 003 — BF16 as competitive differentiator.** BF16 support is the key differentiator between Mid and High class NPU silicon.

**ADR 004 — LLM as memory-bound workload.** LLM decode is BW-bound (not compute-bound) for MoE active-parameter streaming.

**ADR 005 — NPU efficiency factor calibration.** The compute_efficiency constant per tier scales with silicon maturity.

**ADR 006 — Trajectory-driven test harness.** The what-if runner uses trajectory traces rather than synthetic workload generators.

### New in ratchet v0.2.0

**ADR 007 — Canonical tier registry.** TIERS holds the union of all tiers any surface uses. Surfaces compose visible ladders; they don't define new tiers (except via `make_custom_tier()`). Tier silicon facts are invariant across the ecosystem.

**ADR 008 — 4-level capability taxonomy.** Per-tier dtype capability has four levels because consumer Blackwell INT8 (sm80 IMMA via binary compat) is materially different from native tensor-core support. CUDA_CORE level is reserved for future silicon.

**ADR 009 — Measurement attachment unification.** Hardware carries three flat dicts plus one nested dict. Three flat fields chosen over single discriminated dict for type clarity, code-path independence, and incremental field-addition.

**ADR 010 — Stock identity tracking on memory-upgrade clones.** Memory-upgrade variants capture stock_name + stock_mem_bandwidth_gbs so silicon-intrinsic lookups resolve correctly. Adopted from PAI's pattern; replaces keyhole's magic-constant check.

**ADR 011 — Anchor-secrets as post-projection overlay.** Private silicon measurements never enter the Hardware object. Loaded from secrets.toml at runtime, applied via overlay_llm_anchor on Projected. Public/private separation is structural.

**ADR 012 — Workload-pattern multipliers as optional overlay.** PAI's workload-invariant model and keyhole's workload-pattern model reconciled by treating patterns as optional post-projection scaling. Surfaces opt in per their UX.

**ADR 013 — Custom tier factory with silicon-class defaults.** make_custom_tier requires silicon_class parameter to select appropriate calibration. Defaults exposed for canonical silicon classes; "unknown" class uses Hardware defaults with low-confidence warning.

**ADR 014 — Calibration provenance.** Each tier carries a CalibrationSource encoding (method, reference, confidence). Surfaces render appropriate UI banners; ratchet doesn't impose color or wording.

**ADR 015 — DTYPE attribute dispatch.** DTYPE_ATTR_MAP routes dtype strings to Hardware field names for raw-peak-TOPS lookup. LLM compute floor uses raw peak (not effective_tops). Vision compute floor uses effective_tops. Dual calibration convention is critical.

---

## Section 12 — Migration plan

The migration happens in six phases. Each phase is a separate Claude Code session in a single repo, with a tagged release at the end. Sequencing matters: each retrofit's findings can inform ratchet improvements before the next retrofit starts.

### Phase 1 — Build new ratchet (this design)

**Repo:** ratchet
**Duration:** 1-2 days
**Output:** ratchet v0.2.0 tagged and pushed

Phase 1 builds new ratchet per sections 2-10 of this document. Claude Code executes the design without retrofitting any surface yet. At the end of phase 1, ratchet v0.2.0 exists with:
- Full Hardware dataclass (24 fields)
- Eight canonical tiers in the registry
- Capability taxonomy + dtype dispatch
- Projection API with 4-path resolution
- Anchor-secrets loader + parameterized overlay
- LLM catalog schema + reference catalog + constants
- Calibration provenance
- ~200 unit tests across all submodules
- 15 ADRs landed in docs/decisions/

Phase 1 does NOT change any surface. PAI sizer, keyhole-sizer, keyhole, and Skippy remain at their stable v1.0.0 / v5.10.0 tags throughout.

### Phase 2 — PAI sizer retrofit

**Repo:** personal-ai-assistant-sizer
**Duration:** 2-3 days
**Output:** PAI sizer v1.1.0 retrofitted onto ratchet v0.2.0

Tasks:
1. Pin ratchet>=0.2.0 in pyproject.toml
2. Delete sizer/npu_anchors.py (now in ratchet.anchors)
3. Delete sizer/npu_model.py's Hardware definition, TIERS dict, and helpers
4. Delete sizer/precision.py's tier_precision_capability (now in ratchet.precision)
5. Migrate catalog keys from hyphenated to snake_case canonical
6. Update _ANCHOR_LLM_MODEL_KEY_MAP to use canonical keys
7. Update sizer/measured.py to attach measurements to ratchet's RTX_5090_REFERENCE
8. Replace ad-hoc project_llm calls with ratchet's project_llm + overlay_llm_anchor
9. Run full test suite; verify dashboard outputs match v1.0.0 to within float tolerance
10. Tag v1.1.0, push

**Critical discipline during phase 2:** when ratchet turns out to be missing something PAI needs, the answer is *not* to fork. Pause retrofit, switch to ratchet session, add the missing piece to ratchet v0.2.1, push, return to PAI session.

### Phase 3 — keyhole-sizer retrofit

**Repo:** keyhole-sizer
**Duration:** 2-3 days
**Output:** keyhole-sizer v1.1.0 retrofitted onto ratchet

Similar to phase 2. Additional tasks:
- Migrate legacy measured_llm_q4_decode_tok_s / measured_llm_ttft_1k_sec to ratchet's measured_decode_overrides / measured_prefill_overrides
- Migrate from inline capability_levels to importing canonical capability tables
- Migrate from custom Hardware dispatch in effective_tops() to ratchet's external helpers
- Wire workload-pattern multipliers through workload_pattern_overlay
- Retire magic-constant memory-upgrade check in favor of bw_projected
- Tag v1.1.0, push

### Phase 4 — keyhole backend retrofit

**Repo:** keyhole
**Duration:** 3-4 days
**Output:** keyhole v1.1.0 retrofitted onto ratchet

Most keyhole backend code (bake-off harnesses, ncu pipelines, FastAPI service) is keyhole-domain and stays put. The retrofit replaces local Hardware/tier definitions with ratchet imports, local capability classifiers with ratchet.precision, and ad-hoc projection helpers with ratchet's projection API. Tag v1.1.0, push.

### Phase 5 — Skippy framework retrofit

**Repo:** personal-ai-framework
**Duration:** 1-2 days
**Output:** Skippy v5.11.0 retrofitted onto ratchet

Skippy's overlap with ratchet is small: cross-surface canonical specs and sizer-bundle building. Retrofit is small: imports from ratchet, no major code restructuring. Tag v5.11.0, push.

### Phase 6 — Drone surface on mature engine

**Repo:** (new — name TBD; possibly drone-sizer)
**Duration:** TBD
**Output:** drone-sizer v0.1.0 built on ratchet v0.2.x

Phase 6 is the first new surface built on the mature engine. By this point, ratchet has been through five real consumers and the API is solid. The drone surface picks up where nightjar's drone work left off and may motivate ratchet v0.3.0 (heterogeneous-architecture awareness, MPU bus modeling, real-time core support).

### Per-phase tagging strategy

```
ratchet v0.2.0    ← phase 1
ratchet v0.2.x    ← incremental during phases 2-5 if gaps found

PAI sizer v1.1.0      ← phase 2
keyhole-sizer v1.1.0  ← phase 3
keyhole v1.1.0        ← phase 4
Skippy v5.11.0        ← phase 5

drone-sizer v0.1.0    ← phase 6 (new repo)
ratchet v0.3.0        ← potentially during/after phase 6
```

The rollback tags from the stabilization sweep (`pre-ratchet-consolidation`) remain unchanged on each repo, providing the global rollback point.

---

## Section 13 — Phase 1 execution plan

### What Claude Code does in the ratchet repo

Phase 1 starts with a fresh Claude Code session in `~/Documents/GitHub/ratchet`. Order of operations:

**Step 1: Read and acknowledge the design.** Claude Code reads sections 2-10 plus the v0.1.0 ADRs. No code changes yet. Output: summary confirming understanding and flagging ambiguities for resolution.

**Step 2: Module skeleton.** Create the directory structure from section 2 (`tiers/`, `precision/`, `projection/`, `anchors/`, `catalog/`, `calibration/` submodules). Empty `__init__.py` files. Existing `engine/`, `whatif/`, `probes/`, `schemas/` unchanged.

**Step 3: Hardware dataclass.** Implement `ratchet/tiers/hardware.py` per section 3.

**Step 4: Capability taxonomy.** Implement `ratchet/precision/capability.py`, `ratchet/precision/dtype_map.py`, and `ratchet/precision/deployment_path.py` per section 5. Includes the three canonical capability tables.

**Step 5: Tier registry.** Implement `ratchet/tiers/registry.py` with all eight canonical tiers per section 4. Implement `ratchet/tiers/memory_overlay.py`.

**Step 6: Calibration.** Implement `ratchet/calibration/source.py` and `ratchet/calibration/silicon_class.py` per section 6. Implement `ratchet/tiers/custom.py`.

**Step 7: Catalog schema.** Implement `ratchet/catalog/model.py`, `ratchet/catalog/constants.py`, and `ratchet/catalog/reference.py` per section 10.

**Step 8: Projection.** Implement `ratchet/projection/result.py`, `ratchet/projection/feasibility.py`, `ratchet/projection/llm.py` per section 8. Largest single piece of code. Implement `ratchet/projection/workload_pattern.py`.

**Step 9: Anchor-secrets.** Implement `ratchet/anchors/loader.py`, `ratchet/anchors/schemas.py`, `ratchet/anchors/overlay.py`, `ratchet/anchors/spec_routing.py` per section 9.

**Step 10: Public API re-exports.** Update `ratchet/__init__.py` to expose the public API surface from section 2.

**Step 11: ADRs.** Write the 9 new ADR files in `docs/decisions/` per section 11.

**Step 12: Test suite.** Build out unit tests covering each submodule. Target: ~200 unit tests + 15-20 integration tests. v0.1.0 test suite (76 tests) continues to pass.

**Step 13: README and CLAUDE.md.** Update top-level README to reflect v0.2.0's scope. Update CLAUDE.md with consolidated architecture overview.

**Step 14: Tag and push.** Bump pyproject.toml to v0.2.0. Commit, tag `v0.2.0`, push. Confirm GitHub release visible.

### Test strategy

Test categories with rough counts:

- **Hardware** (~25 tests): dataclass field defaults, properties, methods
- **Tiers registry** (~15 tests): TIERS integrity, memory_overlay scaling math
- **Custom tier factory** (~10 tests): silicon_class defaults, calibration_source rules
- **Capability** (~20 tests): CapabilityLevel truthy semantics, hw_supports_dtype, deployment_path classification
- **Catalog** (~15 tests): LLMModel construction, lookup_model, resolve_measurement_key
- **Projection** (~40 tests): each cascade path, memory_feasibility verdicts, dtype mismatch, workload_pattern
- **Anchors** (~30 tests): loader fallback behavior, overlay function with various surface translations
- **Calibration** (~10 tests): CalibrationSource validation, silicon_class default tables

**Plus integration tests:**
- End-to-end LLM projection through all 4 cascade paths (~6 tests)
- Anchor overlay end-to-end with mock secrets (~4 tests)
- Memory-upgrade clone projection (~4 tests)
- WorkloadRecord roundtrip through probe writer (~2 tests)

Target total: ~170-220 tests.

### Success criteria for phase 1

Phase 1 is done when:

1. All tests pass (`pytest tests/ -v`)
2. `pip install -e .` succeeds in a clean venv
3. `python -c "from ratchet import *"` succeeds with all 40+ public names importable
4. The 15 ADRs are written and committed
5. pyproject.toml says version="0.2.0"
6. The tag `v0.2.0` exists on origin/main and is visible on GitHub
7. CLAUDE.md reflects the v0.2.0 architecture

**NOT in the success criteria:** PAI sizer or keyhole-sizer working against the new ratchet. That's phases 2-3.

### When phase 1 is uncertain

If during phase 1 a design decision in this document turns out to be wrong or incomplete:

1. **Stop the Claude Code session.**
2. **Surface the issue in a chat session with the reviewer.**
3. **Decide together whether to amend the design document or work around it.**
4. **If amending, update the design document with a clear marker (`AMENDMENT: 2026-MM-DD —`) and rationale.**
5. **Resume the Claude Code session with the amended design.**

Phase 1 is not the place to invent architecture. The design document is the contract. If the contract is wrong, fix the contract before fixing the code.

### Rollback strategy

If phase 1 produces a broken ratchet v0.2.0, recovery is:

```bash
cd ~/Documents/GitHub/ratchet
git checkout v0.1.0
# Investigate what went wrong in the v0.2.0 branch
# Once understood, decide whether to fix and re-tag or revise design
```

No surface is affected by a broken v0.2.0 because no surface depends on it yet. Four ecosystem surfaces remain at stable tags. Nightjar's v0.3.0-dev continues to use ratchet v0.1.0.

This is the rollback safety the stabilization sweep enabled.

---

*End of RATCHET v0.2.0 design specification.*
