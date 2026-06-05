"""The Hardware dataclass — the central engine object.

Represents an NPU or GPU tier with its silicon facts, calibration constants,
measurement attachments, and clone-tracking metadata. Every projection call
consumes a Hardware instance; every anchor overlay reads from one; every UI
tier-render derives from one.
"""
from dataclasses import dataclass
from typing import Optional

from ratchet.calibration.source import CalibrationSource
from ratchet.precision.capability import CapabilityInfo


@dataclass
class Hardware:
    """Generic compute-and-bandwidth spec for any NPU or GPU.

    A Hardware instance is mutable in principle (plain @dataclass, not frozen)
    but is treated as immutable in practice. The only mutation paths are:
      1. ratchet.tiers.memory_overlay.hw_with_memory() — creates a variant via
         dataclasses.replace() (functional copy, not in-place mutation).
      2. Measurement attachment at import time, on the reference tier only
         (typically RTX_5090_REFERENCE) — populated by sizer-specific
         measurement loaders that read sizer_bundle.json or equivalent.

    Surfaces SHOULD NOT mutate Hardware instances post-construction.
    """

    # ─── Silicon-fact fields (required, immutable in practice) ───
    name: str
    """Display name. Rewritten by hw_with_memory() to add the variant suffix
    (e.g., 'NPU Mid (LPDDR6 @ 14 GT/s)'). For canonical identity lookups, use
    tier_lookup_name property instead."""

    peak_tops_bf16: float
    """Raw peak BF16/FP16 TOPS. 0.0 means tier has no FP16 path (e.g., NPU Mid
    is INT8-only). FP16 and BF16 conflate to this field per common-silicon
    convention."""

    peak_tops_int8: float
    """Raw peak INT8 TOPS."""

    peak_tops_fp8: float
    """Raw peak FP8 TOPS. 0.0 on tiers without FP8 path."""

    mem_bandwidth_gbs: float
    """Raw peak DRAM bandwidth in GB/s. Effective BW = this ×
    bandwidth_efficiency × npu_share. Rewritten by hw_with_memory() via the
    formula: new_bw = mem_bus_width_bits × mem_data_rate_gtps / 8."""

    mem_capacity_gb: float
    """DRAM capacity in GB."""

    mem_bus_width_bits: int
    """Memory bus width (32 / 64 / 128 / 512). Used by hw_with_memory() to
    recompute BW under data-rate swaps."""

    mem_type: str
    """One of: 'LPDDR4', 'LPDDR5', 'LPDDR5X', 'LPDDR5T', 'LPDDR6', 'LPDDR7',
    'GDDR6', 'GDDR6X', 'GDDR7', 'HBM3'."""

    mem_data_rate_gtps: float
    """DRAM I/O data rate in GT/s. Used in the BW recomputation formula on
    memory-upgrade clones. NOT used for anchor-secrets routing (that routes
    through bw_projected + tier_lookup_name instead, per ADR 010)."""

    peak_tops_fp4: float = 0.0
    """Raw peak FP4 (NVFP4/MXFP4) TOPS. 0.0 on tiers without a native FP4
    tensor-core path (every edge NPU today, and pre-Blackwell GPUs). Defaulted
    (added v0.2.5) so existing tier constructions don't break — logically a
    silicon peak_tops_* field alongside bf16/int8/fp8. Native on Blackwell sm_120
    (RTX 5090) / sm_100 (B200): FP4 is a compute format (memory + compute), unlike
    weight-only INT4. Routed via DTYPE_ATTR_MAP['nvfp4']."""

    # ─── Calibration constants (per-tier defaults, mutable per-tier) ───
    compute_efficiency: float = 0.65
    """Real-workload effective TOPS multiplier. effective_tops() returns
    raw_peak × compute_efficiency. Scales with silicon maturity: 0.60 NPU Low /
    0.65 NPU Mid / 0.70 NPU High / 0.70 RTX 5090. USED FOR VISION compute floor
    only — LLM compute floor uses raw peak with llm_prefill_util_factor
    instead, per ADR 015."""

    bandwidth_efficiency: float = 0.70
    """Real-workload BW multiplier. effective_bandwidth_gbs property returns
    mem_bandwidth_gbs × bandwidth_efficiency. 0.70 NPU tiers, 0.85 RTX 5090
    (dedicated VRAM has lower contention overhead)."""

    tdp_watts: float = 0.0
    """Thermal envelope; informational only. Not consumed by projection.
    Surfaces may display this in tier cards."""

    tier_family: Optional[str] = None
    """Memory-class taxonomy for same-class anchor scaling. Tiers in the same
    family share enough silicon characteristics that anchors measured on one
    tier can be BW-scaled within the family (🟡 same_class_anchor in the
    projection cascade). Values like 'LP5X-8.4-128b', 'Neutron-32-LP5',
    'GDDR7-28'. None means 'no family lookup available; fall through to
    cross-class (🔴) in projection."""

    compute_util_factor: float = 0.45
    """VISION compute-floor utilization factor. Per-tier calibration: 0.19
    Neutron-class / 0.45 NPU Mid / 0.50 NPU High / 0.85 RTX 5090. Multiplied
    against effective_tops() in vision projection compute floor. Vision-only;
    LLM uses llm_prefill_util_factor instead. Default 0.45 corresponds to NPU
    Mid (middle of the calibrated range)."""

    llm_prefill_util_factor: float = 0.10
    """LLM PREFILL compute-floor utilization factor. Multiplied against raw
    peak TOPS (NOT effective_tops) in LLM projection compute floor. Calibrated
    against raw peak per ADR 015 to avoid double-discounting with
    compute_efficiency. Typical value 0.10 on NPU silicon; reflects that LLM
    prefill achieves 5-15% of vendor peak due to small per-layer matmuls, MoE
    expert routing, and KV cache writes."""

    llm_decode_bw_realization: float = 1.0
    """LLM decode BW realization fraction. Defaults to 1.0 (pure BW ceiling —
    active_params_GB streaming at full effective BW). Held at default across
    tiers because realization is model-class-specific; using a measured MoE
    value would over-pessimize dense projections. Per-model calibration is
    captured in the anchor itself."""

    compute_overhead_ms: float = 1.0
    """Per-inference kernel-launch + sync overhead. NPU default 1.0; RTX 5090
    0.3 (less overhead on dedicated PCIe device)."""

    npu_share_default: float = 0.75
    """Default fraction of memory bandwidth available to the NPU on an SoC with
    a shared memory bus. 0.75 for NPU SoC tiers (shared with CPU, GPU, ISP),
    1.0 for RTX 5090 (dedicated VRAM, no contention). Surfaces may expose a
    user override via UI slider."""

    # ─── Per-dtype capability taxonomy ───
    capability_levels: Optional[dict[str, CapabilityInfo]] = None
    """Per-dtype 4-level capability dict. Keys are precision strings ('int8',
    'fp8', 'bf16/fp16', 'q4_km'). Values are CapabilityInfo with level
    (tensor_native/tensor_compat/cuda_core/unsupported) and reason string.

    When None, hw_supports_dtype() falls back to the peak-TOPS heuristic
    (peak_tops_<dtype> > 0 → supported). Canonical tiers MUST set this; custom
    tiers MAY leave it None and accept the heuristic fallback."""

    # ─── Calibration provenance ───
    calibration_source: Optional[CalibrationSource] = None
    """Provenance metadata for the calibration constants on this tier. Encodes
    (method, reference, confidence) so surfaces can render appropriate banners.
    Canonical tiers carry measured calibration; custom tiers carry 'default'
    with low confidence."""

    # ─── Stock-identity tracking for memory-upgrade clones ───
    bw_projected: bool = False
    """True iff this Hardware was synthesized via hw_with_memory(). Surfaces
    use this to mark BW-scaled projections as '(BW-proj)' in the UI.
    Anchor-secrets overlay short-circuits when True (memory-upgrade variants
    don't have their own measured anchors)."""

    stock_mem_bandwidth_gbs: Optional[float] = None
    """Snapshot of stock peak BW captured by hw_with_memory(). Lets projection
    hold prefill at stock under memory-upgrade overlays (prefill is
    compute-bound, not BW-bound, so memory upgrades don't affect it). None on
    stock tiers."""

    stock_name: Optional[str] = None
    """Snapshot of stock tier name captured by hw_with_memory(). The
    tier_lookup_name property returns stock_name when set, else name.
    Silicon-intrinsic lookups (precision capability, deployment path) key off
    the stock identity regardless of memory variant."""

    # ─── Measurement attachment (three flat fields, per ADR 009) ───
    measured_decode_overrides: Optional[dict[str, float]] = None
    """LLM decode tok/s overrides keyed by model_key. Per-tier-and-model
    measurements that override projection. Currently populated on NPU_MID with
    the Skippy MoE Q4_K_M anchor (37.85 tok/s). Public source-tree data; NOT
    anchor-secrets."""

    measured_prefill_overrides: Optional[dict[str, float]] = None
    """LLM prefill tok/s overrides keyed by model_key. Held at stock under
    memory-upgrade clones (prefill is compute-bound)."""

    measured_vision_overrides: Optional[dict[str, dict[str, dict[str, float]]]] = None
    """Vision measurements keyed by pipeline_key, then resolution. Inner dict
    shape: {'ms_per_inference': float, 'fps': float, ...}. Populated on tiers
    with vision bake-off data (NPU_LOW_LP5X, IMX95_MEASURED,
    RTX_5090_REFERENCE). Surface uses this to bypass projection entirely for
    matching (pipeline, resolution) pairs."""

    measured_llm: Optional[dict[str, dict[str, dict[str, float]]]] = None
    """Per-cell LLM measurements with full workload-scoped granularity. Shape:
    {model_key: {workload_id: {'decode_tok_s', 'prefill_tok_s', 'ttft_s',
    'host_ms'}}}.

    Populated only on the reference tier (RTX_5090_REFERENCE typically) by
    surface-specific bundle loaders. Read by projection's measured-cell path
    (🟢 measured) as the highest-priority resolution.

    NOT populated on NPU tiers in this version — those use
    measured_decode_overrides / measured_prefill_overrides for now."""

    # ─── Properties ───
    @property
    def effective_bandwidth_gbs(self) -> float:
        """Raw peak BW × bandwidth_efficiency. Does NOT include npu_share (that
        composes downstream in projection)."""
        return self.mem_bandwidth_gbs * self.bandwidth_efficiency

    @property
    def tier_lookup_name(self) -> str:
        """Canonical identity for silicon-intrinsic lookups. Returns stock_name
        if this is a memory-upgrade clone, else name. Used by anchor-secrets
        spec-cell routing, precision-capability lookup, and deployment-path
        lookup. Silicon caps don't change with memory swaps, so these lookups
        must key off the stock identity."""
        return self.stock_name if self.stock_name is not None else self.name

    # ─── Methods ───
    def effective_tops(self, dtype: str) -> float:
        """Effective TOPS for a dtype = raw peak × compute_efficiency. FP16
        routes to peak_tops_bf16 per dtype-conflation convention. Unknown dtype
        falls back to bf16.

        USED FOR VISION compute floor. LLM compute floor uses raw peak via
        hw_peak_tops_for_dtype() because llm_prefill_util_factor was calibrated
        against raw peak (ADR 015)."""
        from ratchet.precision.dtype_map import DTYPE_ATTR_MAP

        attr = DTYPE_ATTR_MAP.get(dtype.lower(), "peak_tops_bf16")
        return getattr(self, attr) * self.compute_efficiency

    def get_measured_llm_cell(
        self, model_key: str, workload_id: str
    ) -> Optional[dict]:
        """Resolve a per-cell LLM measurement via direct lookup only.

        Returns None if no measurement is available for (model_key,
        workload_id). Catalog-aware alias resolution does NOT happen here — it
        happens at the projection call site, where both the catalog and the
        Hardware are visible. Callers should fall through to tier-level
        overrides (measured_decode_overrides), then to projection."""
        if not self.measured_llm:
            return None
        return self.measured_llm.get(model_key, {}).get(workload_id)
