"""The LLMModel schema.

Ratchet owns the schema; each surface ships its own catalog *content* (PAI ~20
entries, keyhole ~17). PAI's transformer-architecture field coverage in
keyhole's frozen-dataclass encoding style.
"""
from dataclasses import dataclass
from typing import Literal, Optional

# "nvfp4" is the canonical FP4 compute dtype (MXFP4 conflates to it via
# DTYPE_ATTR_MAP). FP4 is a native Blackwell sm_120/sm_100 compute format; its
# realized throughput win is runtime-conditional (ADR 016).
ComputeDtype = Literal["fp16", "bf16", "fp8", "int8", "nvfp4"]
QuantScheme = Literal[
    "Q4_K_M", "Q5_K_M", "Q8_0", "FP16", "FP8", "INT8_W8A8", "NVFP4", "MXFP4",
    "INT4_AWQ", "INT4_GPTQ",
]


@dataclass(frozen=True)
class LLMModel:
    """Schema for a single LLM in a sizer catalog. Frozen for safety."""

    # ─── Core identity ───
    key: str
    """Unique catalog key. Snake_case canonical (matches anchor-secrets spec):
    'skippy_7b_v4', 'qwen3_30b_a3b_moe'."""

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
    """Storage quantization. Governs executability key (AMENDMENT 1)."""

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
