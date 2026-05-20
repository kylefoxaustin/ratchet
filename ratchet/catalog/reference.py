"""The canonical reference catalog — a small set of well-characterized models.

Surfaces ship their own catalogs; these reference entries exist for ratchet's
own tests and as a starting point. The MODELS catalog content is per-surface
and NOT exported from ratchet's top-level API.
"""
from ratchet.catalog.constants import BYTES_PER_PARAM
from ratchet.catalog.model import LLMModel

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
    )
}
