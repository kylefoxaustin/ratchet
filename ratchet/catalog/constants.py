"""Quant byte tables and catalog constants.

BYTES_PER_PARAM and GGUF_SIZE_GB are shared lookups; surfaces import these so
they converge automatically. PAI and keyhole diverged on Q5_K_M / Q8_0; ratchet
ships keyhole's values pending empirical investigation (ADR 014).
"""

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
