"""Unit tests for the LLM catalog schema, constants, and reference catalog."""
import dataclasses

import pytest

from ratchet.catalog.alias import resolve_measurement_key as alias_resolve
from ratchet.catalog.constants import ACTIVE_PARAMS, BYTES_PER_PARAM, GGUF_SIZE_GB
from ratchet.catalog.model import (
    LLMModel,
    lookup_model,
    resolve_measurement_key,
)
from ratchet.catalog.reference import (
    QWEN3_30B_A3B_MOE_Q4,
    REFERENCE_MODELS,
)


def _mk(**ov) -> LLMModel:
    base = dict(
        key="m", family="F", base="B",
        total_params_b=7.0, active_params_b=7.0,
        quant_scheme="Q4_K_M", bytes_per_param=0.57,
        gguf_size_gb=4.4, size_gb_inflight=4.4, compute_dtype="fp16",
        num_layers=28, hidden_dim=3584, num_attention_heads=28, num_kv_heads=4,
    )
    base.update(ov)
    return LLMModel(**base)


class TestConstants:
    def test_q4km_bytes(self):
        assert BYTES_PER_PARAM["Q4_K_M"] == 0.57

    def test_keyhole_values_for_disputed(self):
        assert BYTES_PER_PARAM["Q5_K_M"] == 0.68
        assert BYTES_PER_PARAM["Q8_0"] == 1.04

    def test_gguf_table(self):
        assert GGUF_SIZE_GB["Q4_K_M"] == 18.6

    def test_active_params_reference(self):
        assert ACTIVE_PARAMS == 3_000_000_000


class TestLLMModel:
    def test_frozen(self):
        m = _mk()
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.key = "other"  # type: ignore[misc]

    def test_optional_defaults(self):
        m = _mk()
        assert m.is_moe is False
        assert m.measurement_alias is None
        assert m.llm_invariant_decode is True
        assert m.perf_reference_only is False


class TestLookup:
    def test_lookup_hit(self):
        cat = {"m": _mk()}
        assert lookup_model("m", cat).key == "m"

    def test_lookup_miss(self):
        assert lookup_model("nope", {}) is None


class TestMeasurementAlias:
    def test_returns_key_when_no_alias(self):
        m = _mk(key="qwen25_7b_dense")
        assert resolve_measurement_key(m) == "qwen25_7b_dense"

    def test_returns_alias_when_set(self):
        m = _mk(key="skippy_7b_v4", measurement_alias="qwen25_7b_dense")
        assert resolve_measurement_key(m) == "qwen25_7b_dense"

    def test_alias_module_reexport(self):
        m = _mk(key="skippy_7b_v4", measurement_alias="qwen25_7b_dense")
        assert alias_resolve(m) == "qwen25_7b_dense"


class TestReferenceCatalog:
    def test_three_entries(self):
        assert len(REFERENCE_MODELS) == 3

    def test_moe_entry(self):
        m = QWEN3_30B_A3B_MOE_Q4
        assert m.is_moe is True
        assert m.active_params_b == 3.0
        assert m.num_kv_heads == 4
        assert m.quant_scheme == "Q4_K_M"

    def test_keyed_by_model_key(self):
        for k, m in REFERENCE_MODELS.items():
            assert m.key == k
