"""Unit tests for the projection API — feasibility, cascade, result types."""
import pytest

from ratchet.catalog.model import LLMModel
from ratchet.catalog.reference import QWEN3_30B_A3B_MOE_Q4
from ratchet.projection.feasibility import (
    kv_cache_bytes_per_token,
    memory_feasibility,
)
from ratchet.projection.llm import project_llm
from ratchet.projection.result import DtypeMismatch, Projected, WontFit
from ratchet.projection.workload_pattern import (
    WorkloadPatternMultipliers,
    apply_workload_pattern,
)
from ratchet.tiers.registry import (
    NPU_HIGH,
    NPU_LOW_LP4,
    NPU_MID,
    RTX_5090_REFERENCE,
)


def _model(**ov) -> LLMModel:
    base = dict(
        key="m", family="F", base="B",
        total_params_b=7.0, active_params_b=7.0,
        quant_scheme="Q4_K_M", bytes_per_param=0.57,
        gguf_size_gb=4.4, size_gb_inflight=4.4, compute_dtype="fp16",
        num_layers=28, hidden_dim=3584, num_attention_heads=28, num_kv_heads=4,
    )
    base.update(ov)
    return LLMModel(**base)


class TestFeasibility:
    def test_kv_cache_uses_gqa_ratio(self):
        m = _model(num_layers=28, hidden_dim=3584, num_attention_heads=28,
                   num_kv_heads=4)
        # 28 * 2 * 3584 * (4/28) * 2 bytes
        expected = 28 * 2 * 3584 * (4 / 28) * 2
        assert kv_cache_bytes_per_token(m) == pytest.approx(expected)

    def test_fits_small_model_big_tier(self):
        check = memory_feasibility(_model(gguf_size_gb=4.4), NPU_HIGH, 1000)
        assert check.verdict == "fits"

    def test_wont_fit_big_model_tiny_tier(self):
        big = _model(gguf_size_gb=40.0)
        check = memory_feasibility(big, NPU_LOW_LP4, 1000)  # 8 GB tier
        assert check.verdict == "wont_fit"
        assert check.headroom_gb < 0

    def test_breakdown_keys(self):
        check = memory_feasibility(_model(), NPU_HIGH, 500)
        assert set(check.breakdown) == {"weights_gb", "kv_cache_gb", "overhead_gb"}


class TestStep0WontFit:
    def test_returns_wontfit(self):
        big = _model(gguf_size_gb=40.0)
        result = project_llm(big, NPU_LOW_LP4, "w")
        assert isinstance(result, WontFit)
        assert result.required_gb > result.available_gb


class TestStep0bAmendment1:
    def test_q4km_moe_not_rejected_on_neutron_mid(self):
        # THE amendment-1 regression: qwen3_30b_a3b_moe is compute_dtype fp16,
        # NPU Mid is INT8-only. The OLD gate returned DtypeMismatch here and the
        # measured anchor was unreachable. Now it must resolve to the anchor.
        result = project_llm(QWEN3_30B_A3B_MOE_Q4, NPU_MID, "rag_qa")
        assert isinstance(result, Projected)
        assert result.source == "measured_anchor"
        assert result.decode_tok_s == pytest.approx(37.85, abs=0.01)

    def test_genuinely_unsupported_still_mismatches(self):
        # An FP8 model on INT8-only Neutron silicon has no path → DtypeMismatch.
        fp8 = _model(quant_scheme="FP8", compute_dtype="fp8", gguf_size_gb=4.0)
        result = project_llm(fp8, NPU_MID, "w")
        assert isinstance(result, DtypeMismatch)
        assert result.required_dtype == "FP8"


class TestStep1PerCellMeasured:
    def test_measured_cell_wins(self):
        hw = _clone_with_measured_llm()
        result = project_llm(_model(key="qwen25_7b_dense"), hw, "plain_chat")
        assert isinstance(result, Projected)
        assert result.source == "measured"
        assert result.decode_tok_s == pytest.approx(120.0)

    def test_alias_resolves_to_base_measurement(self):
        hw = _clone_with_measured_llm()
        skippy = _model(key="skippy_7b_v4", measurement_alias="qwen25_7b_dense")
        result = project_llm(skippy, hw, "plain_chat")
        assert isinstance(result, Projected)
        assert result.source == "measured"
        assert result.decode_tok_s == pytest.approx(120.0)


class TestStep2TierAnchor:
    def test_tier_anchor_decode(self):
        result = project_llm(QWEN3_30B_A3B_MOE_Q4, NPU_MID, "rag_qa")
        assert isinstance(result, Projected)
        assert result.source == "measured_anchor"
        assert result.decode_tok_s == pytest.approx(37.85, abs=0.01)


class TestStep3SameFamilyAnchor:
    def test_high_borrows_mid_anchor_bw_scaled(self):
        # NPU High shares tier_family 'LP5X-8.4-128b' with NPU Mid (which has
        # the MoE anchor). Same effective BW → scaled decode == 37.85.
        result = project_llm(QWEN3_30B_A3B_MOE_Q4, NPU_HIGH, "rag_qa")
        assert isinstance(result, Projected)
        assert result.source == "same_class_anchor"
        assert result.decode_tok_s == pytest.approx(37.85, abs=0.01)


class TestStep4CrossClass:
    def test_cross_class_fallback(self):
        # A model with no anchor anywhere, on the 5090 → first-principles.
        m = _model(key="novel_model", gguf_size_gb=4.0)
        result = project_llm(m, RTX_5090_REFERENCE, "w")
        assert isinstance(result, Projected)
        assert result.source == "cross_class"
        assert result.decode_tok_s > 0
        assert result.regime in ("bw_bound", "compute_bound")
        assert result.decode_ceiling_tok_s is not None

    def test_compiler_quality_scales_rates(self):
        m = _model(key="novel_model", gguf_size_gb=4.0)
        base = project_llm(m, RTX_5090_REFERENCE, "w")
        scaled = project_llm(m, RTX_5090_REFERENCE, "w", compiler_quality=0.5)
        assert scaled.decode_tok_s == pytest.approx(base.decode_tok_s * 0.5, rel=1e-3)


class TestWorkloadPattern:
    def test_multiplier_applied(self):
        m = _model(key="novel_model", gguf_size_gb=4.0)
        result = project_llm(m, RTX_5090_REFERENCE, "w")
        mult = WorkloadPatternMultipliers(decode_p50_mult=0.5, ttft_p50_mult=2.0)
        out = apply_workload_pattern(result, mult)
        assert out.decode_tok_s == pytest.approx(result.decode_tok_s * 0.5, abs=0.01)
        assert out.base_decode_pre_multiplier == result.decode_tok_s


def _clone_with_measured_llm():
    import dataclasses
    return dataclasses.replace(
        RTX_5090_REFERENCE,
        measured_llm={
            "qwen25_7b_dense": {
                "plain_chat": {
                    "decode_tok_s": 120.0, "prefill_tok_s": 3000.0,
                    "ttft_s": 0.05, "host_ms": 5.0,
                }
            }
        },
    )
