"""End-to-end: drive the four projection cascade paths through the public API."""
import dataclasses

import pytest

from ratchet import (
    NPU_HIGH,
    NPU_LOW_LP4,
    NPU_MID,
    RTX_5090_REFERENCE,
    DtypeMismatch,
    LLMModel,
    Projected,
    WontFit,
    project_llm,
)
from ratchet.catalog.reference import QWEN3_30B_A3B_MOE_Q4, QWEN25_7B_DENSE_Q4


class TestCascadePaths:
    def test_measured_anchor_path(self):
        r = project_llm(QWEN3_30B_A3B_MOE_Q4, NPU_MID, "rag_qa")
        assert isinstance(r, Projected) and r.source == "measured_anchor"
        assert r.decode_tok_s == pytest.approx(37.85, abs=0.01)

    def test_same_class_anchor_path(self):
        # NPU High borrows NPU Mid's family anchor, BW-scaled (same eff BW).
        r = project_llm(QWEN3_30B_A3B_MOE_Q4, NPU_HIGH, "rag_qa")
        assert isinstance(r, Projected) and r.source == "same_class_anchor"

    def test_per_cell_measured_path(self):
        ref = dataclasses.replace(RTX_5090_REFERENCE, measured_llm={
            "qwen25_7b_dense": {"plain_chat": {
                "decode_tok_s": 150.0, "prefill_tok_s": 4000.0,
                "ttft_s": 0.04, "host_ms": 3.0}}})
        r = project_llm(QWEN25_7B_DENSE_Q4, ref, "plain_chat")
        assert isinstance(r, Projected) and r.source == "measured"
        assert r.decode_tok_s == pytest.approx(150.0)

    def test_cross_class_path(self):
        novel = dataclasses.replace(QWEN25_7B_DENSE_Q4, key="novel_7b")
        r = project_llm(novel, RTX_5090_REFERENCE, "w")
        assert isinstance(r, Projected) and r.source == "cross_class"

    def test_wont_fit_path(self):
        r = project_llm(QWEN3_30B_A3B_MOE_Q4, NPU_LOW_LP4, "w")  # 8 GB tier
        assert isinstance(r, WontFit)

    def test_dtype_mismatch_path(self):
        fp8 = LLMModel(
            key="fp8_model", family="F", base="B",
            total_params_b=7.0, active_params_b=7.0,
            quant_scheme="FP8", bytes_per_param=1.0,
            gguf_size_gb=4.0, size_gb_inflight=4.0, compute_dtype="fp8",
            num_layers=28, hidden_dim=3584, num_attention_heads=28, num_kv_heads=4)
        r = project_llm(fp8, NPU_MID, "w")  # Neutron INT8-only, no FP8 path
        assert isinstance(r, DtypeMismatch)


class TestPriorityOrdering:
    def test_per_cell_beats_tier_anchor(self):
        # NPU Mid has a tier anchor for the MoE; a per-cell measurement on the
        # same tier+model+workload must win.
        mid = dataclasses.replace(NPU_MID, measured_llm={
            "qwen3_30b_a3b_moe": {"rag_qa": {
                "decode_tok_s": 99.0, "prefill_tok_s": 5000.0,
                "ttft_s": 0.1, "host_ms": 1.0}}})
        r = project_llm(QWEN3_30B_A3B_MOE_Q4, mid, "rag_qa")
        assert r.source == "measured"
        assert r.decode_tok_s == pytest.approx(99.0)
