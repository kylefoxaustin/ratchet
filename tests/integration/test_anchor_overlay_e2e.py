"""End-to-end: projection cascade → anchor-secrets overlay with mock secrets."""
from types import SimpleNamespace

import pytest

from ratchet import NPU_MID, Projected, overlay_llm_anchor, project_llm
from ratchet.anchors import loader as loader_mod
from ratchet.catalog.reference import QWEN3_30B_A3B_MOE_Q4

_IDENTITY = lambda k: k


def _install_secrets(monkeypatch, tokps, ms):
    secrets = {"npu_llm_anchors": {"mid_int8": {"qwen3_30b_a3b_moe": {
        "tokps": tokps, "ms_per_inference": ms, "peak_bw_gbps": 134.4,
        "source": "measured", "measured_date": "2026-04-15"}}}}
    monkeypatch.setattr(loader_mod, "_HAS_STREAMLIT", True)
    monkeypatch.setattr(loader_mod, "st", SimpleNamespace(secrets=secrets))


class TestOverlayEndToEnd:
    def test_anchor_overrides_projection(self, monkeypatch):
        _install_secrets(monkeypatch, tokps=44.0, ms=18.0)
        base = project_llm(QWEN3_30B_A3B_MOE_Q4, NPU_MID, "rag_qa")
        assert isinstance(base, Projected)
        out = overlay_llm_anchor(base, NPU_MID, QWEN3_30B_A3B_MOE_Q4, _IDENTITY)
        assert out.source == "measured_silicon_anchor"
        assert out.decode_tok_s == pytest.approx(44.0)
        assert out.silicon_anchor_meta["spec_model_key"] == "qwen3_30b_a3b_moe"

    def test_no_secret_passes_through(self, monkeypatch):
        # Empty secrets table → overlay returns the projection unchanged.
        monkeypatch.setattr(loader_mod, "_HAS_STREAMLIT", True)
        monkeypatch.setattr(loader_mod, "st",
                            SimpleNamespace(secrets={"npu_llm_anchors": {}}))
        base = project_llm(QWEN3_30B_A3B_MOE_Q4, NPU_MID, "rag_qa")
        out = overlay_llm_anchor(base, NPU_MID, QWEN3_30B_A3B_MOE_Q4, _IDENTITY)
        assert out is base
        assert out.source == "measured_anchor"

    def test_workload_multiplier_through_overlay(self, monkeypatch):
        _install_secrets(monkeypatch, tokps=44.0, ms=18.0)
        base = project_llm(QWEN3_30B_A3B_MOE_Q4, NPU_MID, "rag_qa")
        out = overlay_llm_anchor(base, NPU_MID, QWEN3_30B_A3B_MOE_Q4, _IDENTITY,
                                 workload_multiplier=0.073)
        assert out.decode_tok_s == pytest.approx(44.0 * 0.073, abs=0.01)
