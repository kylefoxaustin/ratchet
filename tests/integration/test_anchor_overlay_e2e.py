"""End-to-end: projection cascade → anchor-secrets overlay with mock secrets.

v0.2.1 (Amendment 3): real 3-arg loader, rich anchor schema, decode_mult.
"""
from types import SimpleNamespace

import pytest

from ratchet import NPU_MID, Projected, overlay_llm_anchor, project_llm
from ratchet.anchors import loader as loader_mod
from ratchet.catalog.reference import QWEN3_30B_A3B_MOE_Q4

_IDENTITY = lambda k: k


def _install_secrets(monkeypatch, tokps, prefill_tokps=2849.0):
    secrets = {"npu_llm_anchors": {"mid_int8": {"qwen3_30b_a3b_moe": {
        "tokps": tokps, "prefill_tokps": prefill_tokps, "mem_gb": 18.6,
        "seqlen": 4096, "source": "measured", "measured_date": "2026-04-15",
        "peak_bw_gbps": 134.4, "bw_share_frac": 0.75,
        "bw_efficiency_frac": 0.70}}}}
    monkeypatch.setattr(loader_mod, "st", SimpleNamespace(secrets=secrets))


class TestOverlayEndToEnd:
    def test_anchor_overrides_projection(self, monkeypatch):
        _install_secrets(monkeypatch, tokps=44.0)
        base = project_llm(QWEN3_30B_A3B_MOE_Q4, NPU_MID, "rag_qa")
        assert isinstance(base, Projected)
        out = overlay_llm_anchor(base, NPU_MID, QWEN3_30B_A3B_MOE_Q4, _IDENTITY)
        assert out.source == "measured_silicon_anchor"
        assert out.decode_tok_s == pytest.approx(44.0)
        assert out.silicon_anchor_meta["spec_model_key"] == "qwen3_30b_a3b_moe"
        assert out.silicon_anchor_meta["spec_tier_precision"] == "mid_int8"

    def test_no_secret_passes_through(self, monkeypatch):
        monkeypatch.setattr(loader_mod, "st",
                            SimpleNamespace(secrets={"npu_llm_anchors": {}}))
        base = project_llm(QWEN3_30B_A3B_MOE_Q4, NPU_MID, "rag_qa")
        out = overlay_llm_anchor(base, NPU_MID, QWEN3_30B_A3B_MOE_Q4, _IDENTITY)
        assert out is base
        assert out.source == "measured_anchor"

    def test_keyhole_workload_scaling(self, monkeypatch):
        _install_secrets(monkeypatch, tokps=44.0, prefill_tokps=2849.0)
        base = project_llm(QWEN3_30B_A3B_MOE_Q4, NPU_MID, "rag_qa")
        out = overlay_llm_anchor(base, NPU_MID, QWEN3_30B_A3B_MOE_Q4, _IDENTITY,
                                 decode_mult=0.073, ttft_mult=130.5)
        assert out.decode_tok_s == pytest.approx(44.0 * 0.073, abs=0.01)
        assert out.ttft_s == pytest.approx((1024.0 / 2849.0) * 130.5, abs=1e-2)
