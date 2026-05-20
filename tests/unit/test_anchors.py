"""Unit tests for the anchor-secrets loader, spec routing, and overlay.

Reshaped for v0.2.1 (Amendment 3): real 3-arg loader + rich anchor schema,
(tier, precision) routing, decode_mult/ttft_mult overlay.
"""
from types import SimpleNamespace

import pytest

from ratchet.anchors import loader as loader_mod
from ratchet.anchors import overlay as overlay_mod
from ratchet.anchors.loader import LLMAnchor, load_cnn_anchor, load_llm_anchor
from ratchet.anchors.overlay import overlay_llm_anchor
from ratchet.anchors.spec_routing import hw_to_anchor_tier_precision
from ratchet.catalog.reference import QWEN3_30B_A3B_MOE_Q4
from ratchet.projection.llm import project_llm
from ratchet.projection.result import Projected
from ratchet.tiers.memory_overlay import hw_with_memory
from ratchet.tiers.registry import NPU_HIGH, NPU_MID, RTX_5090_REFERENCE


# ─── Loader: rich schema + 3-arg signature + fallback ───────────────

class TestLLMAnchorSchema:
    def test_rich_fields_and_methods(self):
        a = LLMAnchor(
            tokps=40.0, prefill_tokps=2000.0, mem_gb=18.6, seqlen=4096,
            source="measured", measured_date="2026-04-15", peak_bw_gbps=134.4,
            bw_share_frac=0.75, bw_efficiency_frac=0.70, notes="x")
        assert a.badge == "🟢"
        assert a.achieved_bw_gbps() == pytest.approx(134.4 * 0.75 * 0.70)
        # bytes_per_token = achieved_bw * 1e9 / tokps
        assert a.bytes_per_token() == pytest.approx(
            134.4 * 0.75 * 0.70 * 1e9 / 40.0)

    def test_share_override(self):
        a = LLMAnchor(tokps=40.0, prefill_tokps=0.0, mem_gb=0.0, seqlen=0,
                      source="measured", measured_date="", peak_bw_gbps=100.0,
                      bw_share_frac=0.75, bw_efficiency_frac=0.70)
        assert a.achieved_bw_gbps(share_override=0.5) == pytest.approx(35.0)

    def test_badge_unknown_source_blank(self):
        a = LLMAnchor(tokps=1.0, prefill_tokps=0.0, mem_gb=0.0, seqlen=0,
                      source="???", measured_date="", peak_bw_gbps=1.0,
                      bw_share_frac=0.75, bw_efficiency_frac=0.70)
        assert a.badge == ""


class TestLoaderFallback:
    def test_none_without_streamlit(self, monkeypatch):
        monkeypatch.setattr(loader_mod, "st", None)
        assert load_llm_anchor("mid", "int8", "qwen3_30b_a3b_moe") is None
        assert load_cnn_anchor("mid", "int8", "resnet50_w4") is None

    def test_none_when_cell_absent(self, monkeypatch):
        monkeypatch.setattr(loader_mod, "st",
                            SimpleNamespace(secrets={"npu_llm_anchors": {}}))
        assert load_llm_anchor("mid", "int8", "qwen3_30b_a3b_moe") is None

    def test_loads_valid_cell_three_arg(self, monkeypatch):
        secrets = {"npu_llm_anchors": {"mid_int8": {"qwen3_30b_a3b_moe": {
            "tokps": 40.0, "prefill_tokps": 2849.0, "mem_gb": 18.6,
            "seqlen": 4096, "source": "measured", "measured_date": "2026-04-15",
            "peak_bw_gbps": 134.4, "bw_share_frac": 0.75,
            "bw_efficiency_frac": 0.70}}}}
        monkeypatch.setattr(loader_mod, "st", SimpleNamespace(secrets=secrets))
        a = load_llm_anchor("mid", "int8", "qwen3_30b_a3b_moe")
        assert a.tokps == 40.0
        assert a.prefill_tokps == 2849.0
        assert a.seqlen == 4096

    def test_none_when_tokps_nonpositive(self, monkeypatch):
        secrets = {"npu_llm_anchors": {"mid_int8": {"m": {
            "tokps": 0.0, "peak_bw_gbps": 1.0}}}}
        monkeypatch.setattr(loader_mod, "st", SimpleNamespace(secrets=secrets))
        assert load_llm_anchor("mid", "int8", "m") is None

    def test_cnn_derives_fps_from_ms(self, monkeypatch):
        secrets = {"cnn_anchors": {"mid_int8": {"resnet50_w4": {
            "ms_per_inference": 10.0, "source": "measured",
            "peak_bw_gbps": 100.0}}}}
        monkeypatch.setattr(loader_mod, "st", SimpleNamespace(secrets=secrets))
        c = load_cnn_anchor("mid", "int8", "resnet50_w4")
        assert c.fps == pytest.approx(100.0)  # 1000/10


# ─── Spec routing: (tier, precision) tuples ─────────────────────────

class TestSpecRouting:
    def test_mid_q4km_and_int8(self):
        assert hw_to_anchor_tier_precision(NPU_MID, "q4_km") == ("mid", "int8")
        assert hw_to_anchor_tier_precision(NPU_MID, "int8") == ("mid", "int8")

    def test_mid_fp_none(self):
        assert hw_to_anchor_tier_precision(NPU_MID, "bf16/fp16") is None

    def test_high_int8(self):
        assert hw_to_anchor_tier_precision(NPU_HIGH, "int8") == ("high", "int8")

    def test_high_fp_and_q4km(self):
        assert hw_to_anchor_tier_precision(NPU_HIGH, "bf16/fp16") == ("high", "fp")
        assert hw_to_anchor_tier_precision(NPU_HIGH, "q4_km") == ("high", "fp")
        assert hw_to_anchor_tier_precision(NPU_HIGH, "fp8") == ("high", "fp")

    def test_reference_tier_none(self):
        assert hw_to_anchor_tier_precision(RTX_5090_REFERENCE, "fp8") is None

    def test_memory_clone_none(self):
        clone = hw_with_memory(NPU_MID, "LPDDR6", 14.0)
        assert hw_to_anchor_tier_precision(clone, "q4_km") is None


# ─── Overlay: decode_mult / ttft_mult ───────────────────────────────

_IDENTITY = lambda k: k


def _projected(model, hw) -> Projected:
    r = project_llm(model, hw, "rag_qa")
    assert isinstance(r, Projected)
    return r


def _anchor(tokps=42.0, prefill_tokps=2849.0):
    return LLMAnchor(tokps=tokps, prefill_tokps=prefill_tokps, mem_gb=18.6,
                     seqlen=4096, source="measured", measured_date="2026-04-15",
                     peak_bw_gbps=134.4, bw_share_frac=0.75,
                     bw_efficiency_frac=0.70)


class TestOverlay:
    def test_no_secret_unchanged(self, monkeypatch):
        monkeypatch.setattr(overlay_mod, "load_llm_anchor", lambda *a, **k: None)
        r = _projected(QWEN3_30B_A3B_MOE_Q4, NPU_MID)
        assert overlay_llm_anchor(r, NPU_MID, QWEN3_30B_A3B_MOE_Q4, _IDENTITY) is r

    def test_non_measured_source_unchanged(self, monkeypatch):
        proj = _anchor()
        proj = LLMAnchor(**{**proj.__dict__, "source": "vendor_spec"})
        monkeypatch.setattr(overlay_mod, "load_llm_anchor", lambda *a, **k: proj)
        r = _projected(QWEN3_30B_A3B_MOE_Q4, NPU_MID)
        assert overlay_llm_anchor(r, NPU_MID, QWEN3_30B_A3B_MOE_Q4, _IDENTITY) is r

    def test_pai_invariant_overlay(self, monkeypatch):
        # decode_mult=ttft_mult=1.0 → decode swapped, TTFT preserved.
        monkeypatch.setattr(overlay_mod, "load_llm_anchor", lambda *a, **k: _anchor(tokps=42.0))
        r = _projected(QWEN3_30B_A3B_MOE_Q4, NPU_MID)
        out = overlay_llm_anchor(r, NPU_MID, QWEN3_30B_A3B_MOE_Q4, _IDENTITY)
        assert out.source == "measured_silicon_anchor"
        assert out.decode_tok_s == pytest.approx(42.0)
        assert out.ttft_s == r.ttft_s  # preserved
        assert out.silicon_anchor_meta["spec_tier_precision"] == "mid_int8"

    def test_keyhole_decode_mult(self, monkeypatch):
        monkeypatch.setattr(overlay_mod, "load_llm_anchor", lambda *a, **k: _anchor(tokps=42.0))
        r = _projected(QWEN3_30B_A3B_MOE_Q4, NPU_MID)
        out = overlay_llm_anchor(r, NPU_MID, QWEN3_30B_A3B_MOE_Q4, _IDENTITY,
                                 decode_mult=0.073)
        assert out.decode_tok_s == pytest.approx(42.0 * 0.073, abs=0.01)

    def test_keyhole_ttft_recompute(self, monkeypatch):
        monkeypatch.setattr(overlay_mod, "load_llm_anchor",
                            lambda *a, **k: _anchor(tokps=42.0, prefill_tokps=2849.0))
        r = _projected(QWEN3_30B_A3B_MOE_Q4, NPU_MID)
        out = overlay_llm_anchor(r, NPU_MID, QWEN3_30B_A3B_MOE_Q4, _IDENTITY,
                                 ttft_mult=2.0)
        # ttft = (1024/prefill_tokps) * ttft_mult
        assert out.ttft_s == pytest.approx((1024.0 / 2849.0) * 2.0, abs=1e-4)

    def test_ttft_preserved_when_no_prefill(self, monkeypatch):
        monkeypatch.setattr(overlay_mod, "load_llm_anchor",
                            lambda *a, **k: _anchor(tokps=42.0, prefill_tokps=0.0))
        r = _projected(QWEN3_30B_A3B_MOE_Q4, NPU_MID)
        out = overlay_llm_anchor(r, NPU_MID, QWEN3_30B_A3B_MOE_Q4, _IDENTITY,
                                 ttft_mult=2.0)
        assert out.ttft_s == r.ttft_s  # no prefill_tokps → preserve

    def test_unmapped_key_unchanged(self, monkeypatch):
        monkeypatch.setattr(overlay_mod, "load_llm_anchor", lambda *a, **k: _anchor())
        r = _projected(QWEN3_30B_A3B_MOE_Q4, NPU_MID)
        assert overlay_llm_anchor(r, NPU_MID, QWEN3_30B_A3B_MOE_Q4,
                                  lambda k: None) is r

    def test_memory_clone_unchanged(self, monkeypatch):
        monkeypatch.setattr(overlay_mod, "load_llm_anchor", lambda *a, **k: _anchor())
        clone = hw_with_memory(NPU_MID, "LPDDR6", 14.0)
        r = _projected(QWEN3_30B_A3B_MOE_Q4, clone)
        assert overlay_llm_anchor(r, clone, QWEN3_30B_A3B_MOE_Q4, _IDENTITY) is r
