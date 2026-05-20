"""Unit tests for the anchor-secrets loader, spec routing, and overlay."""
from types import SimpleNamespace

import pytest

from ratchet.anchors import loader as loader_mod
from ratchet.anchors import overlay as overlay_mod
from ratchet.anchors.loader import LLMAnchor, load_llm_anchor
from ratchet.anchors.overlay import overlay_llm_anchor
from ratchet.anchors.spec_routing import hw_to_anchor_tier_dtype
from ratchet.catalog.reference import QWEN3_30B_A3B_MOE_Q4
from ratchet.projection.llm import project_llm
from ratchet.projection.result import Projected
from ratchet.tiers.memory_overlay import hw_with_memory
from ratchet.tiers.registry import NPU_HIGH, NPU_MID, RTX_5090_REFERENCE


# ─── Loader fallback semantics ──────────────────────────────────────

class TestLoaderFallback:
    def test_none_without_streamlit(self, monkeypatch):
        monkeypatch.setattr(loader_mod, "_HAS_STREAMLIT", False)
        assert load_llm_anchor("mid_int8", "qwen3_30b_a3b_moe") is None

    def test_none_when_cell_absent(self, monkeypatch):
        monkeypatch.setattr(loader_mod, "_HAS_STREAMLIT", True)
        monkeypatch.setattr(loader_mod, "st",
                            SimpleNamespace(secrets={"npu_llm_anchors": {}}))
        assert load_llm_anchor("mid_int8", "qwen3_30b_a3b_moe") is None

    def test_loads_valid_cell(self, monkeypatch):
        secrets = {"npu_llm_anchors": {"mid_int8": {"qwen3_30b_a3b_moe": {
            "tokps": 40.0, "ms_per_inference": 12.0, "peak_bw_gbps": 134.4,
            "source": "measured", "measured_date": "2026-04-15",
        }}}}
        monkeypatch.setattr(loader_mod, "_HAS_STREAMLIT", True)
        monkeypatch.setattr(loader_mod, "st", SimpleNamespace(secrets=secrets))
        anchor = load_llm_anchor("mid_int8", "qwen3_30b_a3b_moe")
        assert anchor.tokps == 40.0
        assert anchor.bw_share_frac == 0.75  # default fill
        assert anchor.measured_date == "2026-04-15"

    def test_none_on_malformed_value(self, monkeypatch):
        secrets = {"npu_llm_anchors": {"mid_int8": {"m": {
            "tokps": "not_a_number", "ms_per_inference": 1.0, "peak_bw_gbps": 1.0,
        }}}}
        monkeypatch.setattr(loader_mod, "_HAS_STREAMLIT", True)
        monkeypatch.setattr(loader_mod, "st", SimpleNamespace(secrets=secrets))
        assert load_llm_anchor("mid_int8", "m") is None


# ─── Spec routing ───────────────────────────────────────────────────

class TestSpecRouting:
    def test_mid_q4km(self):
        assert hw_to_anchor_tier_dtype(NPU_MID, "q4_km") == "mid_int8"

    def test_mid_int8(self):
        assert hw_to_anchor_tier_dtype(NPU_MID, "int8") == "mid_int8"

    def test_high_int8(self):
        assert hw_to_anchor_tier_dtype(NPU_HIGH, "int8") == "high_int8"

    def test_high_fp(self):
        assert hw_to_anchor_tier_dtype(NPU_HIGH, "bf16") == "high_fp"
        assert hw_to_anchor_tier_dtype(NPU_HIGH, "q4_km") == "high_fp"

    def test_reference_tier_no_cell(self):
        assert hw_to_anchor_tier_dtype(RTX_5090_REFERENCE, "fp8") is None

    def test_memory_clone_returns_none(self):
        clone = hw_with_memory(NPU_MID, "LPDDR6", 14.0)
        assert hw_to_anchor_tier_dtype(clone, "q4_km") is None


# ─── Overlay ────────────────────────────────────────────────────────

def _projected_for(model, hw) -> Projected:
    result = project_llm(model, hw, "rag_qa")
    assert isinstance(result, Projected)
    return result


_IDENTITY = lambda k: k


class TestOverlay:
    def test_no_secrets_returns_unchanged(self, monkeypatch):
        # load returns None (no secrets) → result passes through.
        monkeypatch.setattr(overlay_mod, "load_llm_anchor", lambda *a, **k: None)
        result = _projected_for(QWEN3_30B_A3B_MOE_Q4, NPU_MID)
        out = overlay_llm_anchor(result, NPU_MID, QWEN3_30B_A3B_MOE_Q4, _IDENTITY)
        assert out is result

    def test_swaps_in_anchor(self, monkeypatch):
        anchor = LLMAnchor(tokps=42.0, ms_per_inference=20.0, peak_bw_gbps=134.4,
                           source="measured", measured_date="2026-04-15")
        monkeypatch.setattr(overlay_mod, "load_llm_anchor", lambda *a, **k: anchor)
        result = _projected_for(QWEN3_30B_A3B_MOE_Q4, NPU_MID)
        out = overlay_llm_anchor(result, NPU_MID, QWEN3_30B_A3B_MOE_Q4, _IDENTITY)
        assert out.source == "measured_silicon_anchor"
        assert out.decode_tok_s == pytest.approx(42.0)
        assert out.ttft_s == pytest.approx(0.02)
        assert out.silicon_anchor_meta["tier_dtype"] == "mid_int8"

    def test_workload_multiplier_applied(self, monkeypatch):
        anchor = LLMAnchor(tokps=42.0, ms_per_inference=20.0, peak_bw_gbps=134.4)
        monkeypatch.setattr(overlay_mod, "load_llm_anchor", lambda *a, **k: anchor)
        result = _projected_for(QWEN3_30B_A3B_MOE_Q4, NPU_MID)
        out = overlay_llm_anchor(result, NPU_MID, QWEN3_30B_A3B_MOE_Q4, _IDENTITY,
                                 workload_multiplier=0.5)
        assert out.decode_tok_s == pytest.approx(21.0)

    def test_no_spec_cell_returns_unchanged(self, monkeypatch):
        # RTX 5090 has no anchor spec cell → unchanged regardless of secrets.
        anchor = LLMAnchor(tokps=42.0, ms_per_inference=20.0, peak_bw_gbps=1.0)
        monkeypatch.setattr(overlay_mod, "load_llm_anchor", lambda *a, **k: anchor)
        result = _projected_for(QWEN3_30B_A3B_MOE_Q4, RTX_5090_REFERENCE)
        out = overlay_llm_anchor(result, RTX_5090_REFERENCE,
                                 QWEN3_30B_A3B_MOE_Q4, _IDENTITY)
        assert out is result

    def test_unmapped_spec_key_returns_unchanged(self, monkeypatch):
        anchor = LLMAnchor(tokps=42.0, ms_per_inference=20.0, peak_bw_gbps=1.0)
        monkeypatch.setattr(overlay_mod, "load_llm_anchor", lambda *a, **k: anchor)
        result = _projected_for(QWEN3_30B_A3B_MOE_Q4, NPU_MID)
        out = overlay_llm_anchor(result, NPU_MID, QWEN3_30B_A3B_MOE_Q4,
                                 lambda k: None)  # surface can't map this key
        assert out is result

    def test_memory_clone_returns_unchanged(self, monkeypatch):
        anchor = LLMAnchor(tokps=42.0, ms_per_inference=20.0, peak_bw_gbps=1.0)
        monkeypatch.setattr(overlay_mod, "load_llm_anchor", lambda *a, **k: anchor)
        clone = hw_with_memory(NPU_MID, "LPDDR6", 14.0)
        result = _projected_for(QWEN3_30B_A3B_MOE_Q4, clone)
        out = overlay_llm_anchor(result, clone, QWEN3_30B_A3B_MOE_Q4, _IDENTITY)
        assert out is result
