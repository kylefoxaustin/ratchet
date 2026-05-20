"""End-to-end: tier → memory-upgrade clone → projection."""
import pytest

from ratchet import NPU_MID, Projected, hw_with_memory, project_llm
from ratchet.catalog.reference import QWEN3_30B_A3B_MOE_Q4


class TestMemoryUpgradeProjection:
    def test_clone_decode_scales_with_bandwidth(self):
        stock = project_llm(QWEN3_30B_A3B_MOE_Q4, NPU_MID, "rag_qa")
        clone_hw = hw_with_memory(NPU_MID, "LPDDR6", 14.0)  # 134.4 → 224 GB/s
        clone = project_llm(QWEN3_30B_A3B_MOE_Q4, clone_hw, "rag_qa")
        assert isinstance(clone, Projected)
        # Decode anchor was BW-scaled on the clone (decode is BW-bound).
        ratio = 224.0 / 134.4
        assert clone.decode_tok_s == pytest.approx(stock.decode_tok_s * ratio, rel=1e-3)

    def test_clone_routes_to_stock_identity(self):
        clone_hw = hw_with_memory(NPU_MID, "LPDDR6", 14.0)
        assert clone_hw.tier_lookup_name == "NPU Mid"
        assert clone_hw.bw_projected is True

    def test_prefill_held_under_upgrade(self):
        clone_hw = hw_with_memory(NPU_MID, "LPDDR6", 14.0)
        # Prefill override is compute-bound → held at stock value.
        assert clone_hw.measured_prefill_overrides["qwen3_30b_a3b_moe"] == 2849.0
