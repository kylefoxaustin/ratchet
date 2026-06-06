"""Unit tests for the TIERS registry and memory overlay."""
import pytest

from ratchet.tiers.memory_overlay import MEMORY_UPGRADE_OPTIONS, hw_with_memory
from ratchet.tiers.registry import (
    IMX93_MEASURED,
    IMX95_MEASURED,
    NPU_HIGH,
    NPU_LOW_LP4,
    NPU_LOW_LP5_32BIT,
    NPU_LOW_LP5_64BIT,
    NPU_LOW_LP5X,
    NPU_MID,
    RTX_5090_REFERENCE,
    TIERS,
)


class TestRegistryIntegrity:
    def test_nine_tiers(self):
        # 8 original + i.MX 93 (added v0.3.0 for drone-sizer, ADR 018).
        assert len(TIERS) == 9

    def test_keyed_by_name(self):
        for name, hw in TIERS.items():
            assert hw.name == name

    def test_all_named_tiers_present(self):
        for hw in (NPU_LOW_LP4, NPU_LOW_LP5_32BIT, NPU_LOW_LP5_64BIT,
                   NPU_LOW_LP5X, IMX93_MEASURED, IMX95_MEASURED, NPU_MID,
                   NPU_HIGH, RTX_5090_REFERENCE):
            assert TIERS[hw.name] is hw

    def test_every_tier_has_calibration_source(self):
        for hw in TIERS.values():
            assert hw.calibration_source is not None

    def test_bandwidth_matches_bus_and_rate(self):
        # mem_bandwidth_gbs == bus_width_bits * data_rate_gtps / 8
        for hw in TIERS.values():
            expected = hw.mem_bus_width_bits * hw.mem_data_rate_gtps / 8.0
            assert hw.mem_bandwidth_gbs == pytest.approx(expected, rel=1e-3)


class TestTierFacts:
    def test_npu_mid_int8_only(self):
        assert NPU_MID.peak_tops_int8 == 200.0
        assert NPU_MID.peak_tops_bf16 == 0.0

    def test_npu_mid_carries_moe_anchor(self):
        assert NPU_MID.measured_decode_overrides["qwen3_30b_a3b_moe"] == 37.85
        assert NPU_MID.measured_prefill_overrides["qwen3_30b_a3b_moe"] == 2849.0

    def test_5090_reference_dedicated_vram(self):
        assert RTX_5090_REFERENCE.npu_share_default == 1.0
        assert RTX_5090_REFERENCE.bandwidth_efficiency == 0.85
        assert RTX_5090_REFERENCE.compute_overhead_ms == 0.3

    def test_5090_ships_empty_measured_llm(self):
        # Surfaces populate this at import time; ratchet ships it empty.
        assert RTX_5090_REFERENCE.measured_llm == {}

    def test_imx95_vision_override(self):
        cell = IMX95_MEASURED.measured_vision_overrides[
            "yolov8n_trt_int8_coco128"]["1920x1080"]
        assert cell["ms_per_inference"] == 32.0


class TestMemoryOverlay:
    def test_recomputes_bandwidth(self):
        clone = hw_with_memory(NPU_MID, "LPDDR6", 14.0)
        # 128-bit bus * 14 GT/s / 8 = 224 GB/s
        assert clone.mem_bandwidth_gbs == pytest.approx(224.0)

    def test_sets_bw_projected_and_stock_identity(self):
        clone = hw_with_memory(NPU_MID, "LPDDR6", 14.0)
        assert clone.bw_projected is True
        assert clone.stock_name == "NPU Mid"
        assert clone.stock_mem_bandwidth_gbs == pytest.approx(134.4)

    def test_tier_lookup_name_resolves_to_stock(self):
        clone = hw_with_memory(NPU_MID, "LPDDR6", 14.0)
        assert clone.tier_lookup_name == "NPU Mid"

    def test_decode_override_bw_scaled(self):
        clone = hw_with_memory(NPU_MID, "LPDDR6", 14.0)
        ratio = 224.0 / 134.4
        assert clone.measured_decode_overrides["qwen3_30b_a3b_moe"] == pytest.approx(
            37.85 * ratio
        )

    def test_prefill_override_held_at_stock(self):
        clone = hw_with_memory(NPU_MID, "LPDDR6", 14.0)
        assert clone.measured_prefill_overrides["qwen3_30b_a3b_moe"] == 2849.0

    def test_default_name_suffix(self):
        clone = hw_with_memory(NPU_MID, "LPDDR6", 14.0)
        assert clone.name == "NPU Mid (LPDDR6 @ 14.0 GT/s)"

    def test_custom_name_suffix(self):
        clone = hw_with_memory(NPU_MID, "LPDDR6", 14.0, name_suffix="fast")
        assert clone.name == "NPU Mid (fast)"

    def test_stock_tier_unchanged(self):
        hw_with_memory(NPU_MID, "LPDDR6", 14.0)
        assert NPU_MID.mem_bandwidth_gbs == 134.4
        assert NPU_MID.bw_projected is False

    def test_double_clone_keeps_original_stock(self):
        c1 = hw_with_memory(NPU_MID, "LPDDR5T", 11.2)
        c2 = hw_with_memory(c1, "LPDDR6", 14.0)
        assert c2.stock_name == "NPU Mid"
        assert c2.stock_mem_bandwidth_gbs == pytest.approx(134.4)

    def test_options_present(self):
        assert len(MEMORY_UPGRADE_OPTIONS) == 3
