"""Unit tests for the Hardware dataclass."""
import dataclasses

import pytest

from ratchet.calibration.source import CalibrationSource
from ratchet.precision.capability import (
    NEUTRON_INT8_ONLY_CAPABILITY,
    NPU_FULL_DTYPE_CAPABILITY,
)
from ratchet.tiers.hardware import Hardware


def _mk(**overrides) -> Hardware:
    base = dict(
        name="Test Tier",
        peak_tops_bf16=100.0,
        peak_tops_int8=200.0,
        peak_tops_fp8=100.0,
        mem_bandwidth_gbs=100.0,
        mem_capacity_gb=16.0,
        mem_bus_width_bits=128,
        mem_type="LPDDR5X",
        mem_data_rate_gtps=8.4,
    )
    base.update(overrides)
    return Hardware(**base)


class TestDefaults:
    def test_calibration_defaults(self):
        hw = _mk()
        assert hw.compute_efficiency == 0.65
        assert hw.bandwidth_efficiency == 0.70
        assert hw.compute_util_factor == 0.45
        assert hw.llm_prefill_util_factor == 0.10
        assert hw.llm_decode_bw_realization == 1.0
        assert hw.compute_overhead_ms == 1.0
        assert hw.npu_share_default == 0.75

    def test_optional_attachments_default_none(self):
        hw = _mk()
        assert hw.capability_levels is None
        assert hw.calibration_source is None
        assert hw.measured_decode_overrides is None
        assert hw.measured_prefill_overrides is None
        assert hw.measured_vision_overrides is None
        assert hw.measured_llm is None

    def test_stock_identity_defaults(self):
        hw = _mk()
        assert hw.bw_projected is False
        assert hw.stock_name is None
        assert hw.stock_mem_bandwidth_gbs is None


class TestEffectiveBandwidth:
    def test_effective_bandwidth(self):
        hw = _mk(mem_bandwidth_gbs=100.0, bandwidth_efficiency=0.70)
        assert hw.effective_bandwidth_gbs == pytest.approx(70.0)

    def test_effective_bandwidth_excludes_npu_share(self):
        # npu_share composes downstream in projection, not in this property.
        hw = _mk(mem_bandwidth_gbs=200.0, bandwidth_efficiency=0.85,
                 npu_share_default=0.5)
        assert hw.effective_bandwidth_gbs == pytest.approx(170.0)


class TestTierLookupName:
    def test_returns_name_when_no_stock(self):
        hw = _mk(name="NPU Mid")
        assert hw.tier_lookup_name == "NPU Mid"

    def test_returns_stock_name_when_clone(self):
        hw = _mk(name="NPU Mid (LPDDR6 @ 14 GT/s)", stock_name="NPU Mid")
        assert hw.tier_lookup_name == "NPU Mid"


class TestEffectiveTops:
    def test_int8(self):
        hw = _mk(peak_tops_int8=200.0, compute_efficiency=0.65)
        assert hw.effective_tops("int8") == pytest.approx(130.0)

    def test_fp16_conflates_to_bf16(self):
        hw = _mk(peak_tops_bf16=100.0, compute_efficiency=0.70)
        assert hw.effective_tops("fp16") == pytest.approx(70.0)
        assert hw.effective_tops("bf16") == pytest.approx(70.0)

    def test_unknown_dtype_falls_back_to_bf16(self):
        hw = _mk(peak_tops_bf16=50.0, compute_efficiency=1.0)
        assert hw.effective_tops("nonsense") == pytest.approx(50.0)

    def test_case_insensitive(self):
        hw = _mk(peak_tops_int8=200.0, compute_efficiency=0.5)
        assert hw.effective_tops("INT8") == pytest.approx(100.0)


class TestGetMeasuredLlmCell:
    def test_none_when_no_measured_llm(self):
        hw = _mk()
        assert hw.get_measured_llm_cell("m", "w") is None

    def test_direct_hit(self):
        cell = {"decode_tok_s": 42.0}
        hw = _mk(measured_llm={"m": {"w": cell}})
        assert hw.get_measured_llm_cell("m", "w") == cell

    def test_miss_on_wrong_workload(self):
        hw = _mk(measured_llm={"m": {"w": {"decode_tok_s": 42.0}}})
        assert hw.get_measured_llm_cell("m", "other") is None

    def test_no_alias_resolution_here(self):
        # Alias resolution happens at the call site, not on Hardware.
        hw = _mk(measured_llm={"qwen25_7b_dense": {"w": {"decode_tok_s": 9.0}}})
        assert hw.get_measured_llm_cell("skippy_7b_v4", "w") is None


class TestMutabilityDiscipline:
    def test_plain_dataclass_not_frozen(self):
        hw = _mk()
        # Mutation is allowed (used by import-time measurement attachment).
        hw.measured_llm = {"m": {"w": {"decode_tok_s": 1.0}}}
        assert hw.measured_llm["m"]["w"]["decode_tok_s"] == 1.0

    def test_replace_produces_independent_copy(self):
        hw = _mk(name="A")
        clone = dataclasses.replace(hw, name="B")
        assert hw.name == "A"
        assert clone.name == "B"


class TestWithProvenanceAndCapability:
    def test_carries_calibration_source(self):
        cs = CalibrationSource(method="measured", reference="x", confidence="high")
        hw = _mk(calibration_source=cs)
        assert hw.calibration_source.method == "measured"

    def test_carries_capability_levels(self):
        hw = _mk(capability_levels=NEUTRON_INT8_ONLY_CAPABILITY)
        assert hw.capability_levels["int8"].level.value == "tensor_native"
        hw2 = _mk(capability_levels=NPU_FULL_DTYPE_CAPABILITY)
        assert hw2.capability_levels["bf16/fp16"].level.value == "tensor_native"
