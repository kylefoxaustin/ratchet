"""Unit tests for make_custom_tier()."""
import pytest

from ratchet.precision.capability import CapabilityLevel
from ratchet.precision.dtype_map import hw_supports_dtype
from ratchet.tiers.custom import make_custom_tier


def _neutron(**ov):
    base = dict(
        name="My Neutron Chip",
        silicon_class="neutron",
        peak_tops_int8=2.0,
        mem_bandwidth_gbs=25.6,
        mem_capacity_gb=8.0,
        mem_bus_width_bits=32,
        mem_type="LPDDR5",
        mem_data_rate_gtps=6.4,
    )
    base.update(ov)
    return make_custom_tier(**base)


class TestSiliconClassDefaulting:
    def test_neutron_picks_low_util(self):
        hw = _neutron()
        assert hw.compute_util_factor == 0.19  # not the 0.45 NPU-Mid default
        assert hw.tier_family == "Neutron-custom"

    def test_neutron_capability_int8_only(self):
        hw = _neutron()
        assert hw_supports_dtype(hw, "int8") is CapabilityLevel.TENSOR_NATIVE
        assert hw_supports_dtype(hw, "fp16") is CapabilityLevel.UNSUPPORTED

    def test_gddr_class_defaults(self):
        hw = make_custom_tier(
            name="My GPU", silicon_class="gddr_class",
            peak_tops_int8=400.0, peak_tops_bf16=200.0, peak_tops_fp8=400.0,
            mem_bandwidth_gbs=1000.0, mem_capacity_gb=24.0,
            mem_bus_width_bits=384, mem_type="GDDR7", mem_data_rate_gtps=20.0,
        )
        assert hw.npu_share_default == 1.0
        assert hw.compute_overhead_ms == 0.3

    def test_unknown_class_no_capability(self):
        hw = _neutron(silicon_class="unknown")
        assert hw.capability_levels is None


class TestCalibrationProvenance:
    def test_always_default_low(self):
        hw = _neutron()
        assert hw.calibration_source.method == "default"
        assert hw.calibration_source.confidence == "low"

    def test_reference_records_silicon_class(self):
        hw = _neutron()
        assert "silicon_class=neutron" in hw.calibration_source.reference


class TestOverrides:
    def test_explicit_efficiency_overrides_default(self):
        hw = _neutron(compute_efficiency=0.5, bandwidth_efficiency=0.9)
        assert hw.compute_efficiency == 0.5
        assert hw.bandwidth_efficiency == 0.9

    def test_default_efficiency_when_omitted(self):
        hw = _neutron()
        assert hw.compute_efficiency == 0.60


class TestValidation:
    def test_bad_silicon_class_raises(self):
        with pytest.raises(ValueError, match="unknown silicon_class"):
            make_custom_tier(
                name="x", silicon_class="bogus",  # type: ignore[arg-type]
                peak_tops_int8=2.0, mem_bandwidth_gbs=10.0, mem_capacity_gb=8.0,
                mem_bus_width_bits=32, mem_type="LPDDR5", mem_data_rate_gtps=6.4,
            )
