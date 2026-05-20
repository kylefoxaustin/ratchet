"""Unit tests for CalibrationSource and the silicon-class defaults table."""
import pytest

from ratchet.calibration.silicon_class import SILICON_CLASS_DEFAULTS
from ratchet.calibration.source import CalibrationSource


class TestCalibrationSource:
    def test_valid_measured(self):
        cs = CalibrationSource(method="measured", reference="x", confidence="high")
        assert cs.method == "measured"
        assert cs.confidence == "high"

    def test_default_requires_low_confidence(self):
        with pytest.raises(ValueError, match="confidence='low'"):
            CalibrationSource(method="default", reference="x", confidence="high")

    def test_default_low_ok(self):
        cs = CalibrationSource(method="default", reference="x", confidence="low")
        assert cs.confidence == "low"

    def test_frozen(self):
        cs = CalibrationSource(method="measured", reference="x", confidence="high")
        with pytest.raises(Exception):
            cs.method = "default"  # type: ignore[misc]


class TestSiliconClassDefaults:
    def test_all_six_classes(self):
        assert set(SILICON_CLASS_DEFAULTS) == {
            "neutron", "lp5x_64", "lp5x_128", "lp5x_128_int8",
            "gddr_class", "unknown",
        }

    def test_neutron_low_util(self):
        assert SILICON_CLASS_DEFAULTS["neutron"]["compute_util_factor"] == 0.19

    def test_gddr_dedicated_vram(self):
        d = SILICON_CLASS_DEFAULTS["gddr_class"]
        assert d["npu_share_default"] == 1.0
        assert d["bandwidth_efficiency"] == 0.85

    def test_unknown_has_no_capability_levels(self):
        assert SILICON_CLASS_DEFAULTS["unknown"]["capability_levels"] is None

    def test_every_class_has_required_keys(self):
        required = {
            "compute_efficiency", "bandwidth_efficiency", "tier_family",
            "compute_util_factor", "llm_prefill_util_factor",
            "llm_decode_bw_realization", "compute_overhead_ms",
            "npu_share_default", "capability_levels",
        }
        for d in SILICON_CLASS_DEFAULTS.values():
            assert required <= set(d)
