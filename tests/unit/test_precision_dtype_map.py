"""Unit tests for dtype dispatch, capability keys (AMENDMENT 1), and paths."""
import pytest

from ratchet.precision.capability import (
    NEUTRON_INT8_ONLY_CAPABILITY,
    NPU_FULL_DTYPE_CAPABILITY,
    SM120_BLACKWELL_CAPABILITY,
    CapabilityLevel,
)
from ratchet.precision.deployment_path import deployment_path_for_tier
from ratchet.precision.dtype_map import (
    DTYPE_ATTR_MAP,
    hw_peak_tops_for_dtype,
    hw_supports_dtype,
    hw_supports_dtype_via_key,
    quant_scheme_capability_key,
)
from ratchet.tiers.hardware import Hardware


def _mk(**overrides) -> Hardware:
    base = dict(
        name="T", peak_tops_bf16=0.0, peak_tops_int8=200.0, peak_tops_fp8=0.0,
        mem_bandwidth_gbs=100.0, mem_capacity_gb=16.0, mem_bus_width_bits=128,
        mem_type="LPDDR5X", mem_data_rate_gtps=8.4,
    )
    base.update(overrides)
    return Hardware(**base)


class TestDtypeAttrMap:
    def test_keys(self):
        assert DTYPE_ATTR_MAP["int8"] == "peak_tops_int8"
        assert DTYPE_ATTR_MAP["fp8"] == "peak_tops_fp8"
        assert DTYPE_ATTR_MAP["bf16"] == "peak_tops_bf16"
        assert DTYPE_ATTR_MAP["fp16"] == "peak_tops_bf16"


class TestPeakTopsForDtype:
    def test_resolves_field(self):
        hw = _mk(peak_tops_int8=200.0)
        assert hw_peak_tops_for_dtype(hw, "int8") == 200.0

    def test_fp16_conflates(self):
        hw = _mk(peak_tops_bf16=99.0)
        assert hw_peak_tops_for_dtype(hw, "fp16") == 99.0

    def test_unknown_returns_zero(self):
        # Distinct from effective_tops(), which falls back to bf16.
        hw = _mk(peak_tops_bf16=99.0)
        assert hw_peak_tops_for_dtype(hw, "nonsense") == 0.0


class TestQuantSchemeCapabilityKey:
    @pytest.mark.parametrize("scheme,expected", [
        ("Q4_K_M", "q4_km"),
        ("Q5_K_M", "q4_km"),
        ("Q8_0", "q4_km"),
        ("INT8_W8A8", "int8"),
        ("FP8", "fp8"),
        ("FP16", "bf16/fp16"),
        ("BF16", "bf16/fp16"),
    ])
    def test_known_schemes(self, scheme, expected):
        assert quant_scheme_capability_key(scheme) == expected


class TestHwSupportsDtype:
    def test_neutron_int8_native(self):
        hw = _mk(capability_levels=NEUTRON_INT8_ONLY_CAPABILITY)
        assert hw_supports_dtype(hw, "int8") is CapabilityLevel.TENSOR_NATIVE

    def test_neutron_fp16_unsupported(self):
        hw = _mk(capability_levels=NEUTRON_INT8_ONLY_CAPABILITY)
        assert hw_supports_dtype(hw, "fp16") is CapabilityLevel.UNSUPPORTED
        assert hw_supports_dtype(hw, "bf16") is CapabilityLevel.UNSUPPORTED

    def test_sm120_int8_compat(self):
        hw = _mk(capability_levels=SM120_BLACKWELL_CAPABILITY)
        assert hw_supports_dtype(hw, "int8") is CapabilityLevel.TENSOR_COMPAT

    def test_heuristic_fallback_when_none(self):
        hw = _mk(capability_levels=None, peak_tops_int8=200.0, peak_tops_bf16=0.0)
        assert hw_supports_dtype(hw, "int8") is CapabilityLevel.TENSOR_NATIVE
        assert hw_supports_dtype(hw, "bf16") is CapabilityLevel.UNSUPPORTED


class TestAmendment1QuantGating:
    def test_q4km_runs_on_neutron_int8_silicon(self):
        # The crux of AMENDMENT 1: a Q4_K_M model (compute_dtype fp16) must be
        # executable on INT8-only Neutron silicon via the dequant path.
        hw = _mk(capability_levels=NEUTRON_INT8_ONLY_CAPABILITY)
        key = quant_scheme_capability_key("Q4_K_M")
        assert hw_supports_dtype_via_key(hw, key) is CapabilityLevel.TENSOR_NATIVE
        # ...whereas gating on the raw compute dtype would wrongly reject it.
        assert hw_supports_dtype(hw, "fp16") is CapabilityLevel.UNSUPPORTED

    def test_q4km_heuristic_fallback_prefers_int8(self):
        hw = _mk(capability_levels=None, peak_tops_int8=200.0, peak_tops_bf16=0.0)
        key = quant_scheme_capability_key("Q4_K_M")
        assert hw_supports_dtype_via_key(hw, key) is CapabilityLevel.TENSOR_NATIVE


class TestDeploymentPath:
    def test_native_fast(self):
        hw = _mk(capability_levels=NPU_FULL_DTYPE_CAPABILITY, peak_tops_bf16=200.0)
        assert deployment_path_for_tier(hw, "bf16", "fresh_compile") == "native_fast"

    def test_compat_fast_for_precompiled(self):
        hw = _mk(capability_levels=SM120_BLACKWELL_CAPABILITY)
        assert deployment_path_for_tier(hw, "int8", "precompiled") == "compat_fast"

    def test_compat_blocked_for_fresh_compile(self):
        hw = _mk(capability_levels=SM120_BLACKWELL_CAPABILITY)
        assert deployment_path_for_tier(hw, "int8", "fresh_compile") == "compat_blocked"

    def test_unsupported(self):
        hw = _mk(capability_levels=NEUTRON_INT8_ONLY_CAPABILITY)
        assert deployment_path_for_tier(hw, "fp8", "precompiled") == "unsupported"


# ── NVFP4 / FP4 (added v0.2.5) — FP4 is a COMPUTE format on Blackwell sm_120 ──
def test_nvfp4_routes_to_fp4_peak_and_is_native_on_sm120():
    from ratchet.tiers.registry import RTX_5090_REFERENCE as hw
    from ratchet.precision.dtype_map import (
        hw_peak_tops_for_dtype, quant_scheme_capability_key, DTYPE_ATTR_MAP)
    from ratchet.precision.capability import (
        SM120_BLACKWELL_CAPABILITY, NEUTRON_INT8_ONLY_CAPABILITY,
        NPU_FULL_DTYPE_CAPABILITY, CapabilityLevel)

    # nvfp4 + aliases route to the new peak_tops_fp4 silicon field
    for dt in ("nvfp4", "fp4", "mxfp4"):
        assert DTYPE_ATTR_MAP[dt] == "peak_tops_fp4"
    assert hw.peak_tops_fp4 == 1676.0
    assert hw_peak_tops_for_dtype(hw, "nvfp4") == 1676.0

    # FP4 quant schemes gate on the fp4 capability key; weight-only INT4 does NOT
    # (it dequantizes to bf16 — the measured memory-format / bf16-prefill-floor result).
    assert quant_scheme_capability_key("NVFP4") == "nvfp4"
    assert quant_scheme_capability_key("MXFP4") == "nvfp4"
    assert quant_scheme_capability_key("INT4_AWQ") == "bf16/fp16"

    # capability: native on Blackwell sm_120, unsupported on edge NPUs (no FP4 path)
    assert SM120_BLACKWELL_CAPABILITY["nvfp4"].level is CapabilityLevel.TENSOR_NATIVE
    assert NEUTRON_INT8_ONLY_CAPABILITY["nvfp4"].level is CapabilityLevel.UNSUPPORTED
    assert NPU_FULL_DTYPE_CAPABILITY["nvfp4"].level is CapabilityLevel.UNSUPPORTED


def test_peak_tops_fp4_is_defaulted_zero():
    # the field is defaulted (=0.0) so existing/edge tier constructions don't break
    # and any tier without an explicit FP4 path reads 0.0.
    import dataclasses
    from ratchet.tiers.hardware import Hardware
    field = {f.name: f for f in dataclasses.fields(Hardware)}["peak_tops_fp4"]
    assert field.default == 0.0


# ── ADR 016: FP4's compute win is runtime-conditional ──
class TestFP4RuntimeMaturity:
    def test_is_fp4_compute_dtype(self):
        from ratchet.precision.dtype_map import is_fp4_compute_dtype
        for dt in ("nvfp4", "fp4", "mxfp4", "NVFP4", "MXFP4"):
            assert is_fp4_compute_dtype(dt) is True
        for dt in ("bf16", "fp16", "fp8", "int8"):
            assert is_fp4_compute_dtype(dt) is False

    def test_effective_compute_dtype_immature_fp4_falls_to_bf16(self):
        from ratchet.precision.dtype_map import effective_compute_dtype
        # immature + FP4 -> bf16 floor (modeled as INT4 weight-only)
        assert effective_compute_dtype("nvfp4", "immature") == "bf16"
        assert effective_compute_dtype("mxfp4", "immature") == "bf16"
        # mature + FP4 -> unchanged (realizes the native FP4 win)
        assert effective_compute_dtype("nvfp4", "mature") == "nvfp4"
        # default is mature (non-breaking)
        assert effective_compute_dtype("nvfp4") == "nvfp4"
        # non-FP4 dtypes are identity regardless of maturity
        assert effective_compute_dtype("int8", "immature") == "int8"
        assert effective_compute_dtype("bf16", "immature") == "bf16"

    def test_deployment_path_fp4_mature_native_immature_caveat(self):
        hw = _mk(capability_levels=SM120_BLACKWELL_CAPABILITY, peak_tops_fp4=1676.0)
        # mature (and the default) -> native_fast
        assert deployment_path_for_tier(hw, "nvfp4", "fresh_compile") == "native_fast"
        assert deployment_path_for_tier(
            hw, "nvfp4", "fresh_compile", "mature") == "native_fast"
        # immature -> the FP4-specific caveat label
        assert deployment_path_for_tier(
            hw, "nvfp4", "fresh_compile", "immature") == "fp4_runtime_immature"

    def test_deployment_path_immature_only_affects_fp4(self):
        # immature maturity must not perturb a non-FP4 dtype's path
        hw = _mk(capability_levels=NPU_FULL_DTYPE_CAPABILITY, peak_tops_bf16=200.0)
        assert deployment_path_for_tier(
            hw, "bf16", "fresh_compile", "immature") == "native_fast"
