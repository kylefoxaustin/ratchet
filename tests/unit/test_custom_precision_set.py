"""ADR 017: make_custom_tier(npu_precision_set=...) + projection mechanics.

The precision rung is orthogonal to silicon_class: it overrides the capability
dict, zeros the peak_tops_* the rung can't use, and (via Hardware.npu_precision_set)
makes project_llm run the model at the rung's dtype so the selector isn't inert.
"""
import pytest

from ratchet import make_custom_tier, project_llm
from ratchet.catalog.model import LLMModel
from ratchet.precision.capability import CapabilityLevel
from ratchet.tiers.registry import NPU_MID


def _tier(npu_precision_set=None, **ov):
    base = dict(
        name="High-class",
        silicon_class="lp5x_128",
        peak_tops_int8=400.0, peak_tops_fp8=400.0, peak_tops_fp4=800.0,
        peak_tops_bf16=200.0,
        mem_bandwidth_gbs=120.0, mem_capacity_gb=32.0, mem_bus_width_bits=128,
        mem_type="LPDDR5X", mem_data_rate_gtps=8.4,
        npu_precision_set=npu_precision_set,
    )
    base.update(ov)
    return make_custom_tier(**base)


def _q4_moe():
    # Q4_K_M model: nominal compute_dtype 'fp16' (so its floor would read bf16
    # without the rung override) — the sizer's A/B/C subject.
    return LLMModel(
        key="moe", family="F", base="B", total_params_b=30.0, active_params_b=3.0,
        quant_scheme="Q4_K_M", bytes_per_param=0.5, gguf_size_gb=16.0,
        size_gb_inflight=16.0, compute_dtype="fp16",
        num_layers=48, hidden_dim=2048, num_attention_heads=32, num_kv_heads=4)


class TestFactoryCapabilityOverride:
    def test_int8_rung_caps(self):
        hw = _tier("int8")
        assert hw.capability_levels["int8"].level is CapabilityLevel.TENSOR_NATIVE
        assert hw.capability_levels["fp8"].level is CapabilityLevel.UNSUPPORTED
        assert hw.capability_levels["nvfp4"].level is CapabilityLevel.UNSUPPORTED

    def test_int8_fp8_rung_caps(self):
        hw = _tier("int8_fp8")
        assert hw.capability_levels["fp8"].level is CapabilityLevel.TENSOR_NATIVE
        assert hw.capability_levels["nvfp4"].level is CapabilityLevel.UNSUPPORTED

    def test_int8_fp8_fp4_rung_caps(self):
        hw = _tier("int8_fp8_fp4")
        assert hw.capability_levels["fp8"].level is CapabilityLevel.TENSOR_NATIVE
        assert hw.capability_levels["nvfp4"].level is CapabilityLevel.TENSOR_NATIVE

    def test_peak_tops_gated_per_rung(self):
        # int8 zeros fp8/fp4/bf16; int8_fp8 zeros fp4; full keeps all.
        i8 = _tier("int8")
        assert (i8.peak_tops_int8, i8.peak_tops_fp8, i8.peak_tops_fp4,
                i8.peak_tops_bf16) == (400.0, 0.0, 0.0, 0.0)
        i8f8 = _tier("int8_fp8")
        assert (i8f8.peak_tops_fp8, i8f8.peak_tops_fp4) == (400.0, 0.0)
        full = _tier("int8_fp8_fp4")
        assert (full.peak_tops_fp8, full.peak_tops_fp4) == (400.0, 800.0)

    def test_precision_set_stamped_on_hardware(self):
        assert _tier("int8_fp8_fp4").npu_precision_set == "int8_fp8_fp4"

    def test_none_is_non_breaking(self):
        # default (no precision set) preserves silicon_class capability + leaves
        # the field None; canonical tiers are likewise None.
        assert _tier(None).npu_precision_set is None
        assert NPU_MID.npu_precision_set is None

    def test_unknown_precision_set_raises(self):
        with pytest.raises(ValueError, match="unknown npu_precision_set"):
            _tier("int4")


class TestPrecisionSetProjection:
    """The benefit ladder, as projected prefill TTFT for the same Q4 model."""

    def _ttft_ms(self, ps, maturity="mature"):
        r = project_llm(_q4_moe(), _tier(ps), "w", prompt_tokens=1000,
                        fp4_runtime_maturity=maturity)
        return r.ttft_s * 1000.0

    def test_rung_is_not_inert(self):
        # The whole point: an INT8 rung must beat the naive fp16 floor (else the
        # selector shows zero benefit). peak_tops_int8(400) > peak_tops_bf16(200).
        assert self._ttft_ms("int8") < self._ttft_ms(None)

    def test_fp8_equals_int8(self):
        # FP8 == INT8 TOPS (same 8-bit datapath) -> identical prefill.
        assert self._ttft_ms("int8_fp8") == pytest.approx(self._ttft_ms("int8"))

    def test_fp4_mature_beats_fp8(self):
        # FP4 (mature) doubles the 8-bit rate -> faster prefill than the fp8 rung.
        assert self._ttft_ms("int8_fp8_fp4", "mature") < self._ttft_ms("int8_fp8")

    def test_fp4_immature_collapses_to_naive_floor(self):
        # ADR-016: on an immature runtime FP4 falls to the bf16 floor == naive.
        assert self._ttft_ms("int8_fp8_fp4", "immature") == pytest.approx(
            self._ttft_ms(None))

    def test_fp4_model_executes_on_fp4_rung(self):
        # An actual NVFP4 model is accepted on the FP4 rung (capability native)...
        nvfp4_model = LLMModel(
            key="m4", family="F", base="B", total_params_b=8.0, active_params_b=8.0,
            quant_scheme="NVFP4", bytes_per_param=0.5, gguf_size_gb=5.0,
            size_gb_inflight=5.0, compute_dtype="nvfp4",
            num_layers=32, hidden_dim=4096, num_attention_heads=32, num_kv_heads=8)
        from ratchet.projection.result import Projected, DtypeMismatch
        ok = project_llm(nvfp4_model, _tier("int8_fp8_fp4"), "w", prompt_tokens=1000)
        assert isinstance(ok, Projected)
        # ...but rejected on the int8_fp8 rung (nvfp4 unsupported there).
        bad = project_llm(nvfp4_model, _tier("int8_fp8"), "w", prompt_tokens=1000)
        assert isinstance(bad, DtypeMismatch)
