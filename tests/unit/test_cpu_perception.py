"""v0.3.0 R1-R4: CPU complex, perception anchors, CPU-workload projection.

The drone-sizer engine extensions. R1 (CpuComplex on Hardware), R2 (frame-latency
+ solver-convergence projection), R3 (per-workload BW queryable), R4 (i.MX 93 tier
+ CPU fields). Cut A: ratchet provides mechanism + math + silicon facts; the
verdicts/composition are surface-side and are NOT tested here.
"""
import dataclasses

import pytest

from ratchet.projection.cpu import (
    project_frame_latency,
    project_solver_convergence,
)
from ratchet.tiers.cpu import CpuComplex
from ratchet.tiers.perception import (
    LatencyDistribution,
    PerceptionAnchor,
    SolverConvergenceAnchor,
)
from ratchet.tiers.registry import (
    IMX93_MEASURED,
    IMX95_MEASURED,
    NPU_HIGH,
    NPU_LOW_LP4,
    NPU_MID,
    RTX_5090_REFERENCE,
)
from ratchet.calibration.source import CalibrationSource


_MEASURED = CalibrationSource(method="measured", reference="test", confidence="high")
_PROJECTED = CalibrationSource(method="projected", reference="test", confidence="low")


# ── R1: CPU complex ──
class TestR1CpuComplex:
    def test_drone_tiers_carry_cpu(self):
        assert IMX93_MEASURED.cpu == CpuComplex(2, "A55", 2.0)
        assert IMX95_MEASURED.cpu == CpuComplex(6, "A55", 2.0)
        assert NPU_MID.cpu == CpuComplex(8, "A720", 2.0)
        assert NPU_HIGH.cpu == CpuComplex(8, "A720", 2.0)

    def test_non_drone_tiers_have_no_cpu(self):
        # Non-breaking: NPU-only tiers (PAI/keyhole consumers) leave cpu None.
        assert NPU_LOW_LP4.cpu is None
        assert RTX_5090_REFERENCE.cpu is None

    def test_cpu_field_defaults_none(self):
        field = {f.name: f for f in dataclasses.fields(type(NPU_MID))}["cpu"]
        assert field.default is None


# ── R4: i.MX 93 tier ──
class TestR4Imx93:
    def test_imx93_silicon_facts(self):
        assert IMX93_MEASURED.cpu == CpuComplex(2, "A55", 2.0)
        assert IMX93_MEASURED.mem_type == "LPDDR4X"
        # bandwidth == bus * rate / 8 (16-bit LPDDR4X-3733)
        assert IMX93_MEASURED.mem_bandwidth_gbs == pytest.approx(7.466, rel=1e-3)
        assert IMX93_MEASURED.calibration_source.method == "measured"


# ── Perception anchor shape + R3 ──
class TestPerceptionAnchorAndR3:
    def _hw_with_anchor(self):
        anchor = PerceptionAnchor(
            workload_class="openvins",
            latency=LatencyDistribution(median_ms=5.0, p95_ms=9.0, max_ms=20.0),
            latency_source=_MEASURED,
            bw_gbs=1.38, bw_source=_MEASURED,
            cores=1.3, cores_source=_MEASURED,
        )
        return dataclasses.replace(
            IMX95_MEASURED, measured_perception={"openvins": anchor})

    def test_get_perception_anchor_absent(self):
        # Stock tiers ship no perception anchors (surface attaches them).
        assert IMX95_MEASURED.get_perception_anchor("openvins") is None
        assert IMX95_MEASURED.measured_perception is None

    def test_get_perception_anchor_present_and_bw_queryable(self):
        hw = self._hw_with_anchor()
        a = hw.get_perception_anchor("openvins")
        assert a is not None
        # R3: BW is queryable with its calibration_source.
        assert a.bw_gbs == 1.38
        assert a.bw_source.method == "measured"
        assert hw.get_perception_anchor("orb_slam3") is None

    def test_per_metric_calibration_source(self):
        # Within one anchor, latency can be measured while bw is projected
        # (drone-sizer amendment 3: only OpenVINS BW was measured).
        anchor = PerceptionAnchor(
            workload_class="orb_slam3",
            latency=LatencyDistribution(8.3, 15.0),
            latency_source=_MEASURED,
            bw_gbs=2.0, bw_source=_PROJECTED,
        )
        assert anchor.latency_source.method == "measured"
        assert anchor.bw_source.method == "projected"


# ── R2a: frame-latency projection ──
class TestR2aFrameLatency:
    def test_scales_whole_distribution(self):
        base = LatencyDistribution(median_ms=100.0, p95_ms=200.0, max_ms=400.0)
        p = project_frame_latency(base, 2.3)  # ORB A55->A720
        assert p.median_ms == pytest.approx(100.0 / 2.3)
        assert p.p95_ms == pytest.approx(200.0 / 2.3)
        assert p.max_ms == pytest.approx(400.0 / 2.3)
        assert p.source == "projected"

    def test_speedup_one_is_measured_passthrough(self):
        base = LatencyDistribution(50.0, 80.0, 120.0)
        p = project_frame_latency(base, 1.0)
        assert (p.median_ms, p.p95_ms, p.max_ms) == (50.0, 80.0, 120.0)
        assert p.source == "measured"

    def test_max_optional(self):
        p = project_frame_latency(LatencyDistribution(10.0, 20.0), 2.0)
        assert p.max_ms is None

    def test_invalid_speedup_raises(self):
        with pytest.raises(ValueError):
            project_frame_latency(LatencyDistribution(1.0, 1.0), 0.0)


# ── R2b: solver-convergence projection ──
class TestR2bSolverConvergence:
    def _vins_a55(self):
        # Grounded VINS-Fusion re-run on real i.MX 95 A55 (2026-06-06).
        return SolverConvergenceAnchor(
            iters_median=4, iters_p95=11, iters_max=45,
            solve_ms=LatencyDistribution(median_ms=97.4, p95_ms=262.0, max_ms=1160.0))

    def test_iters_are_hardware_invariant(self):
        a = self._vins_a55()
        p = project_solver_convergence(a, 1.7)  # VINS A55->A720
        # iters do NOT scale with the per-core ratio (deterministic float math).
        assert (p.iters_median, p.iters_p95, p.iters_max) == (4, 11, 45)

    def test_solve_time_scales_by_ratio(self):
        a = self._vins_a55()
        p = project_solver_convergence(a, 1.7)
        # Only per-iteration cost moves -> solve time divides by the ratio.
        assert p.solve_median_ms == pytest.approx(97.4 / 1.7)   # ~57 ms
        assert p.solve_p95_ms == pytest.approx(262.0 / 1.7)     # ~154 ms (fat tail)
        assert p.solve_max_ms == pytest.approx(1160.0 / 1.7)    # ~682 ms
        assert p.source == "projected"

    def test_invalid_speedup_raises(self):
        with pytest.raises(ValueError):
            project_solver_convergence(self._vins_a55(), -1.0)
