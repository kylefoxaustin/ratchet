"""Framework-level tests for ratchet.engine.kpi.

Exercises KpiResult, evaluate_budget, the generic per-engine KPI evaluators
(npu/cpu/memory/vpu_pixel_rate), and chip_summary using synthetic
SubsystemDemand inputs and synthetic profile dicts.
"""

from __future__ import annotations

import pytest

from ratchet.engine.demand import SubsystemDemand
from ratchet.engine.kpi import (
    KpiResult,
    evaluate_budget,
    npu_kpis,
    cpu_kpis,
    vpu_pixel_rate_kpi,
    memory_bw_kpi,
    memory_capacity_kpi,
    chip_summary,
)


# ──────────────────────────────────────────────────────────────────────
# KpiResult dataclass
# ──────────────────────────────────────────────────────────────────────

class TestKpiResult:
    def test_emoji_for_each_status(self):
        r = KpiResult(name="x", scope="chip", target="m", metric="z",
                      required=1, budget=10, units="u",
                      status="PASS", margin=9, margin_pct=90)
        assert r.emoji == "✅"
        r.status = "FAIL"
        assert r.emoji == "❌"
        r.status = "WARN"
        assert r.emoji == "⚠️"
        r.status = "UNKNOWN"
        assert r.emoji == "·"


# ──────────────────────────────────────────────────────────────────────
# evaluate_budget — boundary conditions
# ──────────────────────────────────────────────────────────────────────

class TestEvaluateBudget:
    def test_required_below_warn_threshold_passes(self):
        r = evaluate_budget(name="t", scope="chip", target="x", metric="m",
                            required=50, budget=100, units="u")
        assert r.status == "PASS"
        assert r.margin == 50
        assert r.margin_pct == pytest.approx(50.0)

    def test_required_in_warn_band_warns(self):
        # 95% of budget is above the default 90% warn threshold
        r = evaluate_budget(name="t", scope="chip", target="x", metric="m",
                            required=95, budget=100, units="u")
        assert r.status == "WARN"

    def test_required_over_budget_fails(self):
        r = evaluate_budget(name="t", scope="chip", target="x", metric="m",
                            required=120, budget=100, units="u")
        assert r.status == "FAIL"
        assert r.margin == -20

    def test_zero_budget_with_zero_demand_passes(self):
        r = evaluate_budget(name="t", scope="chip", target="x", metric="m",
                            required=0, budget=0, units="u")
        assert r.status == "PASS"

    def test_zero_budget_with_demand_fails(self):
        r = evaluate_budget(name="t", scope="chip", target="x", metric="m",
                            required=1, budget=0, units="u")
        assert r.status == "FAIL"

    def test_warn_threshold_configurable(self):
        # With warn_at_pct=50, 60% utilization should warn
        r = evaluate_budget(name="t", scope="chip", target="x", metric="m",
                            required=60, budget=100, units="u",
                            warn_at_pct=50)
        assert r.status == "WARN"

    def test_notes_passed_through(self):
        r = evaluate_budget(name="t", scope="chip", target="x", metric="m",
                            required=1, budget=10, units="u",
                            notes=["explainer one", "explainer two"])
        assert r.notes == ["explainer one", "explainer two"]


# ──────────────────────────────────────────────────────────────────────
# Generic per-engine KPIs against synthetic demand lists
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def npu_profile():
    return {
        "npu": {"tops_bf16": 100, "efficiency_factor": 0.5},
        "headroom_pct": {"npu": 0},  # no headroom for cleaner math
    }


class TestNpuKpis:
    def test_emits_one_per_npu_demand_plus_aggregate(self, npu_profile):
        demands = [
            SubsystemDemand(name="alpha", target_engine="npu", tops_required=10),
            SubsystemDemand(name="beta", target_engine="npu", tops_required=15),
            # Non-NPU demands ignored
            SubsystemDemand(name="other", target_engine="cpu", tops_required=99),
        ]
        results = npu_kpis(npu_profile, demands)
        names = {r.name for r in results}
        assert "alpha_fits_in_npu" in names
        assert "beta_fits_in_npu" in names
        assert "npu_concurrent_workload" in names
        assert len(results) == 3

    def test_skips_npu_demands_with_zero_tops(self, npu_profile):
        demands = [
            SubsystemDemand(name="alpha", target_engine="npu", tops_required=0),
            SubsystemDemand(name="beta", target_engine="npu", tops_required=10),
        ]
        results = npu_kpis(npu_profile, demands)
        # alpha skipped; only beta + aggregate
        names = {r.name for r in results}
        assert "alpha_fits_in_npu" not in names
        assert "beta_fits_in_npu" in names

    def test_aggregate_sums_all_npu_demands(self, npu_profile):
        demands = [
            SubsystemDemand(name="a", target_engine="npu", tops_required=10),
            SubsystemDemand(name="b", target_engine="npu", tops_required=20),
            SubsystemDemand(name="c", target_engine="npu", tops_required=15),
        ]
        results = npu_kpis(npu_profile, demands)
        agg = next(r for r in results if r.name == "npu_concurrent_workload")
        assert agg.required == 45

    def test_overage_produces_fail(self, npu_profile):
        # Effective TOPS = 100 × 0.5 = 50; demand 80 should fail
        demands = [SubsystemDemand(name="big", target_engine="npu", tops_required=80)]
        results = npu_kpis(npu_profile, demands)
        big = next(r for r in results if r.name == "big_fits_in_npu")
        assert big.status == "FAIL"


class TestCpuKpis:
    def test_sums_only_cpu_demands(self):
        profile = {
            "cpu": {"cores": 8, "efficiency_factor": 1.0},
            "headroom_pct": {"cpu": 0},
        }
        demands = [
            SubsystemDemand(name="x", target_engine="cpu", cpu_cores_required=2.0),
            SubsystemDemand(name="y", target_engine="cpu", cpu_cores_required=1.5),
            SubsystemDemand(name="z", target_engine="npu", cpu_cores_required=99),
        ]
        results = cpu_kpis(profile, demands)
        assert len(results) == 1
        assert results[0].required == pytest.approx(3.5)
        assert results[0].budget == pytest.approx(8.0)
        assert results[0].status == "PASS"


class TestVpuPixelRateKpi:
    def test_under_budget_passes(self):
        vpu = {"h265_max_mpix_per_sec": 500}
        r = vpu_pixel_rate_kpi(vpu, total_mpix_per_sec=200, headroom_pct=0)
        assert r.status == "PASS"
        assert r.budget == 500

    def test_headroom_reduces_budget(self):
        vpu = {"h265_max_mpix_per_sec": 500}
        r = vpu_pixel_rate_kpi(vpu, total_mpix_per_sec=200, headroom_pct=20)
        # Budget should be 500 × (1 - 0.2) = 400
        assert r.budget == pytest.approx(400)

    def test_over_budget_fails(self):
        vpu = {"h265_max_mpix_per_sec": 100}
        r = vpu_pixel_rate_kpi(vpu, total_mpix_per_sec=200, headroom_pct=0)
        assert r.status == "FAIL"


class TestMemoryBwKpi:
    def test_sums_across_all_demands_regardless_of_engine(self):
        profile = {
            "memory": {"bw_gbps": 100, "controller_efficiency": 1.0,
                       "refresh_overhead_pct": 0},
            "headroom_pct": {"memory_bw": 0},
        }
        demands = [
            SubsystemDemand(name="a", target_engine="npu", memory_bw_gbps=10),
            SubsystemDemand(name="b", target_engine="cpu", memory_bw_gbps=15),
            SubsystemDemand(name="c", target_engine="vpu", memory_bw_gbps=20),
        ]
        r = memory_bw_kpi(profile, demands)
        assert r.required == 45
        assert r.budget == 100
        assert r.status == "PASS"

    def test_controller_efficiency_reduces_budget(self):
        profile = {
            "memory": {"bw_gbps": 100, "controller_efficiency": 0.75,
                       "refresh_overhead_pct": 0},
            "headroom_pct": {"memory_bw": 0},
        }
        r = memory_bw_kpi(profile, [])
        assert r.budget == pytest.approx(75)


class TestMemoryCapacityKpi:
    def test_sums_resident_set_across_demands(self):
        profile = {"memory": {"capacity_gb_max": 1}}
        demands = [
            SubsystemDemand(name="a", target_engine="npu", memory_capacity_mb=300),
            SubsystemDemand(name="b", target_engine="cpu", memory_capacity_mb=500),
        ]
        r = memory_capacity_kpi(profile, demands)
        assert r.required == 800
        assert r.budget == 1024
        assert r.status == "PASS"


# ──────────────────────────────────────────────────────────────────────
# chip_summary
# ──────────────────────────────────────────────────────────────────────

def _result(status: str, name: str = "x", margin: float = 0,
            metric: str = "m", units: str = "u") -> KpiResult:
    return KpiResult(name=name, scope="chip", target="t", metric=metric,
                     required=1, budget=2, units=units,
                     status=status, margin=margin, margin_pct=0)


class TestChipSummary:
    def test_counts_match_total(self):
        results = [_result("PASS"), _result("PASS"), _result("WARN"),
                   _result("FAIL")]
        s = chip_summary(results)
        assert s["pass"] + s["warn"] + s["fail"] == s["total"]
        assert s["total"] == 4

    def test_viable_when_no_failures(self):
        results = [_result("PASS"), _result("WARN")]
        assert chip_summary(results)["viable"] is True

    def test_not_viable_when_any_failure(self):
        results = [_result("PASS"), _result("FAIL")]
        assert chip_summary(results)["viable"] is False

    def test_failures_list_carries_overage(self):
        results = [_result("FAIL", name="big_fail", margin=-15, units="GB/s")]
        s = chip_summary(results)
        assert len(s["failures"]) == 1
        assert s["failures"][0]["name"] == "big_fail"
        assert s["failures"][0]["overage"] == 15
        assert s["failures"][0]["units"] == "GB/s"

    def test_empty_input(self):
        s = chip_summary([])
        assert s["total"] == 0
        assert s["viable"] is True
