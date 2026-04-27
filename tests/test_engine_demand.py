"""Framework-level tests for ratchet.engine.demand.

Exercises SubsystemDemand, eff_tops, and llm_demand with synthetic profiles
and workloads — no drone/video/AI vocabulary in fixtures.
"""

from __future__ import annotations

import pytest

from ratchet.engine.demand import SubsystemDemand, eff_tops, llm_demand


# ──────────────────────────────────────────────────────────────────────
# SubsystemDemand dataclass
# ──────────────────────────────────────────────────────────────────────

class TestSubsystemDemand:
    def test_defaults_zeroed(self):
        d = SubsystemDemand(name="x", target_engine="npu")
        assert d.tops_required == 0.0
        assert d.cpu_cores_required == 0.0
        assert d.memory_bw_gbps == 0.0
        assert d.memory_capacity_mb == 0.0
        assert d.latency_ms_p99 == 0.0
        assert d.notes == []

    def test_field_assignment(self):
        d = SubsystemDemand(
            name="example", target_engine="npu",
            tops_required=5.0, memory_bw_gbps=12.0,
            notes=["explainer"],
        )
        assert d.name == "example"
        assert d.tops_required == 5.0
        assert d.notes == ["explainer"]


# ──────────────────────────────────────────────────────────────────────
# eff_tops
# ──────────────────────────────────────────────────────────────────────

class TestEffTops:
    def test_uses_requested_precision_when_present(self):
        npu = {"tops_bf16": 100, "tops_fp16": 100, "tops_int8": 200,
               "efficiency_factor": 0.5}
        assert eff_tops(npu, "bf16") == 50.0
        assert eff_tops(npu, "int8") == 100.0

    def test_falls_back_to_bf16_for_unknown_precision(self):
        npu = {"tops_bf16": 80, "efficiency_factor": 0.5}
        # No tops_fp32 in profile — should fall back to tops_bf16
        assert eff_tops(npu, "fp32") == 40.0

    def test_default_efficiency_when_missing(self):
        # Default efficiency factor is 0.55 (per ADR 005)
        npu = {"tops_bf16": 100}
        assert eff_tops(npu, "bf16") == pytest.approx(55.0)

    def test_zero_when_no_tops_in_profile(self):
        npu = {"efficiency_factor": 0.55}
        assert eff_tops(npu, "bf16") == 0.0

    def test_zero_efficiency_returns_zero(self):
        npu = {"tops_bf16": 100, "efficiency_factor": 0}
        assert eff_tops(npu, "bf16") == 0.0


# ──────────────────────────────────────────────────────────────────────
# llm_demand — the headline ADR 004 case
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_profile():
    """Minimal profile dict with just the bits llm_demand looks at."""
    return {"npu": {"tops_bf16": 100, "efficiency_factor": 0.55}}


class TestLlmDemandInactive:
    def test_inactive_returns_zero_demand(self, minimal_profile):
        workload = {"llm": {"active": False, "params_b": 7,
                            "tokens_per_sec": 20, "precision": "int8"}}
        d = llm_demand(minimal_profile, workload)
        assert d.tops_required == 0
        assert d.memory_bw_gbps == 0
        assert d.memory_capacity_mb == 0


class TestLlmDemandActive:
    def test_7b_int8_at_20_tps_is_140_gbps(self, minimal_profile):
        """ADR 004 headline: 7B params × 1 byte/param × 20 tok/s = 140 GB/s."""
        workload = {"llm": {"active": True, "params_b": 7,
                            "tokens_per_sec": 20, "precision": "int8"}}
        d = llm_demand(minimal_profile, workload)
        assert d.memory_bw_gbps == pytest.approx(140.0, rel=0.01)

    def test_int4_halves_bandwidth_vs_int8(self, minimal_profile):
        wl_int8 = {"llm": {"active": True, "params_b": 7,
                           "tokens_per_sec": 20, "precision": "int8"}}
        wl_int4 = {"llm": {"active": True, "params_b": 7,
                           "tokens_per_sec": 20, "precision": "int4"}}
        d8 = llm_demand(minimal_profile, wl_int8)
        d4 = llm_demand(minimal_profile, wl_int4)
        assert d4.memory_bw_gbps == pytest.approx(d8.memory_bw_gbps / 2, rel=0.01)

    def test_bf16_doubles_bandwidth_vs_int8(self, minimal_profile):
        wl_int8 = {"llm": {"active": True, "params_b": 7,
                           "tokens_per_sec": 20, "precision": "int8"}}
        wl_bf16 = {"llm": {"active": True, "params_b": 7,
                           "tokens_per_sec": 20, "precision": "bf16"}}
        d8 = llm_demand(minimal_profile, wl_int8)
        dbf = llm_demand(minimal_profile, wl_bf16)
        assert dbf.memory_bw_gbps == pytest.approx(d8.memory_bw_gbps * 2, rel=0.01)

    def test_compute_is_small_compared_to_bandwidth(self, minimal_profile):
        """ADR 004: 7B at 20 tps is ~0.28 TOPS — bandwidth dominates."""
        workload = {"llm": {"active": True, "params_b": 7,
                            "tokens_per_sec": 20, "precision": "int8"}}
        d = llm_demand(minimal_profile, workload)
        assert d.tops_required < 1.0

    def test_capacity_tracks_resident_weights(self, minimal_profile):
        workload = {"llm": {"active": True, "params_b": 7,
                            "tokens_per_sec": 20, "precision": "int8"}}
        d = llm_demand(minimal_profile, workload)
        # 7B params × 1 byte/param = 7000 MB
        assert d.memory_capacity_mb == pytest.approx(7000, rel=0.01)

    def test_target_engine_is_npu(self, minimal_profile):
        workload = {"llm": {"active": True, "params_b": 3,
                            "tokens_per_sec": 20, "precision": "int8"}}
        d = llm_demand(minimal_profile, workload)
        assert d.target_engine == "npu"

    def test_throughput_drives_inverse_latency(self, minimal_profile):
        wl_fast = {"llm": {"active": True, "params_b": 3,
                           "tokens_per_sec": 100, "precision": "int8"}}
        wl_slow = {"llm": {"active": True, "params_b": 3,
                           "tokens_per_sec": 10, "precision": "int8"}}
        d_fast = llm_demand(minimal_profile, wl_fast)
        d_slow = llm_demand(minimal_profile, wl_slow)
        assert d_fast.latency_ms_p99 < d_slow.latency_ms_p99
