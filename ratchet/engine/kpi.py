"""KPI primitives — pass/fail constraints with margin.

Each KPI is a named, tested constraint. KPIs come in three scopes:

  1. Per-subsystem KPIs    — does this block fit on its target engine?
  2. Cross-cutting KPIs    — shared resources (memory BW, latency budgets)
  3. Chip-wide KPIs        — is the whole chip viable for the mission?

This module provides the KpiResult dataclass, the ``evaluate_budget`` helper
(used to build a KpiResult from a required-vs-budget comparison), and the
generic NPU/CPU/memory KPI evaluators that work on any list of
``SubsystemDemand``.

Domain-specific KPIs (ISP line-rate fit, DSP cycle budget, glass-to-glass
latency, perception inference deadline, radar-to-command chain) are owned by
consuming sites — they typically import this module's helpers and add their
own KPI functions on top.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .demand import SubsystemDemand


@dataclass
class KpiResult:
    name: str
    scope: str                    # subsystem | cross_cutting | chip
    target: str                   # which subsystem / engine
    metric: str
    required: float
    budget: float
    units: str
    status: str                   # PASS | FAIL | WARN
    margin: float                 # budget - required (positive is good)
    margin_pct: float
    notes: list[str] = field(default_factory=list)

    @property
    def emoji(self) -> str:
        return {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}.get(self.status, "·")


def evaluate_budget(
    name: str,
    scope: str,
    target: str,
    metric: str,
    required: float,
    budget: float,
    units: str,
    warn_at_pct: float = 90.0,
    notes: Optional[list[str]] = None,
) -> KpiResult:
    """Build a KpiResult from a required-vs-budget comparison.

    Status logic:
      - budget <= 0: PASS if required <= 0, else FAIL
      - required > budget: FAIL
      - required > budget × warn_at_pct/100: WARN
      - else: PASS
    """
    margin = budget - required
    margin_pct = (margin / budget * 100.0) if budget > 0 else 0.0
    if budget <= 0:
        status = "PASS" if required <= 0 else "FAIL"
    elif required > budget:
        status = "FAIL"
    elif required > budget * (warn_at_pct / 100.0):
        status = "WARN"
    else:
        status = "PASS"
    return KpiResult(
        name=name, scope=scope, target=target, metric=metric,
        required=required, budget=budget, units=units,
        status=status, margin=margin, margin_pct=margin_pct,
        notes=notes or [],
    )


# ──────────────────────────────────────────────────────────────────────
# Generic per-engine KPIs
# ──────────────────────────────────────────────────────────────────────

def npu_kpis(profile: dict, demands: list[SubsystemDemand]) -> list[KpiResult]:
    """One per NPU-resident subsystem (does it fit?) plus a sum-of-NPU check."""
    npu = profile["npu"]
    eff_t = npu["tops_bf16"] * npu.get("efficiency_factor", 0.55)
    headroom = profile.get("headroom_pct", {}).get("npu", 25)
    available = eff_t * (1.0 - headroom / 100.0)

    results: list[KpiResult] = []
    npu_demands = [d for d in demands if d.target_engine == "npu"]
    for d in npu_demands:
        if d.tops_required <= 0:
            continue
        results.append(evaluate_budget(
            name=f"{d.name}_fits_in_npu",
            scope="subsystem", target=d.name,
            metric="TOPS",
            required=d.tops_required,
            budget=available,
            units="TOPS",
            notes=d.notes,
        ))
    # Sum-of-NPU check: do all NPU workloads together fit?
    total = sum(d.tops_required for d in npu_demands)
    results.append(evaluate_budget(
        name="npu_concurrent_workload",
        scope="cross_cutting", target="npu",
        metric="aggregate_TOPS",
        required=total,
        budget=available,
        units="TOPS",
        notes=["Sum of all NPU-bound workloads running concurrently"],
    ))
    return results


def cpu_kpis(profile: dict, demands: list[SubsystemDemand]) -> list[KpiResult]:
    """Total effective-cores demand vs available across the application complex."""
    cpu = profile["cpu"]
    eff_cores = cpu["cores"] * cpu.get("efficiency_factor", 0.7)
    headroom = profile.get("headroom_pct", {}).get("cpu", 30)
    available = eff_cores * (1.0 - headroom / 100.0)
    total = sum(d.cpu_cores_required for d in demands if d.target_engine == "cpu")
    return [evaluate_budget(
        name="cpu_fits",
        scope="cross_cutting", target="cpu",
        metric="effective_cores",
        required=total,
        budget=available,
        units="cores",
    )]


def vpu_pixel_rate_kpi(
    profile_vpu: dict,
    total_mpix_per_sec: float,
    headroom_pct: float = 20.0,
    notes: Optional[list[str]] = None,
) -> KpiResult:
    """Generic VPU pixel-rate budget check.

    Sites compute the total Mpix/s demand (sum of per-stream resolution × fps
    across all encode streams) and pass it in. The VPU profile block must
    expose ``h265_max_mpix_per_sec``.
    """
    available = profile_vpu["h265_max_mpix_per_sec"] * (1.0 - headroom_pct / 100.0)
    return evaluate_budget(
        name="vpu_pixel_rate",
        scope="subsystem", target="encode",
        metric="Mpix_per_sec",
        required=total_mpix_per_sec,
        budget=available,
        units="Mpix/s",
        notes=notes or [],
    )


def memory_bw_kpi(profile: dict, demands: list[SubsystemDemand]) -> KpiResult:
    """Sum of memory bandwidth demand across ALL subsystems vs. usable BW."""
    mem = profile["memory"]
    eff = mem.get("controller_efficiency", 0.75)
    refresh = mem.get("refresh_overhead_pct", 5) / 100.0
    headroom = profile.get("headroom_pct", {}).get("memory_bw", 25)
    available = mem["bw_gbps"] * eff * (1.0 - refresh) * (1.0 - headroom / 100.0)
    total = sum(d.memory_bw_gbps for d in demands)
    return evaluate_budget(
        name="memory_bw_total",
        scope="chip", target="memory",
        metric="aggregate_BW",
        required=total,
        budget=available,
        units="GB/s",
        notes=["Sum across ALL subsystems concurrently. The most common chip-wide constraint."],
    )


def memory_capacity_kpi(profile: dict, demands: list[SubsystemDemand]) -> KpiResult:
    """Sum of resident memory across all subsystems vs. configured capacity."""
    cap_gb = profile["memory"].get("capacity_gb_max", 16)
    total_mb = sum(d.memory_capacity_mb for d in demands)
    return evaluate_budget(
        name="memory_capacity",
        scope="chip", target="memory",
        metric="resident_set",
        required=total_mb,
        budget=cap_gb * 1024,
        units="MB",
    )


# ──────────────────────────────────────────────────────────────────────
# Roll-up
# ──────────────────────────────────────────────────────────────────────

def chip_summary(results: list[KpiResult]) -> dict:
    """One-line health number from a KPI result list."""
    n_total = len(results)
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_warn = sum(1 for r in results if r.status == "WARN")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    fails = [r for r in results if r.status == "FAIL"]
    return {
        "total": n_total,
        "pass": n_pass,
        "warn": n_warn,
        "fail": n_fail,
        "viable": n_fail == 0,
        "failures": [
            {
                "name": r.name,
                "metric": r.metric,
                "overage": -r.margin,
                "units": r.units,
            } for r in fails
        ],
    }
