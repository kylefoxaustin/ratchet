"""WhatifRunner — point / sweep / pareto over a site-supplied catalog.

The runner is one consumer of the engine. Sites instantiate it with their own
slider catalog, evaluate function, default workload factory, and profile
loader, then drive it from a CLI (the CLI itself stays site-side).

Three modes:

  - ``point(profile_name, overrides)``  — one evaluation
  - ``sweep(profile_name, slider_name, steps, base_overrides)`` — 1D sweep
  - ``pareto(profile_name, x, y, steps_x, steps_y, base_overrides)`` — 2D grid

All three return structured results suitable for either Markdown rendering or
JSON serialization.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Callable

from ratchet.engine.slider import (
    Slider,
    apply_sliders,
    default_values,
)
from ratchet.engine.kpi import KpiResult, chip_summary


# ──────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PointResult:
    profile: str
    overrides: dict[str, float]
    kpis: list[KpiResult]
    summary: dict


@dataclass
class SweepRow:
    value: float
    summary: dict
    kpis: list[KpiResult]


@dataclass
class SweepResult:
    profile: str
    slider: str
    base_overrides: dict[str, float]
    rows: list[SweepRow]


@dataclass
class ParetoResult:
    profile: str
    x_slider: str
    y_slider: str
    base_overrides: dict[str, float]
    xs: list[float]
    ys: list[float]
    cells: list[list[dict]]   # cells[y_idx][x_idx] = chip_summary dict


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────

class WhatifRunner:
    """Drives evaluations across slider configurations.

    Args:
        sliders: site-supplied catalog (dict[name, Slider]).
        evaluate_fn: function (profile, workload) → list[KpiResult] that the
            site provides. Wires together the site's demand calculators and
            KPI definitions.
        default_workload_factory: zero-arg callable returning a fresh
            workload dict (the site's DEFAULT_WORKLOAD, deep-copied each call).
        profile_loader: callable taking a profile name (string) and returning
            the parsed profile dict (typically loads YAML from a site directory).
    """

    def __init__(
        self,
        sliders: dict[str, Slider],
        evaluate_fn: Callable[[dict, dict], list[KpiResult]],
        default_workload_factory: Callable[[], dict],
        profile_loader: Callable[[str], dict],
    ) -> None:
        self.sliders = sliders
        self.evaluate_fn = evaluate_fn
        self.default_workload_factory = default_workload_factory
        self.profile_loader = profile_loader

    def build_state(
        self,
        profile_name: str,
        overrides: dict[str, float],
    ) -> tuple[dict, dict, dict]:
        """Load profile, build workload, apply slider overrides. Returns
        ``(profile, workload, values)``."""
        profile = self.profile_loader(profile_name)
        workload = copy.deepcopy(self.default_workload_factory())
        values = default_values(self.sliders)
        values.update(overrides)
        apply_sliders(self.sliders, profile, workload, values)
        return profile, workload, values

    def point(
        self,
        profile_name: str,
        overrides: dict[str, float],
    ) -> PointResult:
        profile, workload, _ = self.build_state(profile_name, overrides)
        results = self.evaluate_fn(profile, workload)
        return PointResult(
            profile=profile_name,
            overrides=dict(overrides),
            kpis=results,
            summary=chip_summary(results),
        )

    def sweep(
        self,
        profile_name: str,
        slider_name: str,
        steps: int,
        base_overrides: dict[str, float] | None = None,
    ) -> SweepResult:
        if slider_name not in self.sliders:
            raise KeyError(f"Unknown slider: {slider_name}")
        s = self.sliders[slider_name]
        steps = max(2, steps)
        span = s.max_val - s.min_val
        values = [s.min_val + (span * i / (steps - 1)) for i in range(steps)]
        base = dict(base_overrides or {})
        rows: list[SweepRow] = []
        for v in values:
            ovr = dict(base)
            ovr[slider_name] = v
            profile, workload, _ = self.build_state(profile_name, ovr)
            kpis = self.evaluate_fn(profile, workload)
            rows.append(SweepRow(value=v, summary=chip_summary(kpis), kpis=kpis))
        return SweepResult(
            profile=profile_name,
            slider=slider_name,
            base_overrides=base,
            rows=rows,
        )

    def pareto(
        self,
        profile_name: str,
        x: str,
        y: str,
        steps_x: int,
        steps_y: int,
        base_overrides: dict[str, float] | None = None,
    ) -> ParetoResult:
        for axis in (x, y):
            if axis not in self.sliders:
                raise KeyError(f"Unknown slider: {axis}")
        sx = self.sliders[x]
        sy = self.sliders[y]
        nx, ny = max(2, steps_x), max(2, steps_y)
        xs = [sx.min_val + ((sx.max_val - sx.min_val) * i / (nx - 1)) for i in range(nx)]
        ys = [sy.min_val + ((sy.max_val - sy.min_val) * j / (ny - 1)) for j in range(ny)]
        base = dict(base_overrides or {})
        cells: list[list[dict]] = []
        for yv in ys:
            row: list[dict] = []
            for xv in xs:
                ovr = dict(base)
                ovr[x] = xv
                ovr[y] = yv
                profile, workload, _ = self.build_state(profile_name, ovr)
                kpis = self.evaluate_fn(profile, workload)
                row.append(chip_summary(kpis))
            cells.append(row)
        return ParetoResult(
            profile=profile_name,
            x_slider=x,
            y_slider=y,
            base_overrides=base,
            xs=xs,
            ys=ys,
            cells=cells,
        )
