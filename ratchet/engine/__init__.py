"""Engine primitives — usable without going through the what-if runner."""
from .slider import (
    Slider,
    apply_sliders,
    default_values,
    slider_categories,
)
from .demand import (
    SubsystemDemand,
    eff_tops,
    llm_demand,
)
from .kpi import (
    KpiResult,
    evaluate_budget,
    npu_kpis,
    cpu_kpis,
    vpu_pixel_rate_kpi,
    memory_bw_kpi,
    memory_capacity_kpi,
    chip_summary,
)

__all__ = [
    "Slider",
    "apply_sliders",
    "default_values",
    "slider_categories",
    "SubsystemDemand",
    "eff_tops",
    "llm_demand",
    "KpiResult",
    "evaluate_budget",
    "npu_kpis",
    "cpu_kpis",
    "vpu_pixel_rate_kpi",
    "memory_bw_kpi",
    "memory_capacity_kpi",
    "chip_summary",
]
