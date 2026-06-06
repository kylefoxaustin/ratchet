"""Canonical tier registry + Hardware dataclass + tier construction."""
from ratchet.tiers.cpu import CpuComplex
from ratchet.tiers.custom import SiliconClass, make_custom_tier
from ratchet.tiers.hardware import Hardware
from ratchet.tiers.memory_overlay import MEMORY_UPGRADE_OPTIONS, hw_with_memory
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
    NPU_LOW_LP5_32BIT,
    NPU_LOW_LP5_64BIT,
    NPU_LOW_LP5X,
    NPU_MID,
    RTX_5090_REFERENCE,
    TIERS,
)

__all__ = [
    "Hardware",
    "TIERS",
    "NPU_LOW_LP4", "NPU_LOW_LP5_32BIT", "NPU_LOW_LP5_64BIT", "NPU_LOW_LP5X",
    "IMX93_MEASURED", "IMX95_MEASURED", "NPU_MID", "NPU_HIGH", "RTX_5090_REFERENCE",
    "hw_with_memory", "MEMORY_UPGRADE_OPTIONS",
    "make_custom_tier", "SiliconClass",
    "CpuComplex", "PerceptionAnchor", "LatencyDistribution",
    "SolverConvergenceAnchor",
]
