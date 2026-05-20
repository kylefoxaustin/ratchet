"""Calibration provenance — how much should a projected number be trusted?

A CalibrationSource encodes (method, reference, confidence) so surfaces can
render appropriate confidence banners. Canonical tiers carry measured/vendor
calibration; custom tiers carry 'default' with low confidence.
"""
from dataclasses import dataclass
from typing import Literal

CalibrationMethod = Literal[
    "measured",       # Calibrated against real silicon measurements
    "interpolated",   # Calibrated by interpolating between measured tiers
    "vendor_spec",    # Derived from vendor-published specs only
    "default",        # Engine defaults; not calibrated for the specific silicon
]

CalibrationConfidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class CalibrationSource:
    """Provenance metadata for a Hardware tier's calibration constants."""

    method: CalibrationMethod
    reference: str
    confidence: CalibrationConfidence

    def __post_init__(self):
        # Light validation: default-method tiers must have low confidence.
        if self.method == "default" and self.confidence != "low":
            raise ValueError(
                "CalibrationSource: method='default' requires confidence='low'. "
                "Calibration defaults are not calibrated."
            )
