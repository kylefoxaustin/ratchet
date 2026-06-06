"""Calibration provenance — how much should a projected number be trusted?

A CalibrationSource encodes (method, reference, confidence) so surfaces can
render appropriate confidence banners. Canonical tiers carry measured/vendor
calibration; custom tiers carry 'default' with low confidence.
"""
from dataclasses import dataclass
from typing import Literal

CalibrationMethod = Literal[
    "measured",        # Calibrated against real silicon measurements
    "interpolated",    # Calibrated by interpolating between measured tiers
    "vendor_spec",     # Derived from vendor-published specs only
    "default",         # Engine defaults; not calibrated for the specific silicon
    # Added v0.3.2 (ADR 020): the badge vocabulary (anchors/loader.py
    # BADGE_FOR_SOURCE) and the drone-sizer spec (§9 / amendment 3) already use
    # these words; CalibrationMethod could not express them, so a per-metric
    # PerceptionAnchor source could not carry the label its own docstring intends.
    "projected",            # Computed from a model/ratio, not measured (e.g. A720 IPC ratio)
    "derived_from_measured",  # Transformed from a measurement (e.g. per-camera offload scaling)
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
