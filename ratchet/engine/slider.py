"""Slider primitive — a named, typed, bounded assumption knob.

Sliders are the input layer of the sizing engine. A consuming site (drone,
video, agentic AI) defines its own slider catalog as a ``dict[str, Slider]``
and hands it to the what-if runner.

Each slider carries an ``apply`` callback that mutates a (profile, workload)
pair. The engine doesn't care what those dicts look like — that's the site's
business — it just calls the callback. Sites typically use the dotted-path
helpers ``_set_path`` and ``_scale_path`` exported here.

Sliders fall into four conventional categories: ``capability`` (what the
chip can do), ``workload`` (what the mission demands), ``operating`` (deployment
choices), and ``headroom`` (safety margins). The engine doesn't enforce these
strings — sites are free to pick their own.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Any, List


@dataclass
class Slider:
    """A single assumption knob."""
    name: str
    description: str
    category: str                           # capability | workload | operating | headroom (convention)
    units: str
    default: float
    min_val: float
    max_val: float
    step: float = 1.0
    affects: List[str] = field(default_factory=list)     # KPIs / subsystems impacted
    # How to apply this slider to (profile_dict, workload_dict).
    # If None, the slider is purely informational — caller does the mapping.
    apply: Optional[Callable[[dict, dict, float], None]] = None

    def clamp(self, v: float) -> float:
        return max(self.min_val, min(self.max_val, v))


# ──────────────────────────────────────────────────────────────────────
# Helpers — small functions that mutate nested dicts via dotted paths.
# Sites use these inside slider .apply lambdas.
# ──────────────────────────────────────────────────────────────────────

def _set_path(d: dict, path: str, value: Any) -> None:
    """Set a dotted path inside a nested dict."""
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _scale_path(d: dict, path: str, factor: float) -> None:
    """Multiply a dotted-path value by factor."""
    keys = path.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = d[keys[-1]] * factor


# ──────────────────────────────────────────────────────────────────────
# Catalog operations
# ──────────────────────────────────────────────────────────────────────

def slider_categories(sliders: dict[str, Slider]) -> dict[str, list[Slider]]:
    """Group a slider catalog by category for UI rendering."""
    out: dict[str, list[Slider]] = {}
    for s in sliders.values():
        out.setdefault(s.category, []).append(s)
    return out


def default_values(sliders: dict[str, Slider]) -> dict[str, float]:
    """Pull the default value for every slider in the catalog."""
    return {name: s.default for name, s in sliders.items()}


def apply_sliders(
    sliders: dict[str, Slider],
    profile: dict,
    workload: dict,
    values: dict[str, float],
) -> None:
    """Apply slider overrides in-place to (profile, workload).

    ``values`` is a mapping of slider name → desired value. Unknown names are
    ignored. Values are clamped to each slider's range. Sliders whose
    ``apply`` is None are skipped (informational sliders).
    """
    for name, v in values.items():
        s = sliders.get(name)
        if s is None or s.apply is None:
            continue
        s.apply(profile, workload, s.clamp(v))
