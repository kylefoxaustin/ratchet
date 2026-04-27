"""What-if runner — point / sweep / pareto over a site-supplied catalog.

Consumes the engine primitives. A site (nightjar/keyhole/skippy) instantiates
``WhatifRunner`` with its own slider catalog, evaluate function, default
workload factory, and profile loader, then drives the runner from a CLI.
"""
from .runner import (
    WhatifRunner,
    PointResult,
    SweepResult,
    SweepRow,
    ParetoResult,
)

__all__ = [
    "WhatifRunner",
    "PointResult",
    "SweepResult",
    "SweepRow",
    "ParetoResult",
]
