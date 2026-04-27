"""ratchet — generic SoC sizing engine.

Top-level package. Submodules:

- ``ratchet.engine``  — primitives: Slider, SubsystemDemand, KpiResult,
  generic NPU/CPU/memory KPIs, llm_demand math.
- ``ratchet.whatif``  — point/sweep/pareto runner that consumes the engine.
- ``ratchet.probes``  — Parquet writer + per-op / GPU / NVENC / g2g probes.
- ``ratchet.schemas`` — WorkloadRecord dataclass + PyArrow schema.

Consuming sites (nightjar, keyhole, skippy) supply their own subsystem
demand calculators, KPI definitions, and slider catalogs.
"""

__version__ = "0.1.0"
