# ratchet

Generic SoC sizing engine. Pure-Python primitives for what-if analysis of
edge-class application processors: sliders, KPIs, subsystem demand
calculators, instrumentation probes, and a Parquet workload-record schema.

Designed to be shared across multiple sizer sites:

- [`nightjar`](https://github.com/kylefoxaustin/nightjar) — drone software
  stack + edge-SoC sizer (rescue-bird use case)
- `keyhole` — video-only sizer (planned)
- `skippy` — agentic-AI sizer (planned)

ratchet contains only the framework. Each consuming site supplies its own
subsystem demand calculators, KPI definitions, and slider catalog.

## Install (dev)

```bash
git clone https://github.com/kylefoxaustin/ratchet.git
cd ratchet
pip install -e ".[dev,gpu]"
pytest
```

Consumer sites depend on ratchet via local pip install during development:

```bash
# from a sibling directory
pip install -e ../ratchet
```

## Layout

```
ratchet/
├── engine/      primitives: Slider, SubsystemDemand, KpiResult,
│                generic NPU/CPU/memory KPIs, llm_demand math
├── whatif/      one consumer of the engine: point/sweep/pareto runner
├── probes/      Parquet writer + per-op / GPU / NVENC / glass-to-glass probes
└── schemas/     WorkloadRecord dataclass + PyArrow schema
```

The `engine/` and `whatif/` split is deliberate: a sizer can use the engine
primitives for one-shot evaluation without going through the what-if
runner.

## Status

v0.1.0 — initial extraction from rescue-bird/nightjar v0.2.0. See
[`docs/decisions/`](docs/decisions) for ADRs covering the engine-level
design choices.
