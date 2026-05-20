# ratchet

Generic SoC sizing engine — the shared foundation for an edge-SoC sizing
ecosystem. Pure-Python primitives for what-if analysis of edge-class
application processors: a canonical NPU/GPU tier registry, a 4-level dtype
capability taxonomy, an LLM performance-projection API, an anchor-secrets
overlay for private silicon measurements, an LLM catalog schema, calibration
provenance, plus the carried-forward sliders, KPIs, subsystem demand
calculators, instrumentation probes, and Parquet workload-record schema.

Designed to be shared across multiple sizer sites:

- [`nightjar`](https://github.com/kylefoxaustin/nightjar) — drone software
  stack + edge-SoC sizer (rescue-bird use case)
- `personal-ai-assistant-sizer` (PAI) — LLM sizer
- `keyhole` / `keyhole-sizer` — video sizer
- `personal-ai-framework` (Skippy) — agentic-AI framework
- `drone-sizer` — planned

ratchet owns the canonical engine. Each consuming site composes its own visible
tier ladder from the registry and supplies its own model catalog, subsystem
demand calculators, KPI definitions, and slider catalog.

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
├── tiers/        Hardware dataclass + canonical TIERS registry +
│                 memory-upgrade overlay + custom-tier factory
├── precision/    4-level capability taxonomy + dtype dispatch +
│                 deployment-path classifier
├── projection/   LLM projection API (4-path cascade), result types,
│                 memory feasibility, workload-pattern overlay
├── anchors/      anchor-secrets loader + post-projection overlay for
│                 private silicon measurements (runtime, never in source)
├── catalog/      LLMModel schema + quant byte tables (content per-surface)
├── calibration/  CalibrationSource provenance + silicon-class defaults
├── engine/       primitives: Slider, SubsystemDemand, KpiResult, llm_demand
├── whatif/       one consumer of the engine: point/sweep/pareto runner
├── probes/       Parquet writer + per-op / GPU / NVENC / glass-to-glass probes
└── schemas/      WorkloadRecord dataclass + PyArrow schema
```

Surfaces import the public API from `ratchet` directly (not from submodules).
The `engine/` and `whatif/` split is deliberate: a sizer can use the engine
primitives for one-shot evaluation without going through the what-if runner.

## Usage

```python
from ratchet import project_llm, NPU_MID, Projected, WontFit, DtypeMismatch
from ratchet.catalog.reference import QWEN3_30B_A3B_MOE_Q4

result = project_llm(QWEN3_30B_A3B_MOE_Q4, NPU_MID, "rag_qa",
                     prompt_tokens=4800, decode_tokens=400)
match result:
    case Projected(decode_tok_s=t, source=s):
        print(f"{t} tok/s ({s})")
    case WontFit(required_gb=r, available_gb=a):
        print(f"Won't fit: {r:.1f} GB needed, {a:.1f} available")
    case DtypeMismatch(retargeting_hint=h):
        print(h)
```

## Status

v0.2.0 — engine-consolidation release: absorbs the canonical tier registry,
capability taxonomy, projection API, anchor-secrets system, and LLM catalog
schema that the ecosystem surfaces evolved independently. Backward compatible
with v0.1.0 (all prior imports still work). See [`docs/decisions/`](docs/decisions)
for the 15 ADRs covering engine-level design choices.
