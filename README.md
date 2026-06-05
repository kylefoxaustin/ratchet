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
- `personal-ai-assistant-sizer` (PAI sizer) — LLM-only edge sizer
- `keyhole` / `keyhole-sizer` — video sizer
- `personal-ai-framework` (Skippy) — agentic-AI framework
- `drone-sizer` — planned

ratchet owns the canonical engine. Each consuming site composes its own visible
tier ladder from the registry and supplies its own model catalog, subsystem
demand calculators, KPI definitions, and slider catalog. Surfaces pin
`ratchet>=0.2.0,<0.3.0`; engine additions follow a rule-of-three (no new
primitive until ≥2 surfaces demonstrate the need).

## Canonical tiers

`TIERS` is the canonical registry (8 named tiers). Surfaces select a visible
subset; they never define new `Hardware` except via `make_custom_tier()` or the
`hw_with_memory()` overlay.

| Tier                          | Class     | Notes                                  |
|-------------------------------|-----------|----------------------------------------|
| NPU Low-LP4                   | edge      | LPDDR4X                                |
| NPU Low-LP5-32bit             | edge      | LPDDR5, 32-bit bus                     |
| NPU Low-LP5-64bit             | edge      | LPDDR5, 64-bit bus                     |
| NPU Low-LP5X                  | edge      | LPDDR5X                                |
| NPU i.MX 95 (ground truth)    | edge      | measured silicon (anchor attachment)   |
| NPU Mid                       | edge      | LPDDR5X, mid-class NPU                  |
| NPU High                      | edge      | LPDDR5X, high-class NPU                 |
| RTX 5090 (reference, measured)| reference | Blackwell sm_120; FP4 reference silicon |

## Install (dev)

```bash
git clone https://github.com/kylefoxaustin/ratchet.git
cd ratchet
pip install -e ".[dev,gpu]"
pytest          # 230 tests
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

### Precision: capability vs. runtime realization

The capability taxonomy answers *"can this silicon execute this dtype, and how
well?"* (`tensor_native` / `tensor_compat` / `cuda_core` / `unsupported`),
separately from whether a given **runtime** can realize the win. FP4 is the
clearest case: native on Blackwell sm_120, but its compute win is realized only
on a mature runtime (vLLM ≥ 0.22 / TensorRT-LLM) — on an immature one
(llama.cpp today) FP4 behaves like INT4 weight-only (ADR 016).

```python
from ratchet import RTX_5090_REFERENCE, deployment_path_for_tier

deployment_path_for_tier(RTX_5090_REFERENCE, "nvfp4", "fresh_compile")
# -> "native_fast"
deployment_path_for_tier(RTX_5090_REFERENCE, "nvfp4", "fresh_compile", "immature")
# -> "fp4_runtime_immature"   (silicon is native, runtime can't realize the win)

# project_llm(model, hw, workload, fp4_runtime_maturity="immature")
#   models an FP4 model as INT4 weight-only (prefill falls to the bf16 floor).
#   Default is "mature" — non-breaking.
```

## Status

**v0.2.6** — current. Engine consumed by pinned versions across surfaces;
backward compatible with v0.1.0 (all prior imports still work). See
[`docs/decisions/`](docs/decisions) for the **16 ADRs** covering engine-level
design choices, and [`docs/design/`](docs/design) for the design specs.

Version history:

- **v0.2.0** — engine-consolidation release: absorbs the canonical tier
  registry, capability taxonomy, projection API, anchor-secrets system, and LLM
  catalog schema the ecosystem surfaces had evolved independently.
- **v0.2.1** — anchor-loader correction (Amendment 3): the real `npu_anchors.py`
  contract is canonical, superseding the design-doc §9 sketch.
- **v0.2.2** — tier registry corrected to production silicon specs.
- **v0.2.3** — BW-scale the private anchor overlay on memory-upgrade clones.
- **v0.2.4** — correct NPU i.MX 95 TDP 8 → 10 W (Amendment 6).
- **v0.2.5** — NVFP4/MXFP4 (FP4) added as a first-class compute dtype, distinct
  from weight-only INT4 (a memory format that dequantizes to bf16).
- **v0.2.6** — FP4's compute win is runtime-conditional (ADR 016): new
  `fp4_runtime_maturity` projection axis, defaulting to `"mature"`.
