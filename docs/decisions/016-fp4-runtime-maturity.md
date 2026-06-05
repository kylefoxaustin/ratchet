# ADR 016: FP4's compute win is runtime-conditional

**Status:** Accepted
**Date:** 2026-06-05

## Context

v0.2.5 added FP4 (NVFP4/MXFP4) as a first-class compute dtype, modeled with
keyhole's RTX 5090 anchor: on **vLLM 0.22** (FlashInfer/CUTLASS NVFP4 kernels),
single-stream NVFP4 is **2.24× BF16 decode, 3.59× BF16 prefill** — a clear win.
`SM120_BLACKWELL_CAPABILITY["nvfp4"] = tensor_native`, and the cross-class
compute floor reads the full `peak_tops_fp4`.

But keyhole measured the **same NVFP4 weights on the same 5090 via llama.cpp**
and got the inverse: NVFP4 was **~15–19% slower** than Q4_K_M and 27% larger
(`precision_5090_fp4_vs_int.json`). The reconciliation (`…_vllm_fp4_vs_int.json`):
*"The confound was the RUNTIME (kernel maturity), not the format."* The published
3× FP4 speedups are specific to mature runtimes (vLLM/TensorRT-LLM); an immature
FP4 runtime realizes no win, and FP4 decodes slightly slower (it is BW-bound by a
larger byte footprint — embeddings + LM-head stay BF16).

This matters for ratchet specifically: ratchet feeds **edge** sizers, and edge
LLM runtimes are frequently llama.cpp-class, not vLLM. Modeling FP4 as a flat
`tensor_native` win over-promises for exactly ratchet's consumers. Backing
breadth: 12 model pairs across 6 architectures (`precision_anchors_5090.json`).

## Decision

1. **Separate silicon capability from runtime realization.** `nvfp4` on SM120
   stays `tensor_native` — the cores *are* native (ADR 008: capability answers
   "does the silicon execute this," not "does my software stack realize it").
   The capability `reason` text is updated to state the win is runtime-conditional.

2. **New projection axis** `fp4_runtime_maturity: Literal["mature", "immature"]`,
   keyword-only, **default `"mature"`** (preserves v0.2.5 behavior; non-breaking).
   It is a projection/deployment parameter, **not** a `Hardware` field — runtime
   maturity is a software-stack property, not silicon (keeps Hardware immutable +
   silicon-only).

3. **Immature + FP4 model ⇒ model it as the memory-format equivalent.** The
   cross-class compute floor falls to the **bf16 floor** (no realized FP4 GEMM
   win); decode stays BW-bound by ~4-bit weight bytes. This is *exactly* how
   INT4_AWQ/GPTQ is already modeled (ADR 015 / AMENDMENT 1: weight-only INT4
   dequants to bf16, prefill on the bf16 floor). In effect: **immature-FP4 ≅
   INT4 weight-only.** Reuses existing machinery; adds no new floor math.

4. **`deployment_path_for_tier` gains the same axis.** FP4 + immature returns a
   new label `fp4_runtime_immature` (silicon-native cores present, runtime cannot
   realize the win → memory-format behavior) so surfaces can render the caveat.

## Why a separate axis (not `workload_kernel_source`)

ADR 008's INT8 split is precompiled-vs-fresh-compile: fresh-compile (vLLM
CUTLASS) is the *blocked* path, precompiled (TRT) is fast. FP4 is **orthogonal
and nearly inverted**: vLLM (fresh-compile) is the *mature/winning* path while
llama.cpp (precompiled-ish) is *immature/losing*. The two axes don't collapse,
so FP4 maturity is its own `Literal`, not a reuse of `WorkloadKernelSource`.

## Consequences

- v0.2.5 default projections are unchanged (`mature`).
- Edge consumers opt into `immature` for an honest no-FP4-win projection that
  matches the measured llama.cpp result (prefill loses its win; decode ≈ INT4).
- The anchor *numbers* are not vendored into ratchet here — only the
  runtime-conditionality is encoded. Vendoring the JSONs is a separate decision.
- Supersedes nothing; refines the FP4 modeling introduced in v0.2.5 (ADR 008/015).
