# ADR 008: 4-level capability taxonomy

**Status:** Accepted
**Date:** 2026-05-19

## Context

"Can this tier run this dtype?" has more than a yes/no answer. Consumer
Blackwell (SM120) has INT8 tensor cores reachable via sm80 IMMA binary
compatibility — so pre-compiled TRT engines run fast, but vLLM's CUTLASS
fresh-compile path lacks SM120 templates and is blocked. A binary
supported/unsupported flag cannot express "fast for precompiled, blocked for
fresh-compile."

## Decision

`CapabilityLevel` has four levels:

- `tensor_native` — dedicated tensor-core hardware for this precision.
- `tensor_compat` — tensor cores via binary compat (works precompiled, blocked
  fresh-compile).
- `cuda_core` — general-purpose compute fallback (works, slow). Reserved for
  future silicon; unused by any canonical tier today.
- `unsupported` — no path.

`CapabilityLevel.__bool__` is False only for `unsupported`. Per-(tier, dtype)
entries are `CapabilityInfo(level, reason)`, where `reason` is tooltip-grade
text surfaces can render. `deployment_path_for_tier()` collapses the level plus
a `workload_kernel_source` ("precompiled" | "fresh_compile") into a concrete
path label.

Four canonical precision keys: `int8`, `fp8`, `bf16/fp16` (conflated — both map
to the same tensor-core class on all canonical silicon), and `q4_km` (weight-
only quants need their own tracking because support depends on the dequant
path: INT8 dequant on Neutron, FP16 dequant on FP-capable silicon).

## Consequences

- Surfaces ask question 1 (binary) or question 2 (4-level) as their need
  dictates; both surfaces are exposed.
- The `q4_km` peer key is what makes ADR 015 / AMENDMENT 1 correct: a Q4_K_M
  model's executability is governed by its quant scheme, not its fp16 compute
  dtype.
