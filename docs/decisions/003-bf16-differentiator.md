# ADR 003: Native BF16 on NPU as competitive differentiator

**Status:** Accepted
**Date:** 2026-04 (originally rescue-bird ADR 007)

## Context

Modern transformer-class vision models — SAM2, EdgeTAM, BEVFusion family,
many of the TAM-class detectors — are trained in BF16 and deploy best in
BF16 at the edge. BF16 has the same dynamic range as FP32 (8 exponent
bits) which matters for stable inference with the activation magnitudes
these models produce.

FP16 has only 5 exponent bits and silently saturates on large activations,
producing accuracy regressions that look like quantization noise. In
practice, FP16 deployment of a BF16-trained transformer needs per-layer
scaling and calibration that often costs 1-3% mAP / IoU.

INT8 fixes the dynamic range issue but requires post-training
quantization or QAT, which is a separate engineering task and
historically loses 1-5% accuracy unless done carefully.

The competitive landscape in this class:
- Some NPUs claim BF16 support natively in the MAC array
- Others quietly emulate BF16 via FP16 + scaling tricks
- Some don't support BF16 at all and the SDK silently downcasts

For any sizer use case running transformer-class perception (SAM2/EdgeTAM,
BEVFusion, transformer LLMs), this is a real differentiator.

## Decision

The engine exposes `tops_bf16`, `tops_fp16`, and `tops_int8` as separate
NPU spec fields. Site profiles declare which precisions the candidate
NPU supports natively (`supports_bf16: true|false`).

The KPI evaluator flags any chip that lacks BF16 support when the
workload was characterized in BF16, with a clear note in the failure
message. Reports include a precision-sensitivity section showing per-
precision latency comparison directly.

## Alternatives considered

- **Treat BF16 vs FP16 as equivalent.** Rejected. The accuracy delta on
  segmentation models is real and operationally significant (e.g. missed
  detection of partially-occluded objects).
- **Quantize everything to INT8.** Acceptable for some workloads but not
  all — segmentation masks degrade noticeably under PTQ at INT8 without
  significant calibration work. Sizers should expose this as an operating
  slider rather than the default.

## Consequences

- The NPU TOPS slider has effect at all three precisions (`bf16`, `fp16`,
  `int8`); the workload model picks the right one based on the
  perception precision setting.
- Competitive decks have a clean number to defend: "running [model] in
  native BF16 vs. FP16 emulation gives X% accuracy improvement at Y%
  latency cost."
- Sliders let the user move precision to compare directly.

## Worth knowing

This is the most competitively-relevant single ADR in the ratchet set.
Quantifying the BF16 differentiator with measurements (not just spec
sheets) is the highest-leverage thing to do with the instrumented stack.
Run the same model in BF16 and FP16 and put the numbers side-by-side in
the report.
