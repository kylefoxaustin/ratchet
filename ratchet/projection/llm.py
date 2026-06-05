"""The LLM projection API — 4-path resolution cascade.

AMENDMENT 1 (2026-05-19): the dtype-compatibility gate (Step 0b) keys off the
model's *quant scheme*, not its raw compute_dtype. A Q4_K_M model has
compute_dtype 'fp16' but executes on INT8-only Neutron silicon via the INT8
dequant path; gating on compute_dtype alone wrongly rejects it and would skip a
measured anchor that exists for exactly that tier+model. The gate now uses
quant_scheme_capability_key() + hw_supports_dtype_via_key(). The cross-class
compute floor still uses the raw compute_dtype for raw-peak-TOPS lookup
(capability ≠ which TOPS field to read).

Resolution cascade (first hit wins):
  1. 🟢 measured            — per-cell measurement (hw.measured_llm)
  2. 🟢 measured_anchor     — tier-level override (hw.measured_decode_overrides)
  3. 🟡 same_class_anchor   — sibling tier in tier_family, BW-scaled
  4. 🔴 cross_class         — first-principles MAX(BW-floor, compute-floor)

Anchor-secrets overlays are NOT applied here — they run separately via
ratchet.anchors.overlay.overlay_llm_anchor.
"""
from typing import Optional

from ratchet.catalog.model import LLMModel, resolve_measurement_key
from ratchet.precision.capability import CapabilityLevel
from ratchet.precision.dtype_map import (
    FP4RuntimeMaturity,
    effective_compute_dtype,
    hw_peak_tops_for_dtype,
    hw_supports_dtype_via_key,
    quant_scheme_capability_key,
)
from ratchet.projection.feasibility import memory_feasibility
from ratchet.projection.result import (
    DtypeMismatch,
    Projected,
    ProjectionResult,
    WontFit,
)
from ratchet.tiers.hardware import Hardware


def project_llm(
    model: LLMModel,
    hw: Hardware,
    workload_id: str,
    *,
    prompt_tokens: int = 500,
    decode_tokens: int = 200,
    host_ms: float = 0.0,
    compiler_quality: float = 1.0,
    npu_share: Optional[float] = None,
    fp4_runtime_maturity: FP4RuntimeMaturity = "mature",
) -> ProjectionResult:
    """Project LLM performance for (model, hw, workload).

    Returns Projected | WontFit | DtypeMismatch. See module docstring for the
    cascade and the AMENDMENT 1 dtype-gate note.

    compiler_quality: trust factor for the vendor compiler stack; multiplies
        final rates (not the floor calculations).
    npu_share: bandwidth fraction; scales BW-bound paths only. Defaults to
        hw.npu_share_default.
    fp4_runtime_maturity: ADR 016. 'mature' (default) realizes the native FP4
        compute win; 'immature' models an FP4 model like INT4 weight-only (its
        compute floor falls to the bf16 floor) — for edge runtimes whose FP4
        kernels can't yet hit peak (e.g. llama.cpp today). No-op for non-FP4."""
    npu_share_actual = npu_share if npu_share is not None else hw.npu_share_default

    # Step 0a: Memory feasibility
    context = prompt_tokens + decode_tokens
    feas = memory_feasibility(model, hw, context)
    if feas.verdict == "wont_fit":
        return WontFit(
            hw_name=hw.name,
            model_key=model.key,
            workload_id=workload_id,
            required_gb=feas.required_gb,
            available_gb=feas.available_gb,
            headroom_gb=feas.headroom_gb,
            breakdown=feas.breakdown,
            prompt_tokens=prompt_tokens,
            decode_tokens=decode_tokens,
        )

    # Step 0b: Dtype/quant compatibility (AMENDMENT 1 — gate on quant scheme)
    cap_key = quant_scheme_capability_key(model.quant_scheme)
    cap_level = hw_supports_dtype_via_key(hw, cap_key)
    if cap_level is CapabilityLevel.UNSUPPORTED:
        return DtypeMismatch(
            hw_name=hw.name,
            model_key=model.key,
            workload_id=workload_id,
            required_dtype=model.quant_scheme,
            tier_capability="unsupported",
            retargeting_hint=(
                f"{hw.tier_lookup_name} cannot execute {model.quant_scheme}; "
                f"requires a tier with a {cap_key} path."
            ),
        )

    # Step 1: Per-cell measured (alias-aware)
    cell = _resolve_per_cell_measurement(hw, model, workload_id)
    if cell is not None:
        return _build_from_cell(cell, hw, model, workload_id,
                                prompt_tokens, decode_tokens, host_ms)

    # Step 2: Tier-level anchor (alias-aware)
    anchor = _resolve_tier_level_anchor(hw, model)
    if anchor is not None:
        return _build_from_tier_anchor(anchor, hw, model, workload_id,
                                       prompt_tokens, decode_tokens, host_ms,
                                       compiler_quality, fp4_runtime_maturity)

    # Step 3: Same-family anchor (BW-scaled)
    family = _find_same_family_anchor(hw, model)
    if family is not None:
        return _build_from_family_anchor(family, hw, model, workload_id,
                                         prompt_tokens, decode_tokens, host_ms,
                                         compiler_quality, fp4_runtime_maturity)

    # Step 4: Cross-class first-principles
    return _build_cross_class(hw, model, workload_id,
                              prompt_tokens=prompt_tokens,
                              decode_tokens=decode_tokens,
                              compiler_quality=compiler_quality,
                              npu_share_actual=npu_share_actual,
                              host_ms=host_ms,
                              fp4_runtime_maturity=fp4_runtime_maturity)


# ─── Resolution helpers ─────────────────────────────────────────────

def _resolve_per_cell_measurement(hw, model, workload_id) -> Optional[dict]:
    cell = hw.get_measured_llm_cell(model.key, workload_id)
    if cell is not None:
        return cell
    alias = resolve_measurement_key(model)
    if alias != model.key:
        return hw.get_measured_llm_cell(alias, workload_id)
    return None


def _resolve_tier_level_anchor(hw, model) -> Optional[dict]:
    """Return {'decode_tok_s', 'prefill_tok_s'?} from tier-level overrides."""
    decode_map = hw.measured_decode_overrides or {}
    prefill_map = hw.measured_prefill_overrides or {}
    for key in _candidate_keys(model):
        if key in decode_map:
            out = {"decode_tok_s": decode_map[key]}
            if key in prefill_map:
                out["prefill_tok_s"] = prefill_map[key]
            return out
    return None


def _find_same_family_anchor(hw, model) -> Optional[dict]:
    """Find a sibling tier in the same family carrying a tier-level anchor for
    this model, and BW-scale its decode rate to this tier (decode is BW-bound).

    Memory-upgrade clones (bw_projected) don't host sibling anchors."""
    if hw.tier_family is None or hw.bw_projected:
        return None
    from ratchet.tiers.registry import TIERS

    for sibling in TIERS.values():
        if sibling.name == hw.tier_lookup_name:
            continue
        if sibling.tier_family != hw.tier_family:
            continue
        sib_decode = sibling.measured_decode_overrides or {}
        for key in _candidate_keys(model):
            if key in sib_decode:
                sib_eff = sibling.effective_bandwidth_gbs
                bw_ratio = (hw.effective_bandwidth_gbs / sib_eff) if sib_eff else 1.0
                out = {
                    "decode_tok_s": sib_decode[key] * bw_ratio,
                    "sibling_name": sibling.name,
                }
                sib_prefill = sibling.measured_prefill_overrides or {}
                if key in sib_prefill:
                    # Prefill is compute-bound; carry across unscaled.
                    out["prefill_tok_s"] = sib_prefill[key]
                return out
    return None


def _candidate_keys(model) -> tuple[str, ...]:
    alias = resolve_measurement_key(model)
    return (model.key,) if alias == model.key else (model.key, alias)


# ─── Result builders ────────────────────────────────────────────────

def _timing(decode_tok_s, prefill_tok_s, prompt_tokens, decode_tokens,
            host_s, ttft_s):
    decode_s = decode_tokens / max(decode_tok_s, 1e-6)
    prefill_s = prompt_tokens / max(prefill_tok_s, 1e-6)
    total_s = host_s + prefill_s + decode_s
    return decode_s, prefill_s, total_s


def _build_from_cell(cell, hw, model, workload_id, prompt_tokens,
                     decode_tokens, host_ms) -> Projected:
    decode_tok_s = float(cell["decode_tok_s"])
    prefill_tok_s = float(cell.get("prefill_tok_s") or 0.0)
    host_s = float(cell.get("host_ms", host_ms)) / 1000.0
    if cell.get("ttft_s") is not None:
        ttft_s = float(cell["ttft_s"])
    elif prefill_tok_s > 0:
        ttft_s = prompt_tokens / prefill_tok_s
    else:
        ttft_s = 0.0
    if prefill_tok_s <= 0 and ttft_s > 0:
        prefill_tok_s = prompt_tokens / ttft_s
    decode_s, prefill_s, total_s = _timing(
        decode_tok_s, prefill_tok_s, prompt_tokens, decode_tokens, host_s, ttft_s)
    return Projected(
        decode_tok_s=round(decode_tok_s, 2),
        prefill_tok_s=round(prefill_tok_s, 2),
        ttft_s=round(ttft_s, 4),
        decode_tokens=decode_tokens, prompt_tokens=prompt_tokens,
        decode_s=round(decode_s, 3), prefill_s=round(prefill_s, 3),
        total_s=round(total_s, 3), host_ms=round(host_s * 1000, 2),
        source="measured", regime="bw_bound",
        hw_name=hw.name, model_key=model.key, workload_id=workload_id,
    )


def _build_from_tier_anchor(anchor, hw, model, workload_id, prompt_tokens,
                            decode_tokens, host_ms, compiler_quality,
                            fp4_runtime_maturity="mature") -> Projected:
    return _build_from_rates(
        anchor, hw, model, workload_id, prompt_tokens, decode_tokens, host_ms,
        compiler_quality, source="measured_anchor",
        fp4_runtime_maturity=fp4_runtime_maturity)


def _build_from_family_anchor(anchor, hw, model, workload_id, prompt_tokens,
                              decode_tokens, host_ms, compiler_quality,
                              fp4_runtime_maturity="mature") -> Projected:
    return _build_from_rates(
        anchor, hw, model, workload_id, prompt_tokens, decode_tokens, host_ms,
        compiler_quality, source="same_class_anchor",
        fp4_runtime_maturity=fp4_runtime_maturity)


def _build_from_rates(anchor, hw, model, workload_id, prompt_tokens,
                      decode_tokens, host_ms, compiler_quality,
                      *, source, fp4_runtime_maturity="mature") -> Projected:
    decode_tok_s = anchor["decode_tok_s"] * compiler_quality
    prefill_tok_s = anchor.get("prefill_tok_s")
    host_s = (host_ms or hw.compute_overhead_ms) / 1000.0
    if prefill_tok_s:
        prefill_tok_s = prefill_tok_s * compiler_quality
        ttft_s = prompt_tokens / max(prefill_tok_s, 1e-6)
    else:
        # No measured prefill — fall back to compute-floor prefill estimate.
        prefill_tok_s, ttft_s = _prefill_floor(hw, model, prompt_tokens,
                                                compiler_quality,
                                                fp4_runtime_maturity)
    decode_s, prefill_s, total_s = _timing(
        decode_tok_s, prefill_tok_s, prompt_tokens, decode_tokens, host_s, ttft_s)
    return Projected(
        decode_tok_s=round(decode_tok_s, 2),
        prefill_tok_s=round(prefill_tok_s, 2),
        ttft_s=round(ttft_s, 4),
        decode_tokens=decode_tokens, prompt_tokens=prompt_tokens,
        decode_s=round(decode_s, 3), prefill_s=round(prefill_s, 3),
        total_s=round(total_s, 3), host_ms=round(host_s * 1000, 2),
        source=source, regime="bw_bound",
        hw_name=hw.name, model_key=model.key, workload_id=workload_id,
    )


def _prefill_floor(hw, model, prompt_tokens, compiler_quality,
                   fp4_runtime_maturity="mature"):
    """Prefill rate + ttft from the cross-class compute/BW floor."""
    active_params_gb = model.active_params_b * model.bytes_per_param
    gops_per_token = 2 * model.active_params_b
    floor_dtype = effective_compute_dtype(model.compute_dtype, fp4_runtime_maturity)
    peak_tops_llm = max(hw_peak_tops_for_dtype(hw, floor_dtype), 1e-9)
    bw_floor_ms = (active_params_gb / hw.effective_bandwidth_gbs) * 1000.0
    compute_floor_ms = (
        gops_per_token * prompt_tokens / (peak_tops_llm * hw.llm_prefill_util_factor)
    )
    ttft_ms = max(bw_floor_ms, compute_floor_ms) + hw.compute_overhead_ms
    prefill_tok_s = prompt_tokens / max(ttft_ms / 1000.0, 1e-6) * compiler_quality
    return prefill_tok_s, ttft_ms / 1000.0


def _build_cross_class(hw, model, workload_id, *, prompt_tokens, decode_tokens,
                       compiler_quality, npu_share_actual, host_ms,
                       fp4_runtime_maturity="mature") -> Projected:
    """First-principles projection: max(BW_floor, compute_floor) + overhead.

    LLM compute floor uses RAW peak TOPS (calibrated against
    llm_prefill_util_factor, NOT effective_tops — per ADR 015). ADR 016: on an
    immature FP4 runtime the floor dtype falls to bf16 (no realized FP4 GEMM win);
    decode stays governed by the BW floor either way (4-bit weight bytes)."""
    active_params_gb = model.active_params_b * model.bytes_per_param  # AMENDMENT 1
    gops_per_token = 2 * model.active_params_b                        # AMENDMENT 1
    floor_dtype = effective_compute_dtype(model.compute_dtype, fp4_runtime_maturity)
    peak_tops_llm = max(hw_peak_tops_for_dtype(hw, floor_dtype), 1e-9)

    # Decode-side per-token floors
    decode_bw_realized = (
        hw.effective_bandwidth_gbs * hw.llm_decode_bw_realization * npu_share_actual
    )
    bw_floor_ms_decode = (active_params_gb / max(decode_bw_realized, 1e-9)) * 1000.0
    compute_floor_ms_decode = gops_per_token / (
        peak_tops_llm * hw.llm_prefill_util_factor)

    per_token_ms = max(bw_floor_ms_decode, compute_floor_ms_decode)
    decode_tok_s = (1000.0 / max(per_token_ms, 1e-6)) * compiler_quality

    regime = "bw_bound" if bw_floor_ms_decode >= compute_floor_ms_decode else "compute_bound"

    # Prefill side: per-batch BW, per-token compute
    bw_floor_ms_prefill = (active_params_gb / hw.effective_bandwidth_gbs) * 1000.0
    compute_floor_ms_prefill = (
        gops_per_token * prompt_tokens / (peak_tops_llm * hw.llm_prefill_util_factor)
    )
    ttft_ms = max(bw_floor_ms_prefill, compute_floor_ms_prefill) + hw.compute_overhead_ms
    prefill_tok_s = prompt_tokens / max(ttft_ms / 1000.0, 1e-6) * compiler_quality

    decode_ceiling = 1000.0 / max(bw_floor_ms_decode, 1e-6)
    decode_s = decode_tokens / max(decode_tok_s, 1e-6)
    prefill_s = prompt_tokens / max(prefill_tok_s, 1e-6)
    host_s = (host_ms or hw.compute_overhead_ms) / 1000.0
    total_s = host_s + prefill_s + decode_s

    return Projected(
        decode_tok_s=round(decode_tok_s, 2),
        prefill_tok_s=round(prefill_tok_s, 2),
        ttft_s=round(ttft_ms / 1000.0, 4),
        decode_tokens=decode_tokens, prompt_tokens=prompt_tokens,
        decode_s=round(decode_s, 3), prefill_s=round(prefill_s, 3),
        total_s=round(total_s, 3), host_ms=round(host_s * 1000, 2),
        source="cross_class", regime=regime,
        hw_name=hw.name, model_key=model.key, workload_id=workload_id,
        decode_ceiling_tok_s=round(decode_ceiling, 2),
    )
