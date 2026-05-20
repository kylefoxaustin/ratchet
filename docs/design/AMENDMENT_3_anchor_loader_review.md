# Review request — ratchet anchor-loader mismatch (proposed Amendment 3 / v0.2.1)

**Date:** 2026-05-20
**From:** ratchet engine session (phase 1 complete; phase 2 PAI retrofit blocked)
**For:** design reviewer
**Status:** Holding — no surface edits, no destructive actions — pending this decision.

---

## 1. Background (for a reviewer with no repo access)

`ratchet` is a pure-Python edge-SoC sizing engine being consolidated from four
production "surface" apps that previously stayed in sync by maintainer
discipline rather than shared code:

- **PAI sizer** (`personal-ai-assistant-sizer`) — LLM sizer
- **keyhole-sizer** — video sizer
- **keyhole** — video backend
- **Skippy** (`personal-ai-framework`) — agentic-AI framework

The consolidation runs in phases, each a separate session ending in a tagged
release:

- **Phase 1 — build ratchet v0.2.0.** DONE: tagged + pushed, 213 tests green
  (76 carried from v0.1.0 + 137 new), 15 ADRs, clean editable install. Two
  design amendments were folded in during the build:
  - *Amendment 1 (ADR 015):* `project_llm` gates dtype support on the model's
    **quant_scheme**, not its raw `compute_dtype`, so a Q4_K_M MoE model
    (compute_dtype `fp16`) correctly runs on INT8-only Neutron silicon via the
    INT8 dequant path and reaches its measured anchor instead of returning
    `DtypeMismatch`. (Also fixed projection field names/units to the catalog
    schema: `gguf_size_gb`, `active_params_b`.)
  - *Amendment 2 (ADR 011):* `overlay_llm_anchor` takes the `LLMModel`
    explicitly; the design's undefined `_dtype_for_model(model_key)` helper was
    removed (a projection result only carries a `model_key` string).
- **Phase 2 — retrofit PAI sizer** to import from ratchet instead of its local
  copies. THIS IS WHERE WE'RE BLOCKED.

**Governing rule (from the design spec, §12):** *"When ratchet turns out to be
missing something a surface needs, the answer is not to fork. Pause the
retrofit, switch to the ratchet session, add the missing piece to ratchet
v0.2.x, push, then return."* And §13.5: *"Phase 1 is not the place to invent
architecture. If the contract is wrong, fix the contract before fixing the
code."*

A note on secrecy: the **anchor values** (measured tok/s, TTFT, power) are
private and live only in gitignored secrets. The **schema** (field names,
structure, loader code) is public by design ("KEY-not-VALUE" discipline).
Everything shown below is schema/code only — no secret values.

---

## 2. The blocker

The design spec (§9) claimed ratchet's anchor-secrets loader is *"byte-identical
to what currently lives in PAI sizer and keyhole-sizer — it just needs a single
home."* I built ratchet's loader from the spec's prose.

On starting the PAI retrofit I found the spec's sketch **does not match the real
shared module**:

- PAI's and keyhole-sizer's `npu_anchors.py` **are** byte-identical to *each
  other* (confirmed by `diff` — zero differences).
- But **ratchet v0.2.0's loader diverged from both.** The real module is richer
  and has a different call signature.

### 2a. The real shared module (PAI + keyhole-sizer, byte-identical)

```python
@dataclass(frozen=True)
class LLMAnchor:
    tokps: float
    prefill_tokps: float
    mem_gb: float
    seqlen: int
    source: str
    measured_date: str
    peak_bw_gbps: float
    bw_share_frac: float
    bw_efficiency_frac: float
    notes: str = ""

    @property
    def badge(self) -> str:            # 🟢 measured / 🟡 vendor_spec / 🟠 projected
        return BADGE_FOR_SOURCE.get(self.source, "")

    def achieved_bw_gbps(self, share_override=None) -> float:
        share = share_override if share_override is not None else self.bw_share_frac
        return self.peak_bw_gbps * share * self.bw_efficiency_frac

    def bytes_per_token(self, share_override=None) -> float:
        if self.tokps <= 0:
            return 0.0
        return self.achieved_bw_gbps(share_override) * 1e9 / self.tokps


def load_llm_anchor(tier: str, precision: str, model_key: str) -> Optional[LLMAnchor]:
    # tier in {'mid','high'}, precision in {'int8','fp'}
    sub = f"{tier}_{precision}"
    ...
```

(There is a parallel `CNNAnchor` + `load_cnn_anchor(tier, precision, cnn_key)`
with the same shape, plus `achieved_bw_gbps()`.)

### 2b. What ratchet v0.2.0 actually shipped (from the spec prose)

```python
@dataclass(frozen=True)
class LLMAnchor:
    tokps: float
    ms_per_inference: float           # <- not in the real module
    peak_bw_gbps: float
    bw_share_frac: float = 0.75
    bw_efficiency_frac: float = 0.70
    source: str = "measured"
    measured_date: str = ""
    # missing: prefill_tokps, mem_gb, seqlen, notes
    # missing: badge, achieved_bw_gbps(), bytes_per_token()

def load_llm_anchor(tier_dtype: str, model_key: str) -> Optional[LLMAnchor]:  # 2-arg
    ...
```

### 2c. How PAI actually consumes anchors

PAI's `app.py` calls the loader **directly** and renders from the methods. It
does **not** use ratchet's "overlay onto a projection result" abstraction:

```python
anchor = load_llm_anchor(_tier_key, _prec_key, _model_key)     # 3-arg call
bpt = anchor.bytes_per_token(share_override=npu_share)         # method ratchet lacks
label = f"{anchor.badge} {model_label}"                        # property ratchet lacks
```

So dropping in ratchet's current loader breaks PAI on contact: wrong arity,
missing fields, missing `.bytes_per_token()` / `.badge`, and the overlay model
doesn't match PAI's usage pattern.

---

## 3. Proposed resolution — Amendment 3 / ratchet v0.2.1

Pause PAI. In a dedicated ratchet pass:

1. Replace `ratchet/anchors/loader.py` with the **real byte-identical module**:
   rich `LLMAnchor`/`CNNAnchor` (with `prefill_tokps`, `mem_gb`, `seqlen`,
   `notes`, `badge`, `achieved_bw_gbps()`, `bytes_per_token()`) and the 3-arg
   `load_llm_anchor(tier, precision, model_key)` signature.
2. Rework the dependent ratchet modules (`spec_routing.py`, `overlay.py`) and
   tests to match the corrected loader.
3. Update ADR 011, retag **v0.2.1**, push.
4. Resume PAI as a clean import swap (delete its local `npu_anchors.py`, import
   from `ratchet.anchors`).

---

## 4. Two questions for the reviewer

**Q1 — Adopt the real loader verbatim as canonical?**
The slim schema ratchet shipped loses `bytes_per_token()` / `badge` /
`prefill_tokps` / `mem_gb` / `seqlen` that PAI relies on. Adopting the real
(byte-identical, already-in-production) module as ratchet's canonical loader
seems clearly correct. Is there any reason the spec deliberately slimmed it
(e.g. an intent to migrate the surfaces *down* to a leaner schema) that I should
honor instead?

**Q2 — Keep or drop the `overlay_llm_anchor` abstraction?**
Ratchet shipped `overlay_llm_anchor(projected_result, hw, model, ...)` that
hot-swaps a projection result with a private anchor. **No surface actually uses
this** — PAI renders from `bytes_per_token()` directly. Per the engine's own
"rule of three" (no abstraction until ≥2 surfaces need it), should we **drop the
overlay** as premature and just expose the raw loader + methods? Or keep it as an
optional convenience for a future surface?

---

## 5. What I will NOT do without sign-off

- No edits to any surface repo (PAI included).
- No destructive actions.
- No "invent new architecture in the surface to fit ratchet."

Once you decide Q1/Q2, I'll execute the v0.2.1 anchor correction in ratchet,
verify tests + clean install, retag, and only then resume the PAI retrofit.
