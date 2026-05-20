"""Anchor-secrets loader — private silicon measurements from gitignored secrets.

KEY-not-VALUE discipline: the schema (field names, structure) is public and
lives in source; the values are credentials and live ONLY in a gitignored
.streamlit/secrets.toml, loaded at runtime. The loader is byte-equivalent to the
copies that previously lived in PAI sizer and keyhole-sizer.

Fallback semantics: returns None whenever Streamlit is unavailable, secrets are
absent, the cell is unpopulated, or a value is malformed. Never raises.
"""
from dataclasses import dataclass
from typing import Optional

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


@dataclass(frozen=True)
class LLMAnchor:
    tokps: float
    ms_per_inference: float
    peak_bw_gbps: float
    bw_share_frac: float = 0.75
    bw_efficiency_frac: float = 0.70
    source: str = "measured"
    measured_date: str = ""


@dataclass(frozen=True)
class CNNAnchor:
    ms_per_inference: float
    fps: float
    peak_bw_gbps: float
    bw_share_frac: float = 0.75
    bw_efficiency_frac: float = 0.70
    source: str = "measured"
    measured_date: str = ""


def _try_get(cell, key, default):
    try:
        return cell[key]
    except (KeyError, TypeError):
        return default


def load_llm_anchor(tier_dtype: str, model_key: str) -> Optional[LLMAnchor]:
    """Load an LLM anchor. Returns None when Streamlit unavailable, secrets
    absent, cell unpopulated, or value malformed."""
    if not _HAS_STREAMLIT:
        return None
    try:
        cells = st.secrets["npu_llm_anchors"][tier_dtype]
        cell = cells[model_key]
        return LLMAnchor(
            tokps=float(cell["tokps"]),
            ms_per_inference=float(cell["ms_per_inference"]),
            peak_bw_gbps=float(cell["peak_bw_gbps"]),
            bw_share_frac=float(_try_get(cell, "bw_share_frac", 0.75)),
            bw_efficiency_frac=float(_try_get(cell, "bw_efficiency_frac", 0.70)),
            source=str(_try_get(cell, "source", "measured")),
            measured_date=str(_try_get(cell, "measured_date", "")),
        )
    except (KeyError, ValueError, TypeError):
        return None


def load_cnn_anchor(tier_dtype: str, model_key: str) -> Optional[CNNAnchor]:
    """Load a CNN anchor. Same fallback semantics as load_llm_anchor."""
    if not _HAS_STREAMLIT:
        return None
    try:
        cells = st.secrets["cnn_anchors"][tier_dtype]
        cell = cells[model_key]
        return CNNAnchor(
            ms_per_inference=float(cell["ms_per_inference"]),
            fps=float(cell["fps"]),
            peak_bw_gbps=float(cell["peak_bw_gbps"]),
            bw_share_frac=float(_try_get(cell, "bw_share_frac", 0.75)),
            bw_efficiency_frac=float(_try_get(cell, "bw_efficiency_frac", 0.70)),
            source=str(_try_get(cell, "source", "measured")),
            measured_date=str(_try_get(cell, "measured_date", "")),
        )
    except (KeyError, ValueError, TypeError):
        return None
