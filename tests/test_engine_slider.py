"""Framework-level tests for ratchet.engine.slider.

These exercise the Slider primitive and the apply mechanism using synthetic
Slider instances. No drone/video/AI vocabulary appears in the assertions or
fixtures.
"""

from __future__ import annotations

import pytest

from ratchet.engine.slider import (
    Slider,
    apply_sliders,
    default_values,
    slider_categories,
    _set_path,
    _scale_path,
)


# ──────────────────────────────────────────────────────────────────────
# Slider dataclass
# ──────────────────────────────────────────────────────────────────────

class TestSliderClamp:
    def test_clamp_within_range_returns_input(self):
        s = Slider(name="s", description="", category="capability", units="x",
                   default=5.0, min_val=0.0, max_val=10.0)
        assert s.clamp(5.0) == 5.0
        assert s.clamp(0.0) == 0.0
        assert s.clamp(10.0) == 10.0

    def test_clamp_below_min_returns_min(self):
        s = Slider(name="s", description="", category="capability", units="x",
                   default=5.0, min_val=0.0, max_val=10.0)
        assert s.clamp(-100.0) == 0.0

    def test_clamp_above_max_returns_max(self):
        s = Slider(name="s", description="", category="capability", units="x",
                   default=5.0, min_val=0.0, max_val=10.0)
        assert s.clamp(1e9) == 10.0

    def test_default_affects_list_is_empty(self):
        s = Slider(name="s", description="", category="capability", units="x",
                   default=1.0, min_val=0.0, max_val=2.0)
        assert s.affects == []

    def test_default_apply_is_none(self):
        s = Slider(name="s", description="", category="capability", units="x",
                   default=1.0, min_val=0.0, max_val=2.0)
        assert s.apply is None


# ──────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────

class TestSetPath:
    def test_top_level(self):
        d = {}
        _set_path(d, "x", 7)
        assert d == {"x": 7}

    def test_nested_creates_intermediate_dicts(self):
        d = {}
        _set_path(d, "a.b.c", 42)
        assert d == {"a": {"b": {"c": 42}}}

    def test_overwrites_existing(self):
        d = {"a": {"b": 1}}
        _set_path(d, "a.b", 99)
        assert d["a"]["b"] == 99


class TestScalePath:
    def test_scale_existing(self):
        d = {"x": {"y": 10.0}}
        _scale_path(d, "x.y", 2.5)
        assert d["x"]["y"] == 25.0

    def test_scale_top_level(self):
        d = {"v": 4}
        _scale_path(d, "v", 0.5)
        assert d["v"] == 2.0


# ──────────────────────────────────────────────────────────────────────
# Catalog operations
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_catalog():
    """A small synthetic slider catalog with no domain vocabulary."""
    return {
        "alpha": Slider(
            name="alpha", description="multiplier on profile.foo",
            category="capability", units="x",
            default=1.0, min_val=0.5, max_val=2.0,
            apply=lambda p, w, v: _set_path(p, "foo.alpha", v),
        ),
        "beta": Slider(
            name="beta", description="multiplier on workload.bar",
            category="workload", units="x",
            default=10.0, min_val=1.0, max_val=100.0,
            apply=lambda p, w, v: _set_path(w, "bar.beta", v),
        ),
        "gamma": Slider(
            name="gamma", description="informational, no apply",
            category="capability", units="?",
            default=3.0, min_val=0.0, max_val=5.0,
            apply=None,
        ),
    }


class TestDefaultValues:
    def test_returns_one_per_slider(self, synthetic_catalog):
        d = default_values(synthetic_catalog)
        assert d == {"alpha": 1.0, "beta": 10.0, "gamma": 3.0}

    def test_empty_catalog(self):
        assert default_values({}) == {}


class TestSliderCategories:
    def test_groups_by_category(self, synthetic_catalog):
        cats = slider_categories(synthetic_catalog)
        assert set(cats.keys()) == {"capability", "workload"}
        assert {s.name for s in cats["capability"]} == {"alpha", "gamma"}
        assert {s.name for s in cats["workload"]} == {"beta"}


class TestApplySliders:
    def test_applies_to_profile(self, synthetic_catalog):
        profile, workload = {}, {}
        apply_sliders(synthetic_catalog, profile, workload, {"alpha": 1.5})
        assert profile["foo"]["alpha"] == 1.5
        assert workload == {}    # untouched

    def test_applies_to_workload(self, synthetic_catalog):
        profile, workload = {}, {}
        apply_sliders(synthetic_catalog, profile, workload, {"beta": 50.0})
        assert workload["bar"]["beta"] == 50.0

    def test_clamps_out_of_range(self, synthetic_catalog):
        profile, workload = {}, {}
        # alpha range is 0.5..2.0; pass 1e9 → should clamp to 2.0
        apply_sliders(synthetic_catalog, profile, workload, {"alpha": 1e9})
        assert profile["foo"]["alpha"] == 2.0

    def test_unknown_slider_ignored(self, synthetic_catalog):
        profile, workload = {}, {}
        # Should not raise
        apply_sliders(synthetic_catalog, profile, workload,
                      {"alpha": 1.5, "not_a_slider": 999})
        assert profile["foo"]["alpha"] == 1.5

    def test_informational_slider_skipped(self, synthetic_catalog):
        profile, workload = {}, {}
        apply_sliders(synthetic_catalog, profile, workload, {"gamma": 4.0})
        # gamma has apply=None — nothing should happen
        assert profile == {}
        assert workload == {}
