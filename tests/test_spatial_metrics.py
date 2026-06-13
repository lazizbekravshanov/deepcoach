"""TDD for the new spatial metrics: pitch control (territorial dominance) and
team space occupied (convex-hull area), plus their registry wiring.

These are per-frame metrics computed from positions only — robust to the track
fragmentation that broadcast/highlight footage causes.
"""

from __future__ import annotations

import numpy as np
import pytest

from deepcoach.stages.s6_metrics import (
    frame_metrics,
    hull_area_m2,
    per_team_metrics,
    pitch_control,
)

CTX = {"pitch_length": 105.0, "pitch_width": 68.0}


# --- pitch control (frame-level: needs both teams) ---


def test_pitch_control_symmetric_is_roughly_even():
    by_team = {0: np.array([[10.0, 34.0]]), 1: np.array([[95.0, 34.0]])}
    pc = pitch_control(by_team, CTX)
    assert pc["0"] == pytest.approx(0.5, abs=0.05)
    assert pc["1"] == pytest.approx(0.5, abs=0.05)


def test_pitch_control_fractions_sum_to_one():
    by_team = {0: np.array([[30.0, 20.0], [40.0, 50.0]]), 1: np.array([[80.0, 34.0]])}
    pc = pitch_control(by_team, CTX)
    assert sum(pc.values()) == pytest.approx(1.0, abs=1e-6)


def test_pitch_control_central_team_dominates_a_cornered_team():
    central = {0: np.array([[52.5, 34.0]]), 1: np.array([[2.0, 2.0]])}
    pc = pitch_control(central, CTX)
    assert pc["0"] > pc["1"]


# --- team space occupied (per-team: convex hull area) ---


def test_hull_area_of_a_rectangle():
    pts = np.array([[0.0, 0.0], [30.0, 0.0], [0.0, 10.0], [30.0, 10.0]])
    assert hull_area_m2(pts, CTX) == pytest.approx(300.0)


def test_hull_area_degenerate_is_zero():
    assert hull_area_m2(np.array([[5.0, 5.0], [6.0, 6.0]]), CTX) == 0.0  # <3 points


# --- registry wiring ---


def test_hull_area_is_a_registered_per_team_metric():
    assert "hull_area" in per_team_metrics()


def test_pitch_control_is_a_registered_frame_metric():
    assert "pitch_control" in frame_metrics()
