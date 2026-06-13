"""S6 metric tests (TDD): the five team-shape metrics + the registry mechanics.

Metrics are pure functions of a team's field-player pitch points (meters) + a
context dict, so they're verified here on hand-computed inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from deepcoach.stages.s6_metrics import (
    centroid,
    compactness,
    def_line,
    depth,
    get_metric,
    registered_metrics,
    width,
)

# x-span 30 (depth), y-span 10 (width); centroid (15, 5).
PTS = np.array([[0.0, 0.0], [30.0, 0.0], [0.0, 10.0], [30.0, 10.0]])


def test_centroid_is_mean_position():
    assert centroid(PTS, {}) == pytest.approx((15.0, 5.0))


def test_compactness_is_mean_distance_to_centroid():
    assert compactness(PTS, {}) == pytest.approx(math.sqrt(15**2 + 5**2))


def test_width_is_y_span():
    assert width(PTS, {}) == pytest.approx(10.0)


def test_depth_is_x_span():
    assert depth(PTS, {}) == pytest.approx(30.0)


def test_def_line_when_defending_left_goal_is_rearmost_x():
    assert def_line(PTS, {"own_goal_x": 0.0, "pitch_length": 105.0}) == pytest.approx(0.0)


def test_def_line_when_defending_right_goal_is_distance_from_that_goal():
    assert def_line(PTS, {"own_goal_x": 105.0, "pitch_length": 105.0}) == pytest.approx(75.0)


def test_all_v1_metrics_are_registered():
    for name in ["centroid", "compactness", "width", "depth", "def_line"]:
        assert name in registered_metrics()
        assert callable(get_metric(name))


def test_unknown_metric_raises():
    with pytest.raises(KeyError):
        get_metric("does_not_exist")
