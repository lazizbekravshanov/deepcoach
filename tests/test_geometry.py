"""Geometry tests (TDD): homography solve + pixel->pitch projection + confidence.

This is the make-or-break math. A wrong dot is worse than a missing one, so these
tests pin down both the projection itself and the confidence that surfaces doubt.
"""

from __future__ import annotations

import math

import pytest

from deepcoach.contracts.pitch import PitchLandmark
from deepcoach.stages.s4_homography import solve_homography
from deepcoach.stages.s5_project import (
    dist_outside_hull_px,
    project_point,
    projection_confidence,
)

# A general (non-affine) pixel quad mapped to the pitch corners in meters.
LANDMARKS = [
    PitchLandmark(name="tl", pixel_xy=(100.0, 100.0), pitch_xy=(0.0, 0.0)),
    PitchLandmark(name="tr", pixel_xy=(500.0, 120.0), pitch_xy=(105.0, 0.0)),
    PitchLandmark(name="br", pixel_xy=(520.0, 400.0), pitch_xy=(105.0, 68.0)),
    PitchLandmark(name="bl", pixel_xy=(80.0, 420.0), pitch_xy=(0.0, 68.0)),
]


# --- S4: solve_homography ---


def test_solve_recovers_landmarks_with_near_zero_error():
    H, err = solve_homography(LANDMARKS)
    assert len(H) == 3 and len(H[0]) == 3
    assert err < 1e-6  # exact correspondences -> ~0 reprojection error


def test_solved_homography_maps_pixels_to_known_pitch_points():
    H, _ = solve_homography(LANDMARKS)
    for lm in LANDMARKS:
        x, y = project_point(H, lm.pixel_xy)
        assert math.isclose(x, lm.pitch_xy[0], abs_tol=1e-3)
        assert math.isclose(y, lm.pitch_xy[1], abs_tol=1e-3)


def test_solve_requires_at_least_four_landmarks():
    with pytest.raises(ValueError):
        solve_homography(LANDMARKS[:3])


# --- S5: project_point ---


def test_project_point_identity_is_passthrough():
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert project_point(identity, (5.0, 7.0)) == pytest.approx((5.0, 7.0))


def test_project_point_applies_translation_and_scale():
    # x' = 2x + 10, y' = 3y + 20
    H = [[2.0, 0.0, 10.0], [0.0, 3.0, 20.0], [0.0, 0.0, 1.0]]
    assert project_point(H, (4.0, 5.0)) == pytest.approx((18.0, 35.0))


# --- S5: convex-hull distance ---

HULL = [(100.0, 100.0), (500.0, 120.0), (520.0, 400.0), (80.0, 420.0)]


def test_point_inside_hull_has_zero_outside_distance():
    assert dist_outside_hull_px((300.0, 250.0), HULL) == 0.0


def test_point_outside_hull_has_positive_distance():
    d = dist_outside_hull_px((1000.0, 250.0), HULL)
    assert d > 0.0


# --- S5: projection_confidence ---


def test_confidence_is_one_for_perfect_homography_inside_hull():
    assert projection_confidence(reproj_err_px=0.0, dist_outside_hull=0.0) == pytest.approx(1.0)


def test_confidence_decreases_with_reprojection_error():
    near = projection_confidence(reproj_err_px=2.0, dist_outside_hull=0.0)
    far = projection_confidence(reproj_err_px=20.0, dist_outside_hull=0.0)
    assert 0.0 <= far < near <= 1.0


def test_confidence_decreases_outside_hull():
    inside = projection_confidence(reproj_err_px=2.0, dist_outside_hull=0.0)
    outside = projection_confidence(reproj_err_px=2.0, dist_outside_hull=200.0)
    assert 0.0 <= outside < inside


def test_confidence_is_clamped_to_unit_interval():
    c = projection_confidence(reproj_err_px=999.0, dist_outside_hull=9999.0)
    assert 0.0 <= c <= 1.0
