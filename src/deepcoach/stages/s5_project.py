"""S5 project — apply homography to tracked positions -> ProjectionArtifact (meters).

Pure, deterministic geometry. For each track in each frame: take the bbox ground
point (bottom-center, where feet meet the pitch), apply the homography, emit a
PitchCoordinate in meters plus projection_confidence.

projection_confidence (the wrong-dot guard) decays with:
  (1) the homography reprojection error, and
  (2) the point's distance from the convex hull of the landmarks (extrapolation).

This stage has no model and no randomness — kept boring and stable on purpose.

NOTE: implementation deferred — backbone review pause (ARCHITECTURE.md §13).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..contracts.pitch import (
    HomographyArtifact,
    PitchCoordinate,
    ProjectedEntry,
    ProjectedFrame,
    ProjectionArtifact,
)
from ..contracts.tracks import TrackArtifact
from ..io.artifacts import load_artifact, now_utc_iso, out_dir, save_artifact
from ..io.config import ClipConfig

STAGE = "s5_project"

# A dot below this confidence is "untrustworthy" for the quality report.
LOW_CONF = 0.5

# Confidence decay scales (pixels). Tunable; chosen so a few px of error or being
# tens of px outside the landmark hull noticeably erodes trust.
ERR_SCALE_PX = 8.0
HULL_SCALE_PX = 80.0


def project_point(matrix: list[list[float]], pixel_xy: tuple[float, float]) -> tuple[float, float]:
    """Apply a 3x3 homography to a pixel point -> pitch coords (meters). Pure geometry."""
    H = np.asarray(matrix, dtype=np.float64)
    v = np.array([pixel_xy[0], pixel_xy[1], 1.0], dtype=np.float64)
    w = H @ v
    return (float(w[0] / w[2]), float(w[1] / w[2]))


def dist_outside_hull_px(pixel_xy: tuple[float, float], hull_points: list[tuple[float, float]]) -> float:
    """Distance (px) a point lies OUTSIDE the convex hull of the landmarks; 0 if inside.

    Points outside the hull are extrapolations of the homography and untrustworthy
    (the wrong-dot risk grows with extrapolation distance).
    """
    hull = cv2.convexHull(np.array(hull_points, dtype=np.float32))
    signed = cv2.pointPolygonTest(hull, (float(pixel_xy[0]), float(pixel_xy[1])), True)
    return float(max(0.0, -signed))  # signed > 0 inside, < 0 outside (|.| = distance)


def projection_confidence(reproj_err_px: float, dist_outside_hull: float) -> float:
    """0..1 trust in a projected dot. 1.0 = perfect homography, point inside hull.

    Decays exponentially with reprojection error and with extrapolation distance
    outside the landmark hull. This is the wrong-dot guard made numeric.
    """
    err_term = math.exp(-max(0.0, reproj_err_px) / ERR_SCALE_PX)
    hull_term = math.exp(-max(0.0, dist_outside_hull) / HULL_SCALE_PX)
    return max(0.0, min(1.0, err_term * hull_term))


def run(config: ClipConfig) -> ProjectionArtifact:
    """Project every tracked foot-point to pitch meters; carry projection_confidence."""
    name = config.clip_name()
    od = out_dir(name)
    cfg_hash = config.config_hash()

    tracks = load_artifact(od / "tracks.json", TrackArtifact, cfg_hash)
    homo = load_artifact(od / "homography.json", HomographyArtifact, cfg_hash)
    if not homo.homographies:
        raise ValueError("homography artifact has no homographies")

    # v1 static camera: a single homography for the whole clip.
    H = homo.homographies[0]
    matrix = H.matrix
    reproj = H.reprojection_error_px
    hull = [lm.pixel_xy for lm in H.pitch_landmarks_used]
    L, W = config.pitch.length_m, config.pitch.width_m

    out_frames: list[ProjectedFrame] = []
    total = low = oob = 0
    for tf in tracks.frames:
        entries: list[ProjectedEntry] = []
        for t in tf.tracks:
            gx, gy = t.bbox.ground_point()  # feet, not centroid
            x_m, y_m = project_point(matrix, (gx, gy))
            conf = projection_confidence(reproj, dist_outside_hull_px((gx, gy), hull))
            entries.append(
                ProjectedEntry(
                    track_id=t.track_id,
                    team=t.team,
                    role=t.role,
                    cls=t.cls,
                    pitch_xy=PitchCoordinate(x_m=x_m, y_m=y_m),
                    projection_confidence=conf,
                )
            )
            total += 1
            if conf < LOW_CONF:
                low += 1
            if not (0.0 <= x_m <= L and 0.0 <= y_m <= W):
                oob += 1
        out_frames.append(ProjectedFrame(frame_idx=tf.frame_idx, timestamp_s=tf.timestamp_s, entries=entries))

    art = ProjectionArtifact(
        config_hash=cfg_hash,
        stage=STAGE,
        created_utc=now_utc_iso(),
        pitch_length_m=L,
        pitch_width_m=W,
        frames=out_frames,
    )
    save_artifact(art, od / "projected.json")
    pct_low = (100.0 * low / total) if total else 0.0
    pct_oob = (100.0 * oob / total) if total else 0.0
    print(
        f"[s5_project] {name}: {total} dots, reproj_err={reproj:.2f}px, "
        f"{pct_low:.1f}% low-confidence, {pct_oob:.1f}% outside pitch bounds"
    )
    if pct_oob > 5.0:
        print("[s5_project] WARNING: many dots fall outside the pitch — check homography landmarks.")
    return art
