"""Integration test for the geometry half: S5 project -> S6 metrics.

Runs the real stage run() drivers against synthetic tracks + a solved homography
written to disk — no YOLO, no video. Proves the artifact contracts wire stage to
stage and that projected dots + team shapes come out sane.
"""

from __future__ import annotations

import math

from deepcoach.contracts.common import BBox, ClassLabel, Role
from deepcoach.contracts.pitch import (
    Homography,
    HomographyArtifact,
    PitchLandmark,
)
from deepcoach.contracts.tracks import Track, TrackArtifact, TrackedFrame
from deepcoach.io.artifacts import now_utc_iso, out_dir, save_artifact
from deepcoach.io.config import ClipConfig
from deepcoach.stages import s5_project, s6_metrics
from deepcoach.stages.s4_homography import solve_homography

LANDMARKS = [
    PitchLandmark(name="tl", pixel_xy=(100.0, 100.0), pitch_xy=(0.0, 0.0)),
    PitchLandmark(name="tr", pixel_xy=(500.0, 120.0), pitch_xy=(105.0, 0.0)),
    PitchLandmark(name="br", pixel_xy=(520.0, 400.0), pitch_xy=(105.0, 68.0)),
    PitchLandmark(name="bl", pixel_xy=(80.0, 420.0), pitch_xy=(0.0, 68.0)),
]


def _bbox_with_ground(px: float, py: float) -> BBox:
    """A bbox whose ground point (bottom-center) is exactly (px, py)."""
    return BBox(x1=px - 5, y1=py - 20, x2=px + 5, y2=py)


def _config() -> ClipConfig:
    return ClipConfig.model_validate(
        {
            "clip": {"path": "data/in/synth.mp4"},
            "pitch": {"length_m": 105, "width_m": 68},
            "metrics": {"enabled": ["centroid", "compactness", "width", "depth", "def_line"]},
        }
    )


def _write_inputs(cfg: ClipConfig):
    od = out_dir(cfg.clip_name())
    h = cfg.config_hash()

    matrix, err = solve_homography(LANDMARKS)
    save_artifact(
        HomographyArtifact(
            config_hash=h,
            stage="s4_homography",
            created_utc=now_utc_iso(),
            static_camera=True,
            homographies=[
                Homography(
                    matrix=matrix,
                    source_keyframe_idx=0,
                    reprojection_error_px=err,
                    pitch_landmarks_used=LANDMARKS,
                )
            ],
        ),
        od / "homography.json",
    )

    # Team 0 clustered near the left/top pixel region, team 1 near the right/bottom.
    team0_px = [(150, 160), (180, 200), (160, 240), (200, 180), (140, 220)]
    team1_px = [(460, 360), (440, 320), (480, 340), (420, 300), (470, 380)]
    tracks = []
    tid = 0
    for px, py in team0_px:
        tracks.append(Track(track_id=tid, frame_idx=0, bbox=_bbox_with_ground(px, py), cls=ClassLabel.player, confidence=0.9, team=0, role=Role.field))
        tid += 1
    for px, py in team1_px:
        tracks.append(Track(track_id=tid, frame_idx=0, bbox=_bbox_with_ground(px, py), cls=ClassLabel.player, confidence=0.9, team=1, role=Role.field))
        tid += 1
    tracks.append(Track(track_id=tid, frame_idx=0, bbox=_bbox_with_ground(300, 250), cls=ClassLabel.ball, confidence=0.5, team=None, role=Role.unknown))

    save_artifact(
        TrackArtifact(
            config_hash=h,
            stage="s3_track",
            created_utc=now_utc_iso(),
            fps=25.0,
            frame_count=1,
            n_tracks=len(tracks),
            frames=[TrackedFrame(frame_idx=0, timestamp_s=0.0, tracks=tracks)],
        ),
        od / "tracks.json",
    )


def test_project_then_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep artifacts inside the temp dir
    cfg = _config()
    _write_inputs(cfg)

    proj = s5_project.run(cfg)
    assert len(proj.frames) == 1
    entries = proj.frames[0].entries
    assert len(entries) == 11  # 10 players + 1 ball
    for e in entries:
        assert 0.0 <= e.projection_confidence <= 1.0
        # exact correspondences -> ~0 reprojection error -> high confidence inside hull
        assert e.projection_confidence > 0.9

    metrics = s6_metrics.run(cfg)
    assert len(metrics.frames) == 1
    teams = metrics.frames[0].teams
    assert {t.team for t in teams} == {0, 1}
    for t in teams:
        assert t.n_players == 5
        assert 0.0 <= t.centroid.x_m <= cfg.pitch.length_m
        assert 0.0 <= t.centroid.y_m <= cfg.pitch.width_m
        assert t.compactness_m >= 0.0
        assert math.isfinite(t.defensive_line_height_m)

    # Team 0 (left cluster) should sit at lower x than team 1 (right cluster).
    cx = {t.team: t.centroid.x_m for t in teams}
    assert cx[0] < cx[1]


def test_radar_and_report_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _config()
    _write_inputs(cfg)
    s5_project.run(cfg)
    s6_metrics.run(cfg)

    from deepcoach.quality import report as quality_report
    from deepcoach.render import heatmap, radar

    radar_path = radar.render(cfg)
    assert radar_path.endswith("radar.mp4")
    assert (tmp_path / radar_path).exists()

    # one synthetic frame -> too few points for a heatmap; should run and produce none
    assert heatmap.render(cfg) == []

    r = quality_report.build_report(cfg)
    assert r["project"]["projected_dots"] == 11
    assert r["metrics"]["frames_with_both_teams"] == 1
    assert r["homography"]["trustworthy"] is True
    assert (out_dir(cfg.clip_name()) / "report.txt").exists()
