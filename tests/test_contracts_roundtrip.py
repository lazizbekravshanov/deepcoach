"""Contract round-trip tests: every artifact survives save -> load unchanged, and
the loader enforces schema/config compatibility.

These guard the integration surface (ARCHITECTURE.md §6, §10). If a contract
changes shape, these are the first thing that should break.
"""

from __future__ import annotations

import warnings

import pytest

from deepcoach.contracts import (
    SCHEMA_VERSION,
    BBox,
    ClassLabel,
    Detection,
    DetectionArtifact,
    DetectionFrame,
    Homography,
    HomographyArtifact,
    MetricsArtifact,
    PitchCoordinate,
    PitchLandmark,
    ProjectedEntry,
    ProjectedFrame,
    ProjectionArtifact,
    Role,
    TeamShape,
    TeamShapeFrame,
    Track,
    TrackArtifact,
    TrackedFrame,
)
from deepcoach.io.artifacts import load_artifact, now_utc_iso, save_artifact

HASH = "deadbeefcafe0000"


def _hdr() -> dict:
    return {"config_hash": HASH, "created_utc": now_utc_iso()}


def _detection_artifact() -> DetectionArtifact:
    return DetectionArtifact(
        **_hdr(),
        stage="s1_detect",
        fps=25.0,
        frame_count=1,
        frames=[
            DetectionFrame(
                frame_idx=0,
                timestamp_s=0.0,
                detections=[
                    Detection(frame_idx=0, bbox=BBox(x1=10, y1=20, x2=30, y2=80), cls=ClassLabel.player, confidence=0.9),
                    Detection(frame_idx=0, bbox=BBox(x1=300, y1=300, x2=310, y2=310), cls=ClassLabel.ball, confidence=0.4),
                ],
                source_meta={"frame_w": 1920, "frame_h": 1080},
            )
        ],
    )


def _track_artifact() -> TrackArtifact:
    return TrackArtifact(
        **_hdr(),
        stage="s3_track",
        fps=25.0,
        frame_count=1,
        n_tracks=1,
        frames=[
            TrackedFrame(
                frame_idx=0,
                timestamp_s=0.0,
                tracks=[
                    Track(track_id=7, frame_idx=0, bbox=BBox(x1=10, y1=20, x2=30, y2=80), cls=ClassLabel.player, confidence=0.9, team=0, role=Role.field),
                ],
            )
        ],
    )


def _homography_artifact() -> HomographyArtifact:
    return HomographyArtifact(
        **_hdr(),
        stage="s4_homography",
        static_camera=True,
        homographies=[
            Homography(
                matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                source_keyframe_idx=0,
                reprojection_error_px=2.5,
                pitch_landmarks_used=[
                    PitchLandmark(name="corner_tl", pixel_xy=(0, 0), pitch_xy=(0, 0)),
                    PitchLandmark(name="corner_tr", pixel_xy=(100, 0), pitch_xy=(105, 0)),
                ],
            )
        ],
    )


def _projection_artifact() -> ProjectionArtifact:
    return ProjectionArtifact(
        **_hdr(),
        stage="s5_project",
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        frames=[
            ProjectedFrame(
                frame_idx=0,
                timestamp_s=0.0,
                entries=[
                    ProjectedEntry(track_id=7, team=0, role=Role.field, cls=ClassLabel.player, pitch_xy=PitchCoordinate(x_m=52.5, y_m=34.0), projection_confidence=0.8),
                ],
            )
        ],
    )


def _metrics_artifact() -> MetricsArtifact:
    return MetricsArtifact(
        **_hdr(),
        stage="s6_metrics",
        metrics_enabled=["centroid", "compactness"],
        frames=[
            TeamShapeFrame(
                frame_idx=0,
                timestamp_s=0.0,
                teams=[
                    TeamShape(team=0, n_players=10, centroid=PitchCoordinate(x_m=40, y_m=34), compactness_m=12.3, width_m=45.0, depth_m=30.0, defensive_line_height_m=20.0),
                ],
                extra={"pitch_control": {"team0": 0.55}},
            )
        ],
    )


CASES = [
    (_detection_artifact, DetectionArtifact),
    (_track_artifact, TrackArtifact),
    (_homography_artifact, HomographyArtifact),
    (_projection_artifact, ProjectionArtifact),
    (_metrics_artifact, MetricsArtifact),
]


@pytest.mark.parametrize("factory,model_cls", CASES, ids=[c[1].__name__ for c in CASES])
def test_roundtrip(factory, model_cls, tmp_path):
    original = factory()
    path = tmp_path / "artifact.json"
    save_artifact(original, path)
    loaded = load_artifact(path, model_cls, expect_config_hash=HASH)
    assert loaded == original
    assert loaded.schema_version == SCHEMA_VERSION


def test_config_hash_mismatch_warns(tmp_path):
    path = tmp_path / "a.json"
    save_artifact(_detection_artifact(), path)
    with pytest.warns(UserWarning, match="stale"):
        load_artifact(path, DetectionArtifact, expect_config_hash="different00000000")


def test_major_schema_mismatch_raises(tmp_path):
    import json

    art = _detection_artifact()
    path = tmp_path / "a.json"
    save_artifact(art, path)
    data = json.loads(path.read_text())
    data["schema_version"] = "99.0"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incompatible"):
        load_artifact(path, DetectionArtifact)


def test_bbox_rejects_unordered():
    with pytest.raises(ValueError):
        BBox(x1=50, y1=10, x2=10, y2=80)  # x2 < x1


def test_bbox_ground_point_is_bottom_center():
    assert BBox(x1=10, y1=20, x2=30, y2=80).ground_point() == (20.0, 80.0)


def test_confidence_bounds():
    with pytest.raises(ValueError):
        Detection(frame_idx=0, bbox=BBox(x1=0, y1=0, x2=1, y2=1), cls=ClassLabel.player, confidence=1.5)


if __name__ == "__main__":
    warnings.simplefilter("error")
