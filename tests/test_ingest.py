"""S0 ingest tests: IngestArtifact round-trips, and the scene-cut detector finds
hard cuts on synthetic frames (no real video file / codec needed)."""

from __future__ import annotations

import numpy as np

from deepcoach.contracts import IngestArtifact, SceneCut
from deepcoach.io.artifacts import load_artifact, now_utc_iso, save_artifact
from deepcoach.io.clip import Frame
from deepcoach.stages.s0_ingest import detect_scene_cuts

HASH = "0011223344556677"


def _solid(color_bgr, n_start, n) -> list[Frame]:
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    img[:, :] = color_bgr
    return [Frame(frame_idx=i, timestamp_s=i / 25.0, image=img.copy()) for i in range(n_start, n_start + n)]


def test_detects_single_hard_cut():
    # 10 red frames then 10 blue frames -> exactly one cut, at the boundary (idx 10).
    frames = _solid((0, 0, 255), 0, 10) + _solid((255, 0, 0), 10, 10)
    cuts = detect_scene_cuts(frames, threshold=0.5)
    assert len(cuts) == 1
    assert cuts[0].frame_idx == 10
    assert cuts[0].score >= 0.5


def test_no_cut_on_uniform_video():
    frames = _solid((0, 128, 0), 0, 20)
    assert detect_scene_cuts(frames, threshold=0.5) == []


def test_ingest_artifact_roundtrip(tmp_path):
    art = IngestArtifact(
        config_hash=HASH,
        stage="s0_ingest",
        created_utc=now_utc_iso(),
        source_kind="youtube",
        source_ref="https://youtube.com/watch?v=abc123",
        normalized_path="data/in/abc123.mp4",
        fps=25.0,
        width=1920,
        height=1080,
        frame_count=750,
        duration_s=30.0,
        detected_cuts=[SceneCut(frame_idx=120, timestamp_s=4.8, score=0.92)],
        n_shots=2,
        notes=["ffmpeg missing: copied source without transcoding"],
    )
    path = tmp_path / "ingest.json"
    save_artifact(art, path)
    loaded = load_artifact(path, IngestArtifact, expect_config_hash=HASH)
    assert loaded == art
    assert loaded.n_shots == 2
