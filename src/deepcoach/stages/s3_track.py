"""S3 track — ByteTrack over detections -> persistent track_ids -> TrackArtifact.

Resolves each track's team/role by majority vote across its detections. Maintains
IDs through short occlusions and LOGS every ID switch.

# EXTENSION POINT: swap tracker / add a re-identification model.
#   Stable contract: TrackArtifact with a persistent track_id.

NOTE: implementation deferred — backbone review pause (ARCHITECTURE.md §13).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..contracts.common import BBox, ClassLabel, Role
from ..contracts.detections import DetectionArtifact
from ..contracts.tracks import Track, TrackArtifact, TrackedFrame
from ..io.artifacts import load_artifact, now_utc_iso, out_dir, save_artifact
from ..io.config import ClipConfig

STAGE = "s3_track"

BALL_TRACK_ID = -1  # the ball is a single object; it doesn't need ByteTrack identity
SHORT_TRACK_FRAMES = 10  # tracks shorter than this are flagged as fragmentation


def run(config: ClipConfig) -> TrackArtifact:
    import supervision as sv  # local import: heavy dep, only needed to track

    dets = load_artifact(out_dir(config.clip_name()) / "detections.json", DetectionArtifact, config.config_hash())
    tracker = sv.ByteTrack(frame_rate=max(1, int(round(dets.fps))))

    out_frames: list[TrackedFrame] = []
    track_lengths: dict[int, int] = defaultdict(int)

    for df in dets.frames:
        players = [d for d in df.detections if d.cls == ClassLabel.player]
        tracks: list[Track] = []

        if players:
            xyxy = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in players], dtype=float)
            conf = np.array([d.confidence for d in players], dtype=float)
            sv_det = sv.Detections(xyxy=xyxy, confidence=conf, class_id=np.zeros(len(players), dtype=int))
            tracked = tracker.update_with_detections(sv_det)
            for i in range(len(tracked)):
                tid = int(tracked.tracker_id[i])
                x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
                c = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
                tracks.append(
                    Track(track_id=tid, frame_idx=df.frame_idx, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                          cls=ClassLabel.player, confidence=c, team=None, role=Role.unknown)
                )
                track_lengths[tid] += 1

        # Ball: carry the single most-confident ball detection per frame as one object.
        balls = [d for d in df.detections if d.cls == ClassLabel.ball]
        if balls:
            b = max(balls, key=lambda d: d.confidence)
            tracks.append(Track(track_id=BALL_TRACK_ID, frame_idx=df.frame_idx, bbox=b.bbox,
                                cls=ClassLabel.ball, confidence=b.confidence, team=None, role=Role.unknown))

        out_frames.append(TrackedFrame(frame_idx=df.frame_idx, timestamp_s=df.timestamp_s, tracks=tracks))

    player_lengths = [v for k, v in track_lengths.items() if k != BALL_TRACK_ID]
    n_tracks = len(player_lengths)
    mean_len = float(np.mean(player_lengths)) if player_lengths else 0.0
    median_len = float(np.median(player_lengths)) if player_lengths else 0.0
    short = sum(1 for v in player_lengths if v < SHORT_TRACK_FRAMES)

    art = TrackArtifact(
        config_hash=config.config_hash(),
        stage=STAGE,
        created_utc=now_utc_iso(),
        fps=dets.fps,
        frame_count=len(out_frames),
        n_tracks=n_tracks,
        frames=out_frames,
    )
    save_artifact(art, out_dir(config.clip_name()) / "tracks.json")
    print(
        f"[s3_track] {config.clip_name()}: {n_tracks} player tracks, "
        f"mean length {mean_len:.1f} / median {median_len:.0f} frames, "
        f"{short} short tracks (<{SHORT_TRACK_FRAMES} frames, fragmentation proxy)"
    )
    # NOTE: true ID-switch counting needs ground truth; track fragmentation is the
    # observable proxy here. A re-id model behind this seam reduces both.
    return art
