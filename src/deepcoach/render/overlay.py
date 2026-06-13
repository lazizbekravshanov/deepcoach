"""overlay — annotated ORIGINAL video: boxes + team color + track_id.

The primary tool for eyeballing detection/tracking quality. Consumes TrackArtifact
+ the clip frames (via io/clip.py). Output: data/out/<name>/render/overlay.mp4.
"""

from __future__ import annotations

import cv2

from ..contracts.common import ClassLabel, Role
from ..contracts.tracks import TrackArtifact
from ..io.artifacts import load_artifact, out_dir
from ..io.clip import ClipReader
from ..io.config import ClipConfig

# BGR. Team 0 = blue, team 1 = red, ball = yellow, unknown/gk/ref = gray.
TEAM_BGR = {0: (255, 128, 0), 1: (0, 0, 255)}
BALL_BGR = (0, 255, 255)
UNKNOWN_BGR = (180, 180, 180)


def _color(team, cls, role):
    if cls == ClassLabel.ball:
        return BALL_BGR
    if role != Role.field or team is None:
        return UNKNOWN_BGR
    return TEAM_BGR.get(team, UNKNOWN_BGR)


def render(config: ClipConfig) -> str:
    od = out_dir(config.clip_name())
    tracks = load_artifact(od / "tracks.json", TrackArtifact, config.config_hash())
    by_frame = {tf.frame_idx: tf for tf in tracks.frames}

    render_dir = od / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    out_path = render_dir / "overlay.mp4"

    with ClipReader(config) as reader:
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), reader.fps, (reader.width, reader.height))
        for fr in reader:
            img = fr.image
            tf = by_frame.get(fr.frame_idx)
            if tf:
                for t in tf.tracks:
                    color = _color(t.team, t.cls, t.role)
                    p1 = (int(t.bbox.x1), int(t.bbox.y1))
                    p2 = (int(t.bbox.x2), int(t.bbox.y2))
                    cv2.rectangle(img, p1, p2, color, 2)
                    label = "ball" if t.cls == ClassLabel.ball else f"#{t.track_id}"
                    cv2.putText(img, label, (p1[0], p1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            writer.write(img)
        writer.release()

    print(f"[render.overlay] wrote {out_path}")
    return str(out_path)
