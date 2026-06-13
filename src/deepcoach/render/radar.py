"""radar — top-down minimap video: two team colors + ball on a drawn pitch.

Consumes ProjectionArtifact (pitch meters). Dots are drawn at low alpha when
projection_confidence is low — the wrong-dot principle made visible: uncertain
dots literally look faint. Output: data/out/<name>/render/radar.mp4.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..contracts.common import ClassLabel, Role
from ..contracts.pitch import ProjectionArtifact
from ..io.artifacts import load_artifact, out_dir
from ..io.config import ClipConfig

PX_PER_M = 10
TEAM_BGR = {0: (255, 128, 0), 1: (0, 0, 255)}
BALL_BGR = (0, 255, 255)
UNKNOWN_BGR = (180, 180, 180)
PITCH_GREEN = (40, 110, 40)


def _draw_pitch(length_m: float, width_m: float) -> np.ndarray:
    h, w = int(width_m * PX_PER_M), int(length_m * PX_PER_M)
    img = np.full((h, w, 3), PITCH_GREEN, dtype=np.uint8)
    white = (220, 220, 220)
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), white, 2)
    cv2.line(img, (w // 2, 0), (w // 2, h), white, 2)
    cv2.circle(img, (w // 2, h // 2), int(9.15 * PX_PER_M), white, 2)
    return img


def render(config: ClipConfig) -> str:
    od = out_dir(config.clip_name())
    proj = load_artifact(od / "projected.json", ProjectionArtifact, config.config_hash())
    L, W = proj.pitch_length_m, proj.pitch_width_m
    base = _draw_pitch(L, W)
    h, w = base.shape[:2]

    render_dir = od / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    out_path = render_dir / "radar.mp4"

    # fps: derive from timestamps if possible, else default 25.
    fps = 25.0
    if len(proj.frames) >= 2:
        dt = proj.frames[1].timestamp_s - proj.frames[0].timestamp_s
        if dt > 0:
            fps = 1.0 / dt

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in proj.frames:
        frame = base.copy()
        for e in f.entries:
            px = int(np.clip(e.pitch_xy.x_m * PX_PER_M, 0, w - 1))
            py = int(np.clip(e.pitch_xy.y_m * PX_PER_M, 0, h - 1))
            if e.cls == ClassLabel.ball:
                color = BALL_BGR
            elif e.role == Role.field and e.team is not None:
                color = TEAM_BGR.get(e.team, UNKNOWN_BGR)
            else:
                color = UNKNOWN_BGR
            # low confidence -> faint dot (blend toward pitch color)
            a = max(0.15, min(1.0, e.projection_confidence))
            dot = tuple(int(a * c + (1 - a) * g) for c, g in zip(color, PITCH_GREEN))
            cv2.circle(frame, (px, py), 6, dot, -1)
        writer.write(frame)
    writer.release()
    print(f"[render.radar] wrote {out_path} ({len(proj.frames)} frames @ {fps:.1f}fps)")
    return str(out_path)
