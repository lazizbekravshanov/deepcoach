"""pitchcontrol — aggregate territorial dominance map over the clip.

Consumes ProjectionArtifact. For each frame, partitions the pitch by nearest
player (Voronoi/nearest-neighbour) using only confidence-filtered field players,
then averages ownership across frames into one map: which team controls which
zones, on average. Output: data/out/<name>/render/pitch_control.png.

This is the headline spatial metric made visual, and it's per-frame /
position-only — robust to the track fragmentation broadcast footage causes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..contracts.common import ClassLabel, Role  # noqa: E402
from ..contracts.pitch import ProjectionArtifact  # noqa: E402
from ..io.artifacts import load_artifact, out_dir  # noqa: E402
from ..io.config import ClipConfig  # noqa: E402

STEP_M = 2.0
TEAM_RGB = {0: np.array([0.20, 0.40, 1.0]), 1: np.array([1.0, 0.25, 0.25])}


def render(config: ClipConfig) -> str | None:
    od = out_dir(config.clip_name())
    proj = load_artifact(od / "projected.json", ProjectionArtifact, config.config_hash())
    L, W = proj.pitch_length_m, proj.pitch_width_m
    min_conf = config.metrics.min_confidence

    xs = np.arange(STEP_M / 2.0, L, STEP_M)
    ys = np.arange(STEP_M / 2.0, W, STEP_M)
    gx, gy = np.meshgrid(xs, ys)
    cells = np.column_stack([gx.ravel(), gy.ravel()])
    shape = gx.shape  # (rows=y, cols=x)

    control = {0: np.zeros(len(cells)), 1: np.zeros(len(cells))}
    frames_used = 0
    for f in proj.frames:
        by_team: dict[int, list] = {0: [], 1: []}
        for e in f.entries:
            if e.cls == ClassLabel.player and e.role == Role.field and e.team in (0, 1):
                if e.projection_confidence >= min_conf:
                    by_team[e.team].append((e.pitch_xy.x_m, e.pitch_xy.y_m))
        if not by_team[0] and not by_team[1]:
            continue
        mind = np.full(len(cells), np.inf)
        owner = np.full(len(cells), -1)
        for t in (0, 1):
            if not by_team[t]:
                continue
            pts = np.array(by_team[t], dtype=float)
            d = np.min(np.linalg.norm(cells[:, None, :] - pts[None, :, :], axis=2), axis=1)
            closer = d < mind
            mind[closer] = d[closer]
            owner[closer] = t
        for t in (0, 1):
            control[t][owner == t] += 1
        frames_used += 1

    if frames_used == 0:
        print("[render.pitch_control] no usable frames (no confident team dots) — skipped")
        return None

    # per-cell dominance fraction for team 0 (1 - that ~ team 1); build RGB image
    tot = control[0] + control[1]
    frac0 = np.divide(control[0], tot, out=np.full_like(control[0], 0.5), where=tot > 0)
    frac0 = frac0.reshape(shape)
    img = np.zeros((*shape, 3))
    for i in range(shape[0]):
        for j in range(shape[1]):
            a = frac0[i, j]
            img[i, j] = a * TEAM_RGB[0] + (1 - a) * TEAM_RGB[1]

    overall0 = float(control[0].sum() / (tot.sum() or 1))
    fig, ax = plt.subplots(figsize=(L / 12, W / 12))
    ax.imshow(img, extent=[0, L, 0, W], origin="lower", aspect="equal")
    ax.add_patch(plt.Circle((L / 2, W / 2), 9.15, fill=False, color="white", lw=1.5))
    ax.axvline(L / 2, color="white", lw=1.5)
    ax.set_title(f"Territorial control  —  team0(blue) {overall0*100:.0f}% / team1(red) {(1-overall0)*100:.0f}%")
    ax.set_xlabel("x_m")
    ax.set_ylabel("y_m")
    out_path = od / "render" / "pitch_control.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[render.pitch_control] wrote {out_path} (avg control team0={overall0*100:.0f}%, {frames_used} frames)")
    return str(out_path)
