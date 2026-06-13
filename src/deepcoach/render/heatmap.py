"""heatmap — per-player occupancy heatmaps over the clip (pitch coordinates).

Consumes ProjectionArtifact. One PNG per player track with enough projected
points; low-confidence points are down-weighted so unreliable dots don't fabricate
hot zones. Output: data/out/<name>/render/heatmaps/track_<id>.png.

STRETCH: a registered pitch_control (Voronoi) map is a renderer over the same
projected contract, not a special case.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..contracts.common import ClassLabel  # noqa: E402
from ..contracts.pitch import ProjectionArtifact  # noqa: E402
from ..io.artifacts import load_artifact, out_dir  # noqa: E402
from ..io.config import ClipConfig  # noqa: E402

MIN_POINTS = 20  # tracks with fewer projected points are skipped (not enough to map)


def render(config: ClipConfig) -> list[str]:
    od = out_dir(config.clip_name())
    proj = load_artifact(od / "projected.json", ProjectionArtifact, config.config_hash())
    L, W = proj.pitch_length_m, proj.pitch_width_m

    xs: dict[int, list] = defaultdict(list)
    ys: dict[int, list] = defaultdict(list)
    wts: dict[int, list] = defaultdict(list)
    for f in proj.frames:
        for e in f.entries:
            if e.cls != ClassLabel.player:
                continue
            xs[e.track_id].append(e.pitch_xy.x_m)
            ys[e.track_id].append(e.pitch_xy.y_m)
            wts[e.track_id].append(e.projection_confidence)

    hm_dir = od / "render" / "heatmaps"
    hm_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for tid in sorted(xs):
        if len(xs[tid]) < MIN_POINTS:
            continue
        fig, ax = plt.subplots(figsize=(L / 20, W / 20))
        ax.hist2d(xs[tid], ys[tid], bins=[int(L / 3), int(W / 3)], range=[[0, L], [0, W]],
                  weights=wts[tid], cmap="hot")
        ax.set_xlim(0, L)
        ax.set_ylim(0, W)
        ax.set_aspect("equal")
        ax.set_title(f"track #{tid} occupancy")
        ax.set_xlabel("x_m")
        ax.set_ylabel("y_m")
        out_path = hm_dir / f"track_{tid}.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        written.append(str(out_path))

    print(f"[render.heatmap] wrote {len(written)} per-player heatmaps to {hm_dir}")
    return written
