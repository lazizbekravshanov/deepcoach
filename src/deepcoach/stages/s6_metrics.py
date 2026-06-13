"""S6 metrics — ProjectionArtifact -> TeamShapeFrame[] via a metric REGISTRY.

The registry IS the extension seam: a new metric is added by registering a
function under a name and enabling it in config.metrics.enabled — never by editing
the pipeline. v1 will register centroid / compactness / width / depth / def_line.
Later: pitch_control (Voronoi) and, far further out, event metrics.

# EXTENSION POINT: register new metrics here (see register() below).
#   Stable contract: a metric is fn(team_points, ctx) -> value; team-level results
#   compose TeamShape; non-core results land in TeamShapeFrame.extra.

The registry scaffold below is part of the backbone (the seam must exist to be
reviewable). The metric FUNCTIONS and the run() driver are implemented after the
backbone review pause (ARCHITECTURE.md §13).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np

from ..contracts.common import ClassLabel, Role
from ..contracts.metrics import MetricsArtifact, TeamShape, TeamShapeFrame
from ..contracts.pitch import PitchCoordinate, ProjectionArtifact
from ..io.artifacts import load_artifact, now_utc_iso, out_dir, save_artifact
from ..io.config import ClipConfig

STAGE = "s6_metrics"

CORE_METRICS = {"centroid", "compactness", "width", "depth", "def_line"}
# Below this many field players, team-shape numbers are weakly supported.
THIN_FRAME_PLAYERS = 5

# name -> metric function. A metric receives the field-player pitch points for one
# team (and a context dict for pitch dims / both teams) and returns a value.
MetricFn = Callable[..., object]
_REGISTRY: dict[str, MetricFn] = {}


def register(name: str) -> Callable[[MetricFn], MetricFn]:
    """Decorator to register a metric under `name`.

    Usage (post-pause):
        @register("compactness")
        def _compactness(team_points, ctx): ...
    """

    def _wrap(fn: MetricFn) -> MetricFn:
        if name in _REGISTRY:
            raise ValueError(f"metric {name!r} already registered")
        _REGISTRY[name] = fn
        return fn

    return _wrap


def get_metric(name: str) -> MetricFn:
    if name not in _REGISTRY:
        raise KeyError(f"unknown metric {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered_metrics() -> list[str]:
    return sorted(_REGISTRY)


# --- v1 metrics. Each is a pure fn(points_xy: (N,2) meters, ctx: dict) -> value.
#     Field players only (gk/ref are excluded by the caller). New metrics attach
#     by adding a function + @register here; no pipeline edit needed. ---


@register("centroid")
def centroid(points: np.ndarray, ctx: dict) -> tuple[float, float]:
    """Team center of mass on the pitch (meters)."""
    c = np.asarray(points, dtype=float).mean(axis=0)
    return (float(c[0]), float(c[1]))


@register("compactness")
def compactness(points: np.ndarray, ctx: dict) -> float:
    """Spread: mean distance of players to the team centroid (meters). Lower = tighter."""
    pts = np.asarray(points, dtype=float)
    c = pts.mean(axis=0)
    return float(np.linalg.norm(pts - c, axis=1).mean())


@register("width")
def width(points: np.ndarray, ctx: dict) -> float:
    """Touchline-to-touchline extent: span along y_m (meters)."""
    ys = np.asarray(points, dtype=float)[:, 1]
    return float(ys.max() - ys.min())


@register("depth")
def depth(points: np.ndarray, ctx: dict) -> float:
    """Goal-to-goal extent: span along x_m (meters)."""
    xs = np.asarray(points, dtype=float)[:, 0]
    return float(xs.max() - xs.min())


@register("def_line")
def def_line(points: np.ndarray, ctx: dict) -> float:
    """Defensive line height: distance (m) of the rearmost outfield player from the
    team's OWN goal. Needs to know which goal the team defends.

    ctx: own_goal_x in {0, pitch_length}, pitch_length. Higher value = line pushed
    further up the pitch. The own_goal_x is supplied by run() (inferred per team);
    inferring/overriding the defending side is an EXTENSION POINT.
    """
    xs = np.asarray(points, dtype=float)[:, 0]
    own_goal_x = ctx.get("own_goal_x", 0.0)
    pitch_length = ctx.get("pitch_length", 105.0)
    if own_goal_x == 0.0:
        return float(xs.min())
    return float(pitch_length - xs.max())


def _infer_own_goal_x(proj: ProjectionArtifact, pitch_length: float) -> dict[int, float]:
    """Heuristic: a team defends the goal nearer to its average position over the clip.

    Honest about its weakness — this is a coarse guess for a short clip and is an
    EXTENSION POINT (infer from GK position, or override in config). The choice is
    surfaced in the quality output so a wrong guess is visible, not silent.
    """
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for f in proj.frames:
        for e in f.entries:
            if e.cls == ClassLabel.player and e.role == Role.field and e.team is not None:
                sums[e.team] += e.pitch_xy.x_m
                counts[e.team] += 1
    own: dict[int, float] = {}
    for team, n in counts.items():
        mean_x = sums[team] / n
        own[team] = 0.0 if mean_x < pitch_length / 2.0 else pitch_length
    return own


def run(config: ClipConfig) -> MetricsArtifact:
    name = config.clip_name()
    od = out_dir(name)
    cfg_hash = config.config_hash()
    proj = load_artifact(od / "projected.json", ProjectionArtifact, cfg_hash)

    L, W = config.pitch.length_m, config.pitch.width_m
    own_goal = _infer_own_goal_x(proj, L)
    enabled = config.metrics.enabled
    extra_metrics = [m for m in enabled if m not in CORE_METRICS]  # e.g. pitch_control later

    out_frames: list[TeamShapeFrame] = []
    thin = 0
    for f in proj.frames:
        by_team: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for e in f.entries:
            if e.cls == ClassLabel.player and e.role == Role.field and e.team is not None:
                by_team[e.team].append((e.pitch_xy.x_m, e.pitch_xy.y_m))

        shapes: list[TeamShape] = []
        extra: dict[str, object] = {}
        for team in sorted(by_team):
            pts = np.array(by_team[team], dtype=float)
            if len(pts) < THIN_FRAME_PLAYERS:
                thin += 1
            ctx = {"own_goal_x": own_goal.get(team, 0.0), "pitch_length": L, "pitch_width": W}
            cx, cy = centroid(pts, ctx)
            shapes.append(
                TeamShape(
                    team=team,
                    n_players=len(pts),
                    centroid=PitchCoordinate(x_m=cx, y_m=cy),
                    compactness_m=compactness(pts, ctx),
                    width_m=width(pts, ctx),
                    depth_m=depth(pts, ctx),
                    defensive_line_height_m=def_line(pts, ctx),
                )
            )
            for m in extra_metrics:
                extra.setdefault(str(team), {})[m] = get_metric(m)(pts, ctx)  # type: ignore[index]

        out_frames.append(
            TeamShapeFrame(frame_idx=f.frame_idx, timestamp_s=f.timestamp_s, teams=shapes, extra=extra)
        )

    art = MetricsArtifact(
        config_hash=cfg_hash,
        stage=STAGE,
        created_utc=now_utc_iso(),
        metrics_enabled=enabled,
        frames=out_frames,
    )
    save_artifact(art, od / "metrics.json")
    sides = {t: ("left" if x == 0.0 else "right") for t, x in own_goal.items()}
    print(
        f"[s6_metrics] {name}: {len(out_frames)} frames, metrics={enabled}, "
        f"inferred defending side per team={sides}, {thin} thin team-frames (<{THIN_FRAME_PLAYERS} players)"
    )
    return art
