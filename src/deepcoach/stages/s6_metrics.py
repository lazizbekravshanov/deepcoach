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


# Frame-level metrics need ALL teams' positions at once (e.g. pitch control), so
# they have their own registry with signature fn(by_team: dict[int,ndarray], ctx).
_FRAME_REGISTRY: dict[str, Callable[..., object]] = {}


def register_frame(name: str) -> Callable[[Callable], Callable]:
    def _wrap(fn):
        if name in _FRAME_REGISTRY:
            raise ValueError(f"frame metric {name!r} already registered")
        _FRAME_REGISTRY[name] = fn
        return fn

    return _wrap


def get_frame_metric(name: str) -> Callable[..., object]:
    if name not in _FRAME_REGISTRY:
        raise KeyError(f"unknown frame metric {name!r}; registered: {sorted(_FRAME_REGISTRY)}")
    return _FRAME_REGISTRY[name]


def per_team_metrics() -> list[str]:
    return sorted(_REGISTRY)


def frame_metrics() -> list[str]:
    return sorted(_FRAME_REGISTRY)


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


@register("hull_area")
def hull_area_m2(points: np.ndarray, ctx: dict) -> float:
    """Space occupied by a team: area (m^2) of the convex hull of its players.

    Per-frame and position-only, so it survives track fragmentation. <3 players or
    a degenerate (collinear) set -> 0.0.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull

        return float(ConvexHull(pts).volume)  # 2D ConvexHull.volume == polygon area
    except Exception:
        return 0.0  # collinear / degenerate


@register_frame("pitch_control")
def pitch_control(by_team: dict, ctx: dict, step: float = 2.0) -> dict:
    """Territorial dominance: fraction of the pitch each team's players are nearest to.

    Grid-sample the pitch; each cell is controlled by the team owning the closest
    player (a nearest-neighbour / Voronoi partition). Per-frame, position-only —
    robust to the ID fragmentation that highlight footage causes. Returns
    {team_str: fraction}; fractions over teams sum to 1.
    """
    L, W = ctx.get("pitch_length", 105.0), ctx.get("pitch_width", 68.0)
    xs = np.arange(step / 2.0, L, step)
    ys = np.arange(step / 2.0, W, step)
    gx, gy = np.meshgrid(xs, ys)
    cells = np.column_stack([gx.ravel(), gy.ravel()])  # (M, 2)

    teams = [t for t in sorted(by_team) if len(by_team[t]) > 0]
    if not teams:
        return {}
    mind = np.full(len(cells), np.inf)
    owner = np.full(len(cells), -1)
    for t in teams:
        pts = np.asarray(by_team[t], dtype=float)
        d = np.min(np.linalg.norm(cells[:, None, :] - pts[None, :, :], axis=2), axis=1)
        closer = d < mind
        mind[closer] = d[closer]
        owner[closer] = t
    total = len(cells)
    return {str(t): float(np.sum(owner == t)) / total for t in teams}


def run(config: ClipConfig) -> MetricsArtifact:
    name = config.clip_name()
    od = out_dir(name)
    cfg_hash = config.config_hash()
    proj = load_artifact(od / "projected.json", ProjectionArtifact, cfg_hash)

    L, W = config.pitch.length_m, config.pitch.width_m
    min_conf = config.metrics.min_confidence
    own_goal = _infer_own_goal_x(proj, L)
    enabled = config.metrics.enabled
    team_extras = [m for m in enabled if m not in CORE_METRICS and m in _REGISTRY]  # e.g. hull_area
    frame_extras = [m for m in enabled if m in _FRAME_REGISTRY]  # e.g. pitch_control

    out_frames: list[TeamShapeFrame] = []
    thin = 0
    used = dropped = 0
    for f in proj.frames:
        by_team: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for e in f.entries:
            if e.cls == ClassLabel.player and e.role == Role.field and e.team is not None:
                if e.projection_confidence < min_conf:  # wrong-dot guard: skip untrusted dots
                    dropped += 1
                    continue
                by_team[e.team].append((e.pitch_xy.x_m, e.pitch_xy.y_m))
                used += 1

        team_pts = {t: np.array(v, dtype=float) for t, v in by_team.items() if v}
        shapes: list[TeamShape] = []
        extra: dict[str, object] = {}
        for team in sorted(team_pts):
            pts = team_pts[team]
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
            for m in team_extras:
                extra.setdefault(str(team), {})[m] = get_metric(m)(pts, ctx)  # type: ignore[index]

        fctx = {"pitch_length": L, "pitch_width": W}
        for m in frame_extras:
            extra[m] = get_frame_metric(m)(team_pts, fctx)

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
    total = used + dropped
    kept_pct = (100.0 * used / total) if total else 0.0
    print(
        f"[s6_metrics] {name}: {len(out_frames)} frames, metrics={enabled}, "
        f"inferred defending side per team={sides}, {thin} thin team-frames (<{THIN_FRAME_PLAYERS} players)"
    )
    print(
        f"[s6_metrics] confidence>={min_conf}: used {used}/{total} field-player dots "
        f"({kept_pct:.0f}%), dropped {dropped} as low-confidence"
    )
    if min_conf > 0 and kept_pct < 40:
        print("[s6_metrics] WARNING: most dots dropped as low-confidence — metrics rest on few players.")
    return art
