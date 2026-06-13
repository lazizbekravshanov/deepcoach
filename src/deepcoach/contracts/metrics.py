"""S6 output contract: per-frame team shape & space metrics.

New metrics are added by registering a function in the S6 registry (not by
editing this contract). Registered extras land in `TeamShapeFrame.extra` so the
core shape fields stay stable while the metric set grows — including, eventually,
pitch control (Voronoi) and event metrics (ARCHITECTURE.md §7.6, §8).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .common import ArtifactHeader
from .pitch import PitchCoordinate


class TeamShape(BaseModel):
    team: int
    n_players: int  # field players actually used (gk/ref excluded) — exposes thin frames
    centroid: PitchCoordinate
    compactness_m: float  # spread: mean distance of players to centroid (m)
    width_m: float  # span along y_m (touchline-to-touchline extent)
    depth_m: float  # span along x_m (goal-to-goal extent)
    defensive_line_height_m: float  # x_m of the rearmost outfield line


class TeamShapeFrame(BaseModel):
    frame_idx: int
    timestamp_s: float
    teams: list[TeamShape]
    # EXTENSION POINT: registered metrics (e.g. pitch_control) land here keyed by name.
    extra: dict[str, object] = Field(default_factory=dict)


class MetricsArtifact(ArtifactHeader):
    metrics_enabled: list[str]
    frames: list[TeamShapeFrame]
