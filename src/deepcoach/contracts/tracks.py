"""S2+S3 output contract: persistent tracks enriched with team and role.

A tracker swap (e.g. with a re-id model) must preserve this shape; downstream
code depends on `track_id` being persistent and `team`/`role` being present
(ARCHITECTURE.md §8).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .common import ArtifactHeader, BBox, ClassLabel, Role


class Track(BaseModel):
    track_id: int
    frame_idx: int
    bbox: BBox  # pixels
    cls: ClassLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    team: int | None = None  # None until S2 assigns; None for ball / referee
    role: Role = Role.unknown


class TrackedFrame(BaseModel):
    frame_idx: int
    timestamp_s: float
    tracks: list[Track]


class TrackArtifact(ArtifactHeader):
    fps: float
    frame_count: int
    n_tracks: int
    frames: list[TrackedFrame]
