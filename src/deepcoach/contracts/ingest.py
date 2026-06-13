"""S0 ingest contract: how a raw source becomes a normalized single clip + an
inspection report.

S0 sits UPSTREAM of the clip contract: it accepts a local file or a YouTube URL,
normalizes it into data/in/<name>.mp4, and probes it. The inspection report's
`detected_cuts` is how we judge whether a video is a single continuous shot (v1's
core assumption) or an edited reel that needs segmenting — the wrong-dot principle
applied at the door (ARCHITECTURE.md §7.0, §11).

This is the same seam the future broadcast shot-segmenter plugs into.
"""

from __future__ import annotations

from pydantic import BaseModel

from .common import ArtifactHeader


class SceneCut(BaseModel):
    """A detected hard cut between consecutive shots."""

    frame_idx: int  # first frame of the new shot
    timestamp_s: float
    score: float  # dissimilarity at the boundary (0 identical .. 1 fully different)


class IngestArtifact(ArtifactHeader):
    source_kind: str  # "file" | "youtube"
    source_ref: str  # local path or URL the clip came from
    normalized_path: str  # where the usable clip now lives (data/in/<name>.mp4)
    fps: float
    width: int
    height: int
    frame_count: int
    duration_s: float
    detected_cuts: list[SceneCut]
    n_shots: int  # len(detected_cuts) + 1 — how many continuous shots the source holds
    notes: list[str]  # warnings: ffmpeg missing, normalization skipped, etc.
