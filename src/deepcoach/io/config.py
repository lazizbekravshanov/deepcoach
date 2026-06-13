"""Per-clip config: load + validate + hash.

Adding a new clip is a config change, never a code change (ARCHITECTURE.md §9).
The validated `ClipConfig` is the single source of clip-specific truth, and its
`config_hash` is stamped into every artifact so output is traceable to its config.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ..contracts.pitch import PitchLandmark


class SourceSpec(BaseModel):
    """What S0 ingest pulls from. `path` (below) is where it WRITES the result."""

    kind: str = "file"  # "file" | "youtube"
    ref: str = ""  # local file path or YouTube URL


class ClipSpec(BaseModel):
    path: str  # normalized clip location: S0 ingest output == S1 detect input
    source: SourceSpec | None = None  # optional; if absent, `path` is taken as-is
    fps_override: float | None = None
    frame_range: tuple[int, int] | None = None  # [start, end) frame indices


class IngestSpec(BaseModel):
    detect_cuts: bool = True  # scan for scene cuts (highlight-reel detector)
    cut_threshold: float = 0.5  # Bhattacharyya distance >= this = a cut (0..1)
    scan_step: int = 1  # decimate frames during the cut scan for speed
    normalize: bool = True  # transcode to H.264 mp4 via ffmpeg when available
    max_height: int | None = None  # cap download resolution (e.g. 720); None = best mp4


class PitchSpec(BaseModel):
    length_m: float = 105.0
    width_m: float = 68.0


class ModelSpec(BaseModel):
    weights: str = "yolov8n.pt"
    conf_threshold: float = Field(0.25, ge=0.0, le=1.0)
    device: str = "mps"  # mps | cpu | cuda


class TeamsSpec(BaseModel):
    n_teams: int = 2
    sample_frames: list[int] = Field(default_factory=list)  # frames to learn colors from


class HomographySpec(BaseModel):
    mode: str = "manual"  # EXTENSION POINT: "auto" -> pitch-line detection
    keyframe_idx: int = 0
    landmarks: list[PitchLandmark] = Field(default_factory=list)


class MetricsSpec(BaseModel):
    enabled: list[str] = Field(
        default_factory=lambda: ["centroid", "compactness", "width", "depth", "def_line"]
    )
    # Only dots with projection_confidence >= this feed the metrics (wrong-dot guard).
    # 0.0 = use everything. Raise it on shaky footage so untrustworthy dots are excluded.
    min_confidence: float = 0.0


class RenderSpec(BaseModel):
    radar: bool = True
    overlay: bool = True
    heatmap: bool = True
    pitch_control: bool = True  # aggregate territorial dominance map


class ClipConfig(BaseModel):
    clip: ClipSpec
    ingest: IngestSpec = Field(default_factory=IngestSpec)
    pitch: PitchSpec = Field(default_factory=PitchSpec)
    model: ModelSpec = Field(default_factory=ModelSpec)
    teams: TeamsSpec = Field(default_factory=TeamsSpec)
    homography: HomographySpec = Field(default_factory=HomographySpec)
    metrics: MetricsSpec = Field(default_factory=MetricsSpec)
    render: RenderSpec = Field(default_factory=RenderSpec)

    def config_hash(self) -> str:
        """sha256[:16] of the canonical (sorted-key) JSON serialization.

        Stable across key ordering and formatting, so the same logical config
        always yields the same hash.
        """
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def clip_name(self) -> str:
        """Stable short name used for the output directory (data/out/<name>/)."""
        return Path(self.clip.path).stem


def load_config(path: str | Path) -> ClipConfig:
    """Load and validate a per-clip YAML config."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"empty config file: {path}")
    return ClipConfig.model_validate(raw)
