"""S1 output contract: per-frame raw detections.

This is the integration surface for the detector. A better/fine-tuned detector
swaps in behind S1 and must emit exactly this shape (ARCHITECTURE.md §8).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .common import ArtifactHeader, BBox, ClassLabel


class Detection(BaseModel):
    frame_idx: int
    bbox: BBox  # pixels
    cls: ClassLabel
    confidence: float = Field(..., ge=0.0, le=1.0)


class DetectionFrame(BaseModel):
    frame_idx: int
    timestamp_s: float
    detections: list[Detection]
    source_meta: dict = Field(default_factory=dict)  # e.g. {"frame_w":.., "frame_h":..}


class DetectionArtifact(ArtifactHeader):
    fps: float
    frame_count: int
    frames: list[DetectionFrame]
