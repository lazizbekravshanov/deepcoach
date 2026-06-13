"""Shared contract primitives.

`contracts/` is the neutral middle of the pipeline. It has ZERO dependencies on
`stages/`, `render/`, or `quality/`. Everything else depends on these shapes;
these shapes depend on nothing internal.

Coordinate frames (see ARCHITECTURE.md §5):
- Pixel space: origin top-left, x right, y down, unit = pixels.
- Pitch space: origin one corner, x_m along length, y_m along width, unit = meters.
A single field never mixes the two; the unit is named in the field.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

# Bumped when any contract changes shape. Loaders refuse a major-version mismatch.
SCHEMA_VERSION = "1.0"


class ClassLabel(str, Enum):
    """What the detector (S1) can emit."""

    player = "player"
    ball = "ball"


class Role(str, Enum):
    """Refined by S2. Field player, goalkeeper, referee, or not-yet-known.

    GK and referee are explicit edge cases that must NOT be silently forced into
    an outfield team cluster (see ARCHITECTURE.md §11).
    """

    field = "field"
    gk = "gk"
    ref = "ref"
    unknown = "unknown"


class BBox(BaseModel):
    """Axis-aligned bounding box in PIXEL space, xyxy convention."""

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _ordered(self) -> "BBox":
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"bbox not ordered: ({self.x1},{self.y1})->({self.x2},{self.y2})")
        return self

    def ground_point(self) -> tuple[float, float]:
        """Bottom-center of the box in pixels — where the player meets the ground.

        This (not the centroid) is the point S5 projects to the pitch; the
        centroid would float above the pitch plane and project incorrectly.
        """
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class ArtifactHeader(BaseModel):
    """Mixed into every top-level *Artifact envelope.

    Makes every artifact traceable to the schema and config that produced it.
    """

    schema_version: str = SCHEMA_VERSION
    config_hash: str = Field(..., description="sha256[:16] of the canonical config")
    stage: str = Field(..., description="emitting stage, e.g. 's1_detect'")
    created_utc: str = Field(..., description="ISO-8601 UTC timestamp")
