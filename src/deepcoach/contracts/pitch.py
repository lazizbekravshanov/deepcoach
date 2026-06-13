"""S4 (homography) and S5 (projection) contracts.

The Homography contract is the seam behind which automatic pitch-line detection
will one day replace the manual picker — the auto version emits the identical
HomographyArtifact and S5 never knows the difference (ARCHITECTURE.md §8).

Pitch coordinates are in METERS, origin at one corner (ARCHITECTURE.md §5.2).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .common import ArtifactHeader, ClassLabel, Role


class PitchLandmark(BaseModel):
    """A pixel<->meters correspondence used to solve the homography."""

    name: str  # "corner_tl", "center_spot", "box_tl", "halfway_top", ...
    pixel_xy: tuple[float, float]  # clicked location, pixels
    pitch_xy: tuple[float, float]  # known real-world location, meters


class Homography(BaseModel):
    matrix: list[list[float]] = Field(..., description="3x3 pixel->pitch(meters)")
    source_keyframe_idx: int
    reprojection_error_px: float  # RMS reprojection error over the landmarks
    pitch_landmarks_used: list[PitchLandmark]

    @property
    def n_landmarks(self) -> int:
        return len(self.pitch_landmarks_used)


class HomographyArtifact(ArtifactHeader):
    # Static-camera v1 = a single homography. The list leaves room for a
    # per-keyframe set if the camera pans (see ARCHITECTURE.md §7.4).
    homographies: list[Homography]
    static_camera: bool = True


class PitchCoordinate(BaseModel):
    x_m: float  # meters, 0..pitch.length_m, along pitch length
    y_m: float  # meters, 0..pitch.width_m, along pitch width


class ProjectedEntry(BaseModel):
    track_id: int
    team: int | None
    role: Role
    cls: ClassLabel
    pitch_xy: PitchCoordinate  # meters
    # 0..1; decays with homography error and distance from landmark hull.
    # A wrong dot is worse than a missing one — this is how we surface doubt.
    projection_confidence: float = Field(..., ge=0.0, le=1.0)


class ProjectedFrame(BaseModel):
    frame_idx: int
    timestamp_s: float
    entries: list[ProjectedEntry]


class ProjectionArtifact(ArtifactHeader):
    pitch_length_m: float
    pitch_width_m: float
    frames: list[ProjectedFrame]
