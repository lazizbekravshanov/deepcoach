"""deepcoach data contracts — the integration surface between stages.

This package has ZERO dependencies on stages/render/quality. Import contracts
from here; never import a stage to get at a shape.
"""

from .common import (
    SCHEMA_VERSION,
    ArtifactHeader,
    BBox,
    ClassLabel,
    Role,
)
from .detections import Detection, DetectionArtifact, DetectionFrame
from .ingest import IngestArtifact, SceneCut
from .metrics import MetricsArtifact, TeamShape, TeamShapeFrame
from .pitch import (
    Homography,
    HomographyArtifact,
    PitchCoordinate,
    PitchLandmark,
    ProjectedEntry,
    ProjectedFrame,
    ProjectionArtifact,
)
from .tracks import Track, TrackArtifact, TrackedFrame

__all__ = [
    "SCHEMA_VERSION",
    "ArtifactHeader",
    "BBox",
    "ClassLabel",
    "Role",
    "Detection",
    "DetectionFrame",
    "DetectionArtifact",
    "SceneCut",
    "IngestArtifact",
    "Track",
    "TrackedFrame",
    "TrackArtifact",
    "PitchLandmark",
    "Homography",
    "HomographyArtifact",
    "PitchCoordinate",
    "ProjectedEntry",
    "ProjectedFrame",
    "ProjectionArtifact",
    "TeamShape",
    "TeamShapeFrame",
    "MetricsArtifact",
]
