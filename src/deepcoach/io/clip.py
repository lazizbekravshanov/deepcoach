"""The ONLY video reader in deepcoach.

Every stage that needs frames goes through `iter_frames` / `ClipReader`. No other
module opens a `VideoCapture`. This keeps frame indexing, fps handling, and
frame-range slicing consistent across the whole pipeline.

v1 assumes a single continuous shot with no scene cuts (ARCHITECTURE.md §3).
A future broadcast pre-stage that segments shots feeds the SAME clip contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import ClipConfig

try:  # OpenCV is only needed when actually reading video, not for contract tests.
    import cv2
except ImportError:  # pragma: no cover - import guarded so contracts import without cv2
    cv2 = None  # type: ignore


@dataclass
class Frame:
    frame_idx: int
    timestamp_s: float
    image: np.ndarray  # BGR HxWx3, pixel space


class ClipReader:
    """Iterate decoded frames of a clip, honoring fps_override and frame_range."""

    def __init__(self, config: ClipConfig):
        if cv2 is None:
            raise ImportError("opencv-python is required to read video frames")
        self.config = config
        self.path = Path(config.clip.path)
        if not self.path.exists():
            raise FileNotFoundError(f"clip not found: {self.path}")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"could not open clip: {self.path}")
        native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.fps = config.clip.fps_override or native_fps
        if not self.fps or self.fps <= 0:
            raise ValueError(f"could not determine fps for {self.path}; set clip.fps_override")
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _range(self) -> tuple[int, int]:
        start, end = 0, self.frame_count
        if self.config.clip.frame_range is not None:
            start, end = self.config.clip.frame_range
        return start, end

    def __iter__(self) -> Iterator[Frame]:
        start, end = self._range()
        if start > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        while idx < end:
            ok, img = self._cap.read()
            if not ok:
                break
            yield Frame(frame_idx=idx, timestamp_s=idx / self.fps, image=img)
            idx += 1

    def read_frame(self, frame_idx: int) -> Frame:
        """Random-access read of a single frame (used by S2 sample_frames, S4 keyframe)."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = self._cap.read()
        if not ok:
            raise IOError(f"could not read frame {frame_idx} of {self.path}")
        return Frame(frame_idx=frame_idx, timestamp_s=frame_idx / self.fps, image=img)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "ClipReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def iter_frames(config: ClipConfig) -> Iterator[Frame]:
    """Convenience generator: open, yield frames, close."""
    with ClipReader(config) as reader:
        yield from reader


# --- raw-path helpers (used by S0 ingest, which works on sources upstream of the
#     clip contract). All VideoCapture usage lives in this module on purpose. ---


def probe_video(path: str | Path) -> dict:
    """Read basic metadata from any video file without decoding all of it."""
    if cv2 is None:
        raise ImportError("opencv-python is required to probe video")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {
            "fps": fps,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": frame_count,
            "duration_s": (frame_count / fps) if fps else 0.0,
        }
    finally:
        cap.release()


def iter_video_frames(
    path: str | Path, fps_override: float | None = None, step: int = 1
) -> Iterator[Frame]:
    """Iterate frames of an arbitrary video path (no ClipConfig needed).

    `step` decimates frames (every Nth) — handy for fast scene-cut scans.
    """
    if cv2 is None:
        raise ImportError("opencv-python is required to read video frames")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"could not open video: {path}")
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 0.0
    try:
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if idx % step == 0:
                ts = (idx / fps) if fps else 0.0
                yield Frame(frame_idx=idx, timestamp_s=ts, image=img)
            idx += 1
    finally:
        cap.release()
