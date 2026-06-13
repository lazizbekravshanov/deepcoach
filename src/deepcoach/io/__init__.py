"""I/O layer: the only video reader, typed artifact load/save, and config loading."""

from .artifacts import load_artifact, now_utc_iso, out_dir, save_artifact
from .clip import ClipReader, Frame, iter_frames
from .config import ClipConfig, load_config

__all__ = [
    "load_artifact",
    "save_artifact",
    "out_dir",
    "now_utc_iso",
    "ClipReader",
    "Frame",
    "iter_frames",
    "ClipConfig",
    "load_config",
]
