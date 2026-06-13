"""Typed artifact load/save with schema versioning.

JSON for v1 — human-readable and diffable while we're still eyeballing trust.
Parquet is a later swap for the large per-frame artifacts; the contract is
unchanged (ARCHITECTURE.md §10).

A loader refuses a MAJOR schema_version mismatch (the shape may have changed
incompatibly) and warns on a config_hash mismatch (output is stale vs. the
current config).
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from ..contracts.common import SCHEMA_VERSION, ArtifactHeader

T = TypeVar("T", bound=BaseModel)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def out_dir(clip_name: str, base: str | Path = "data/out") -> Path:
    """Output directory for a clip's artifacts; created if missing."""
    d = Path(base) / clip_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _major(version: str) -> str:
    return version.split(".", 1)[0]


def save_artifact(model: BaseModel, path: str | Path) -> Path:
    """Serialize a contract model to JSON on disk (pretty-printed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(model.model_dump(mode="json"), f, indent=2)
    return path


def load_artifact(path: str | Path, model_cls: type[T], expect_config_hash: str | None = None) -> T:
    """Load + validate an artifact, enforcing schema/config compatibility.

    - Raises on a major schema_version mismatch (incompatible shape).
    - Warns (does not raise) on a config_hash mismatch, if `expect_config_hash`
      is given — the artifact is stale relative to the current config.
    """
    path = Path(path)
    with open(path, "r") as f:
        raw = json.load(f)

    if issubclass(model_cls, ArtifactHeader):
        found = str(raw.get("schema_version", ""))
        if found and _major(found) != _major(SCHEMA_VERSION):
            raise ValueError(
                f"{path}: schema_version {found!r} incompatible with current {SCHEMA_VERSION!r}"
            )
        if expect_config_hash is not None and raw.get("config_hash") != expect_config_hash:
            warnings.warn(
                f"{path}: config_hash {raw.get('config_hash')!r} != current "
                f"{expect_config_hash!r} — artifact is stale; re-run the producing stage.",
                stacklevel=2,
            )

    return model_cls.model_validate(raw)
