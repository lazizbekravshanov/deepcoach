"""S2 teams — k-means jersey-color clustering -> per-player team + role.

Reads DetectionArtifact + clip sample frames; attaches provisional team/role
carried forward to S3. GK and referee are explicit edge cases: flagged as
role=gk/ref/unknown, never silently forced into an outfield cluster.

# EXTENSION POINT: swap team-assignment strategy.
#   v1 = k-means on jersey color. Future = jersey-classifier CNN / temporal voting.
#   Stable contract: Track.team / Track.role.

NOTE: implementation deferred — backbone review pause (ARCHITECTURE.md §13).
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from ..contracts.common import ClassLabel, Role
from ..contracts.tracks import TrackArtifact
from ..io.artifacts import load_artifact, now_utc_iso, out_dir, save_artifact
from ..io.clip import ClipReader
from ..io.config import ClipConfig

STAGE = "s2_teams"

SAMPLES_PER_TRACK = 5  # crops sampled across a track's life to build its color feature
OUTLIER_SIGMA = 2.0  # >this many std from its team center => flagged (gk/ref/unknown)


def jersey_feature(image: np.ndarray, bbox) -> np.ndarray:
    """Mean HSV of the torso region of a player bbox. Pure; the team-color signal.

    Torso = the central upper band of the box, avoiding head, shorts, and grass at
    the edges. A jersey-classifier CNN would replace this behind the S2 seam.
    """
    h, w = image.shape[:2]
    x1, x2 = max(0, int(bbox.x1)), min(w, int(bbox.x2))
    y1, y2 = max(0, int(bbox.y1)), min(h, int(bbox.y2))
    bh, bw = y2 - y1, x2 - x1
    if bh <= 2 or bw <= 2:
        return np.zeros(3, dtype=float)
    crop = image[y1 + int(0.20 * bh): y1 + int(0.55 * bh), x1 + int(0.20 * bw): x1 + int(0.80 * bw)]
    if crop.size == 0:
        return np.zeros(3, dtype=float)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    return hsv.mean(axis=0)


def run(config: ClipConfig) -> TrackArtifact:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    od = out_dir(config.clip_name())
    tracks = load_artifact(od / "tracks.json", TrackArtifact, config.config_hash())

    # Per-track player appearances, then sample frames evenly across each track's life.
    appearances: dict[int, list] = defaultdict(list)
    for tf in tracks.frames:
        for t in tf.tracks:
            if t.cls == ClassLabel.player:
                appearances[t.track_id].append((tf.frame_idx, t.bbox))

    need: dict[int, list] = defaultdict(list)
    for tid, apps in appearances.items():
        sel = np.unique(np.linspace(0, len(apps) - 1, min(SAMPLES_PER_TRACK, len(apps))).astype(int))
        for j in sel:
            fidx, bbox = apps[j]
            need[fidx].append((tid, bbox))

    feats: dict[int, list] = defaultdict(list)
    with ClipReader(config) as reader:
        for fidx in sorted(need):
            img = reader.read_frame(fidx).image
            for tid, bbox in need[fidx]:
                feats[tid].append(jersey_feature(img, bbox))

    track_ids = [t for t in feats if len(feats[t]) > 0]
    n_teams = config.teams.n_teams
    team_of: dict[int, int | None] = {}
    role_of: dict[int, Role] = {}
    silhouette = 0.0
    flagged = 0

    if len(track_ids) < n_teams:
        print(f"[s2_teams] WARNING: only {len(track_ids)} tracks < n_teams={n_teams}; leaving teams unassigned")
        for tid in track_ids:
            team_of[tid], role_of[tid] = None, Role.unknown
    else:
        X = np.array([np.mean(feats[t], axis=0) for t in track_ids])
        km = KMeans(n_clusters=n_teams, n_init=10, random_state=0).fit(X)
        labels = km.labels_
        dists = np.linalg.norm(X - km.cluster_centers_[labels], axis=1)
        thr = dists.mean() + OUTLIER_SIGMA * dists.std()
        if len(set(labels)) > 1:
            silhouette = float(silhouette_score(X, labels))
        for i, tid in enumerate(track_ids):
            if dists[i] > thr:  # gk / referee / odd crop — do NOT force into a team
                team_of[tid], role_of[tid], = None, Role.unknown
                flagged += 1
            else:
                team_of[tid], role_of[tid] = int(labels[i]), Role.field

    for tf in tracks.frames:
        for t in tf.tracks:
            if t.cls == ClassLabel.player and t.track_id in team_of:
                t.team = team_of[t.track_id]
                t.role = role_of[t.track_id]

    save_artifact(tracks, od / "tracks.json")  # enrich in place
    counts = {team: sum(1 for v in team_of.values() if v == team) for team in range(n_teams)}
    print(
        f"[s2_teams] {config.clip_name()}: {len(track_ids)} tracks, cluster separation "
        f"(silhouette)={silhouette:.2f}, per-team={counts}, {flagged} flagged gk/ref/unknown"
    )
    return tracks
