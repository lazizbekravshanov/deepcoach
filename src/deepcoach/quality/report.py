"""report — aggregate per-stage quality signals into one readable report.

A first-class output, not a debug log. Reads whichever artifacts exist for a clip
and summarizes the signals that decide whether a run is trustworthy. Writes both a
human-readable report.txt and the structured numbers as report.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..contracts.common import ClassLabel, Role
from ..contracts.detections import DetectionArtifact
from ..contracts.ingest import IngestArtifact
from ..contracts.metrics import MetricsArtifact
from ..contracts.pitch import HomographyArtifact, ProjectionArtifact
from ..contracts.tracks import TrackArtifact
from ..io.artifacts import load_artifact, now_utc_iso, out_dir
from ..io.config import ClipConfig

LOW_CONF = 0.5


def _try_load(path: Path, cls):
    return load_artifact(path, cls) if path.exists() else None


def build_report(config: ClipConfig) -> dict:
    od = out_dir(config.clip_name())
    r: dict = {"clip": config.clip_name(), "config_hash": config.config_hash(), "generated_utc": now_utc_iso()}

    ingest = _try_load(od / "ingest.json", IngestArtifact)
    if ingest:
        r["ingest"] = {
            "resolution": f"{ingest.width}x{ingest.height}",
            "fps": round(ingest.fps, 3),
            "duration_s": round(ingest.duration_s, 1),
            "scene_cuts": len(ingest.detected_cuts),
            "n_shots": ingest.n_shots,
            "single_shot_ok": ingest.n_shots == 1,
        }

    dets = _try_load(od / "detections.json", DetectionArtifact)
    if dets:
        per_frame = [len([d for d in f.detections if d.cls == ClassLabel.player]) for f in dets.frames]
        confs = [d.confidence for f in dets.frames for d in f.detections]
        r["detect"] = {
            "frames": dets.frame_count,
            "mean_players_per_frame": round(sum(per_frame) / max(1, len(per_frame)), 2),
            "min_players_per_frame": min(per_frame) if per_frame else 0,
            "frames_with_no_players": sum(1 for p in per_frame if p == 0),
            "mean_confidence": round(sum(confs) / max(1, len(confs)), 3),
        }

    tracks = _try_load(od / "tracks.json", TrackArtifact)
    if tracks:
        lengths: dict[int, int] = {}
        teamed = flagged = 0
        for tf in tracks.frames:
            for t in tf.tracks:
                if t.cls == ClassLabel.player:
                    lengths[t.track_id] = lengths.get(t.track_id, 0) + 1
                    if t.role == Role.field and t.team is not None:
                        teamed += 1
                    elif t.role == Role.unknown:
                        flagged += 1
        vals = list(lengths.values())
        r["track"] = {
            "n_player_tracks": tracks.n_tracks,
            "mean_track_length": round(sum(vals) / max(1, len(vals)), 1),
            "short_tracks_lt10": sum(1 for v in vals if v < 10),
            "player_detections_teamed": teamed,
            "flagged_gk_ref_unknown": flagged,
        }

    homo = _try_load(od / "homography.json", HomographyArtifact)
    if homo and homo.homographies:
        H = homo.homographies[0]
        r["homography"] = {
            "reprojection_error_px": round(H.reprojection_error_px, 2),
            "n_landmarks": H.n_landmarks,
            "static_camera": homo.static_camera,
            "trustworthy": H.reprojection_error_px <= 15.0,
        }

    proj = _try_load(od / "projected.json", ProjectionArtifact)
    if proj:
        total = low = oob = 0
        for f in proj.frames:
            for e in f.entries:
                total += 1
                if e.projection_confidence < LOW_CONF:
                    low += 1
                if not (0 <= e.pitch_xy.x_m <= proj.pitch_length_m and 0 <= e.pitch_xy.y_m <= proj.pitch_width_m):
                    oob += 1
        r["project"] = {
            "projected_dots": total,
            "low_confidence_pct": round(100.0 * low / max(1, total), 1),
            "outside_pitch_pct": round(100.0 * oob / max(1, total), 1),
        }

    metrics = _try_load(od / "metrics.json", MetricsArtifact)
    if metrics:
        frames_with_two = sum(1 for f in metrics.frames if len(f.teams) == 2)
        r["metrics"] = {
            "frames": len(metrics.frames),
            "metrics_enabled": metrics.metrics_enabled,
            "frames_with_both_teams": frames_with_two,
        }

    # write outputs
    (od / "report.json").write_text(json.dumps(r, indent=2))
    text = _format_text(r)
    (od / "report.txt").write_text(text)
    print(text)
    return r


def _format_text(r: dict) -> str:
    lines = [f"=== deepcoach quality report: {r['clip']} ===", f"config_hash={r['config_hash']}", ""]
    for stage in ["ingest", "detect", "track", "homography", "project", "metrics"]:
        if stage in r:
            lines.append(f"[{stage}]")
            for k, v in r[stage].items():
                lines.append(f"  {k}: {v}")
            lines.append("")
    if "homography" in r and not r["homography"].get("trustworthy", True):
        lines.append("!! homography reprojection error is high — dots are unreliable.")
    if "project" in r and r["project"]["outside_pitch_pct"] > 5.0:
        lines.append("!! many dots fall outside the pitch — check homography landmarks.")
    if "ingest" in r and not r["ingest"].get("single_shot_ok", True):
        lines.append("!! source has scene cuts — analyze ONE continuous segment (clip.frame_range).")
    return "\n".join(lines)
