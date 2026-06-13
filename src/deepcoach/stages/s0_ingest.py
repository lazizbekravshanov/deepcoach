"""S0 ingest — get a normalized single clip into data/in/ and inspect it.

This stage sits UPSTREAM of the clip contract. It accepts:
  - a local file  (config.clip.source.kind == "file"), or
  - a YouTube URL (config.clip.source.kind == "youtube", via yt-dlp + ffmpeg)
normalizes it to data/in/<name>.mp4, probes it, and scans for scene cuts. The
cut scan is the inspection that tells us whether the source is one continuous
shot (v1's assumption) or an edited highlight reel that needs a segment chosen /
shot-segmentation (ARCHITECTURE.md §7.0, §11).

# EXTENSION POINT: this is the source seam.
#   - a real upload/web layer feeds files here without changing anything downstream
#   - the future broadcast shot-segmenter splits a multi-shot source into per-shot
#     clips, each re-entering the SAME S1 input contract
# Whatever the source, everything downstream sees only "a clip at data/in/<name>.mp4".

Output: IngestArtifact at data/out/<name>/ingest.json + the normalized clip.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from ..contracts.ingest import IngestArtifact, SceneCut
from ..io.artifacts import now_utc_iso, out_dir, save_artifact
from ..io.clip import Frame, iter_video_frames, probe_video
from ..io.config import ClipConfig

STAGE = "s0_ingest"


def detect_scene_cuts(frames, threshold: float, downscale: int = 64) -> list[SceneCut]:
    """Hard-cut detection via HSV-histogram dissimilarity between consecutive frames.

    Dependency-free (OpenCV only). A solid, standard heuristic: a hard cut produces
    a large jump in color-histogram distance. Pure function over a frame iterable,
    so it's unit-testable on synthetic frames without a real video file.
    """
    import cv2  # local import: only needed when actually scanning

    cuts: list[SceneCut] = []
    prev_hist = None
    for f in frames:  # f: Frame
        small = cv2.resize(f.image, (downscale, downscale))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            d = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            if d >= threshold:
                cuts.append(SceneCut(frame_idx=f.frame_idx, timestamp_s=f.timestamp_s, score=d))
        prev_hist = hist
    return cuts


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _download_youtube(url: str, workdir: Path, notes: list[str], max_height: int | None = None) -> Path:
    """Download a YouTube URL to a local file. Requires the `ingest` extra (yt-dlp).

    For CV we don't need audio, so we prefer a single VIDEO-ONLY mp4 stream — this
    unlocks 720p/1080p without ffmpeg (no audio merge needed). max_height caps it.
    """
    try:
        import yt_dlp
    except ImportError as e:  # pragma: no cover - exercised only on the youtube path
        raise ImportError("YouTube ingest needs yt-dlp: pip install -e '.[ingest]'") from e

    hcap = f"[height<={max_height}]" if max_height else ""
    # video-only mp4 first (no merge, no ffmpeg), then progressive mp4, then anything.
    fmt = f"bestvideo[ext=mp4]{hcap}/best[ext=mp4]{hcap}/best{hcap}/best"
    notes.append(f"downloaded video-only mp4 (no audio; not needed for CV){' capped at '+str(max_height)+'p' if max_height else ''}")

    out_tmpl = str(workdir / "%(id)s.%(ext)s")
    opts = {"format": fmt, "outtmpl": out_tmpl, "quiet": True, "noprogress": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded = Path(ydl.prepare_filename(info))
    if not downloaded.exists():
        # prepare_filename can disagree with the merged container extension
        candidates = list(workdir.glob(f"{info['id']}.*"))
        if not candidates:
            raise IOError(f"yt-dlp reported success but no file found for {url}")
        downloaded = candidates[0]
    return downloaded


def _normalize(src: Path, dst: Path, do_normalize: bool, notes: list[str]) -> None:
    """Produce dst from src: transcode to H.264 mp4 if ffmpeg is available, else copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_normalize and _have_ffmpeg():
        cmd = ["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(dst)]
        subprocess.run(cmd, check=True, capture_output=True)
    else:
        if do_normalize and not _have_ffmpeg():
            notes.append("ffmpeg missing: copied source without transcoding")
        if src.resolve() != dst.resolve():
            shutil.copy(src, dst)


def run(config: ClipConfig) -> IngestArtifact:
    src_spec = config.clip.source
    if src_spec is None or not src_spec.ref:
        raise ValueError("S0 ingest needs clip.source.{kind,ref} in the config")

    name = config.clip_name()
    notes: list[str] = []
    dst = Path(config.clip.path)

    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        if src_spec.kind == "youtube":
            working = _download_youtube(src_spec.ref, workdir, notes, config.ingest.max_height)
        elif src_spec.kind == "file":
            working = Path(src_spec.ref)
            if not working.exists():
                raise FileNotFoundError(f"source file not found: {working}")
        else:
            raise ValueError(f"unknown source kind {src_spec.kind!r} (file|youtube)")

        _normalize(working, dst, config.ingest.normalize, notes)

    meta = probe_video(dst)
    cuts: list[SceneCut] = []
    if config.ingest.detect_cuts:
        cuts = detect_scene_cuts(
            iter_video_frames(dst, fps_override=config.clip.fps_override, step=config.ingest.scan_step),
            threshold=config.ingest.cut_threshold,
        )

    artifact = IngestArtifact(
        config_hash=config.config_hash(),
        stage=STAGE,
        created_utc=now_utc_iso(),
        source_kind=src_spec.kind,
        source_ref=src_spec.ref,
        normalized_path=str(dst),
        fps=meta["fps"],
        width=meta["width"],
        height=meta["height"],
        frame_count=meta["frame_count"],
        duration_s=meta["duration_s"],
        detected_cuts=cuts,
        n_shots=len(cuts) + 1,
        notes=notes,
    )
    save_artifact(artifact, out_dir(name) / "ingest.json")

    print(
        f"[s0_ingest] {name}: {meta['width']}x{meta['height']} @ {meta['fps']:.3f}fps, "
        f"{meta['duration_s']:.1f}s, {len(cuts)} cut(s) -> {len(cuts)+1} shot(s)"
    )
    if cuts:
        print(
            "[s0_ingest] WARNING: scene cuts detected — this looks like an edited reel. "
            "v1 needs ONE continuous shot; pick a segment via clip.frame_range "
            "(see ingest.json detected_cuts)."
        )
    return artifact
