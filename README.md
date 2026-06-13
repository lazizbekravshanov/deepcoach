# deepcoach

Turn **one** soccer video clip into tactical-spatial data: every player and the
ball as reliable **dots on a 2D pitch**, then team **shape** and **space** metrics
computed from those positions.

This is a learning project and a **backbone** — a sound, extensible skeleton with
stable contracts between stages, not a finished product.

## What it is (and is not)

A coach's value is **spatial, not event-based**. deepcoach v1 does **not** detect
passes, shots, xG, or possession. Its entire job is positions-on-a-pitch and the
shape/space metrics that follow from them.

**The wrong-dot principle:** a *wrong* dot on the pitch is worse than a *missing*
one. Every stage surfaces uncertainty (detection confidence, cluster separation,
track length, homography reprojection error, projection confidence) instead of
faking precision. Pixel→pitch projection (homography) is the make-or-break step;
everything downstream is easy geometry once positions are in meters.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design: data flow, the
pydantic contracts, config-driven design, the manual-homography decision, every
extension seam, and the known failure modes.

## Scope (v1)

- **In:** a single clip from a wide/tactical camera (or one continuous broadcast
  shot, no scene cuts); anonymous tracked player IDs; manual homography via
  clicked landmarks; two-team color clustering with GK/referee as edge cases.
- **Out (seams left for):** event detection, full matches / multi-clip, automatic
  pitch-line detection, player identity, web/DB/UI.

## Pipeline

```
clip.mp4
  -> S1 detect      (YOLO)            detections.json
  -> S2 teams       (k-means color)   team + role
  -> S3 track       (ByteTrack)       tracks.json
  -> S4 homography  (MANUAL clicks)   homography.json
  -> S5 project     (pure geometry)   projected.json   (pitch meters)
  -> S6 metrics     (registry)        metrics.json
  -> render/ (radar, overlay, heatmap)  +  quality/report
```

Each stage is independent: it reads artifacts from disk, does its work, writes
artifacts, and never reaches into another stage. The schema each stage emits is
the integration surface; downstream depends on the **contract**, not the producer.

## Install

Targets the latest CPython (**3.14**); the full CV stack (torch, ultralytics,
opencv, supervision) ships 3.14 wheels and is verified on it.

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -e ".[cv,ingest,dev]"   # full stack (stages + render + YouTube ingest + tests)
# or, contract/io work only (fast, no torch):
pip install -e ".[dev]"
```

Apple Silicon: the model stack uses **MPS** where available (`model.device: mps`).

**System dependency for ingest:** YouTube download and video transcoding need
**ffmpeg** on PATH (`brew install ffmpeg`). Local-file ingest and scene-cut
detection work without it (with a warning instead of transcoding).

## Run

Each stage runs independently against a per-clip config:

```bash
deepcoach run s0_ingest     --config configs/my_clip.yaml  # file or YouTube -> data/in/ + cut scan
deepcoach run s1_detect     --config configs/my_clip.yaml  # YOLO detections
deepcoach run s3_track      --config configs/my_clip.yaml  # ByteTrack -> persistent ids
deepcoach run s2_teams      --config configs/my_clip.yaml  # team/role per track (after tracking)
deepcoach run s4_homography --config configs/my_clip.yaml  # interactive landmark picking (or from config)
deepcoach run s5_project    --config configs/my_clip.yaml  # dots on the pitch (meters)
deepcoach run s6_metrics    --config configs/my_clip.yaml  # team shape metrics
deepcoach render            --config configs/my_clip.yaml  # radar / overlay / heatmap
deepcoach report            --config configs/my_clip.yaml  # quality report
deepcoach stages                                           # list stages
```

Stage order is **detect → track → teams**: team/role is assigned per *track*
(aggregating jersey color over the whole track) rather than per detection — more
robust, and `Track.team`/`Track.role` stays the contract either way.

1. Point `clip.source` at a local file or a YouTube URL (or place a file at
   `data/in/<clip>.mp4` yourself). Copy `configs/clip_template.yaml` to
   `configs/<clip>.yaml`.
2. `deepcoach run s0_ingest` normalizes the clip into `data/in/` and writes
   `ingest.json`. **If it reports scene cuts, the source is an edited reel** —
   pick one continuous shot via `clip.frame_range` before continuing (v1 analyzes
   a single shot).
3. Fill in pitch dimensions, team sample frames, and homography landmarks, then
   run stages in order; artifacts land in `data/out/<clip>/`.

## Status

Backbone in place: contracts, io/config layer, CLI dispatcher, stage seams.
Stage implementations (S1–S6), renderers, and the quality report are the next
step. See `ARCHITECTURE.md` §13.
