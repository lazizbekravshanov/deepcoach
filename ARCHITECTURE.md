# deepcoach — Architecture

> Status: **v1 backbone**. This document is the design contract for the project.
> It is written *before* the stage implementations so the contracts and seams can
> be reviewed first. Read this top-to-bottom before reading any code.

---

## 1. Core thesis

A coach's value is **spatial, not event-based**. deepcoach v1 does **not** detect
passes, shots, xG, possession, or any semantic event. The entire job:

> Get every player and the ball as reliable **dots on a 2D pitch**, then compute
> team **shape** and **space** metrics from those positions.

**Pixel→pitch projection (homography) is the make-or-break step.** Everything
downstream is easy geometry once positions are in real pitch coordinates (meters).

**The wrong-dot principle:** a *wrong* dot on the pitch is worse than a *missing*
one. Every stage surfaces uncertainty (confidence, reprojection error, cluster
separation, track length) and never fakes precision. When we can't trust a value,
we say so — we do not silently emit a clean-looking number.

---

## 2. Architectural principles

These are *why the backbone exists* — not decoration. Each is enforced concretely:

1. **Stages are independent.** Each stage is a separate module with a CLI
   entrypoint. It reads artifacts from disk, does work, writes artifacts to disk,
   and never imports or reaches into another stage's internals. The only shared
   code is `contracts/` (data shapes) and `io/` (read/write/config).

2. **Data contracts are sacred.** The pydantic model a stage emits *is* the
   integration surface. Downstream code depends on the **contract**, never on how
   the previous stage produced it. Contracts are versioned (`schema_version`) and
   every artifact records the `config_hash` that produced it.

3. **Config over hardcoding.** Every clip-specific value (paths, pitch dimensions,
   team colors, homography landmarks, fps, model choice) lives in a per-clip YAML
   config. Adding a new clip = writing a new config, never editing code.

4. **Extension points are explicit.** Wherever future work attaches, there is a
   marked `# EXTENSION POINT:` seam with a *stable interface*. v1's implementation
   behind the seam may be trivial or manual; future work swaps the implementation
   without touching callers. Section 8 names every seam and the contract that
   stays fixed across the swap.

5. **Quality is observable.** Every stage emits quality signals. `quality/report.py`
   aggregates them into one readable report so you can *see* where output is
   unreliable instead of trusting clean-but-wrong data.

---

## 3. Scope (v1)

**In scope**
- A **single** clip from a wide/tactical camera, OR one continuous broadcast shot
  with **no scene cuts**.
- Anonymous tracked player IDs (no names).
- Manual homography via clicked pitch landmarks.
- Two-team color clustering, with GK / referee handled as explicit edge cases.

**Out of scope (but seams left for — see §8)**
- Event detection of any kind (passes/shots/xG/possession).
- Full matches, multi-clip, broadcast scene-cut handling.
- Automatic pitch-line / keypoint detection.
- Player identity/name mapping.
- Web UI, database, deployment, auth, multi-user.

---

## 4. Pipeline data flow

```
source ────┐   (local file OR YouTube URL)
           ▼
┌──────────────┐  normalized clip + IngestArtifact (probe + scene-cut scan)
│ S0 ingest    │  data/in/<name>.mp4 , data/out/<name>/ingest.json
│ file|youtube │  ── scene cuts here mean "edited reel"; pick a single-shot
└──────────────┘     segment via clip.frame_range before going further
           │
           ▼
clip.mp4 ─┐
          │   (io/clip.py is the ONLY video reader)
          ▼
┌──────────────┐  DetectionArtifact   ┌──────────────┐  TrackArtifact (team+role)
│ S1 detect    │ ───────────────────▶ │ S2 teams     │ ──────────┐
│ YOLO         │  detections.json     │ kmeans color │           │
└──────────────┘                      └──────────────┘           │
        │ (S2 reads detections, adds nothing to geometry — it     │
        │  enriches each player with team/role; emitted on the    │
        │  Track contract so S3 can carry it through)             │
        ▼                                                         ▼
┌──────────────┐  TrackArtifact       ┌──────────────────────────────┐
│ S3 track     │ ───────────────────▶ │  (team/role merged onto       │
│ ByteTrack    │  tracks.json         │   persistent track_ids)       │
└──────────────┘                      └──────────────────────────────┘
        │
        │  + Homography (from S4, computed once per keyframe)
        ▼
┌──────────────┐  HomographyArtifact
│ S4 homography│  homography.json   ◀── you click landmarks on keyframe(s)
│ MANUAL pick  │
└──────────────┘
        │
        ▼
┌──────────────┐  ProjectionArtifact  ┌──────────────┐  MetricsArtifact
│ S5 project   │ ───────────────────▶ │ S6 metrics   │ ──▶ metrics.json
│ pure geometry│  projected.json      │ registry     │
└──────────────┘  (pitch coords, m)   └──────────────┘
        │                                    │
        └──────────────┬─────────────────────┘
                       ▼
        render/ (radar, overlay, heatmap)   quality/report.py
```

**Note on S2's place in the flow.** Team/role is a *property of a player*, not of
geometry. The clean contract that survives is `Track.team` / `Track.role`. Whether
team assignment happens before or after tracking is an implementation detail behind
the S2 seam — the *contract* is "every `Track` carries `team` and `role`".
**v1 order: S1 → S3 (track) → S2 (assign team/role per track).** Assigning per
track — clustering jersey color aggregated over a track's whole life — is more
robust than per-frame voting and needs no cross-stage imports: S2 reads
`tracks.json`, enriches it in place, and writes it back. (Earlier drafts ran S2
before S3; the contract is identical either way.)

Every stage is **idempotent**: re-running with the same inputs + config produces
the same artifact. Artifacts are keyed by `config_hash` so a config change forces
a recompute rather than silently reusing stale output.

---

## 5. Coordinate frames & units (read carefully — this is where bugs hide)

Two coordinate systems exist and are **never mixed in one field**. Every field
that holds a coordinate names its space and unit.

### 5.1 Pixel space
- Origin: top-left of the video frame.
- Axes: `x` →right, `y` →down.
- Unit: **pixels** (float; sub-pixel allowed).
- Bounding boxes: `xyxy` = `(x1, y1, x2, y2)` with `x1<=x2`, `y1<=y2`.
- A player's ground position in pixel space = **bottom-center of the bbox**
  `((x1+x2)/2, y2)` — the point where the player meets the ground. This is the
  point S5 projects (feet, not centroid — centroid would float above the pitch).

### 5.2 Pitch space
- Origin `(0, 0)` = **one designated corner** of the pitch.
- Axis `x_m`: along pitch **length**, range `0 .. pitch.length_m` (default 105).
- Axis `y_m`: along pitch **width**, range `0 .. pitch.width_m` (default 68).
- Unit: **meters**.
- Orientation is fixed by the config's `homography.landmarks` list: the first
  landmark whose `pitch_xy` is `[0, 0]` defines which physical corner is the
  origin. The config documents the chosen corner explicitly. Convention for v1:
  origin = the corner that is bottom-left when the pitch is drawn with the
  attacking direction left→right; `x_m` increases toward the right goal, `y_m`
  increases toward the far touchline.

Any coordinate leaving pixel space and entering pitch space does so **only** in
S5, **only** through the homography matrix. No other stage performs this mapping.

---

## 6. Data contracts (the backbone)

All contracts are pydantic v2 models in `src/deepcoach/contracts/`, which has
**zero** dependency on `stages/`, `render/`, or `quality/`. It is the neutral
middle that everything else points at.

Conventions shared by all artifacts:
- `schema_version: str` — semver-ish string (`"1.0"`). Bumped when a contract
  changes shape. Loaders refuse to load a major-version mismatch.
- `config_hash: str` — sha256 (first 16 hex chars) of the canonicalized config
  that produced this artifact.
- `stage: str` — which stage emitted it (`"s1_detect"`, …).
- `created_utc: str` — ISO-8601 timestamp.

### 6.1 `contracts/common.py`
```
SCHEMA_VERSION = "1.0"

class ClassLabel(str, Enum):       # what S1 can detect
    player = "player"
    ball   = "ball"

class Role(str, Enum):             # refined by S2; field/gk/ref or unknown
    field   = "field"
    gk      = "gk"
    ref     = "ref"
    unknown = "unknown"

class BBox(BaseModel):             # PIXEL space, xyxy
    x1: float; y1: float; x2: float; y2: float
    # validators: x2>=x1, y2>=y1; helpers: .ground_point() -> (cx, y2)

class ArtifactHeader(BaseModel):   # mixed into every *Artifact envelope
    schema_version: str = SCHEMA_VERSION
    config_hash: str
    stage: str
    created_utc: str
```

### 6.2 `contracts/detections.py`  — output of S1
```
class Detection(BaseModel):
    frame_idx: int
    bbox: BBox                     # pixels
    cls: ClassLabel
    confidence: float              # 0..1

class DetectionFrame(BaseModel):
    frame_idx: int
    timestamp_s: float
    detections: list[Detection]
    source_meta: dict              # e.g. {"frame_w":..,"frame_h":..}

class DetectionArtifact(ArtifactHeader):
    fps: float
    frame_count: int
    frames: list[DetectionFrame]
```

### 6.3 `contracts/tracks.py`  — output of S2+S3
```
class Track(BaseModel):
    track_id: int
    frame_idx: int
    bbox: BBox                     # pixels
    cls: ClassLabel
    confidence: float
    team: int | None               # None until S2 assigns; None for ball/ref
    role: Role                     # field/gk/ref/unknown

class TrackedFrame(BaseModel):
    frame_idx: int
    timestamp_s: float
    tracks: list[Track]

class TrackArtifact(ArtifactHeader):
    fps: float
    frame_count: int
    n_tracks: int
    frames: list[TrackedFrame]
```

### 6.4 `contracts/pitch.py`  — output of S4 (homography) and S5 (projection)
```
class PitchLandmark(BaseModel):
    name: str                      # "corner_tl", "center_circle", "box_tl", ...
    pixel_xy: tuple[float, float]  # clicked pixel location
    pitch_xy: tuple[float, float]  # known real-world meters

class Homography(BaseModel):
    matrix: list[list[float]]      # 3x3 pixel->pitch
    source_keyframe_idx: int
    reprojection_error_px: float   # RMS reprojection error of landmarks
    pitch_landmarks_used: list[PitchLandmark]

class HomographyArtifact(ArtifactHeader):
    # static-camera v1 = one homography; list leaves room for per-keyframe sets
    homographies: list[Homography]
    static_camera: bool            # True in v1

class PitchCoordinate(BaseModel):
    x_m: float                     # meters, 0..length
    y_m: float                     # meters, 0..width

class ProjectedEntry(BaseModel):
    track_id: int
    team: int | None
    role: Role
    cls: ClassLabel
    pitch_xy: PitchCoordinate      # meters
    projection_confidence: float   # 0..1, see §7.3

class ProjectedFrame(BaseModel):
    frame_idx: int
    timestamp_s: float
    entries: list[ProjectedEntry]

class ProjectionArtifact(ArtifactHeader):
    pitch_length_m: float
    pitch_width_m: float
    frames: list[ProjectedFrame]
```

### 6.5 `contracts/metrics.py`  — output of S6
```
class TeamShape(BaseModel):
    team: int
    n_players: int                 # field players used (gk/ref excluded)
    centroid: PitchCoordinate
    compactness_m: float           # spread: mean distance to centroid (m)
    width_m: float                 # span along y_m
    depth_m: float                 # span along x_m
    defensive_line_height_m: float # x_m of deepest defender / rearmost line

class TeamShapeFrame(BaseModel):
    frame_idx: int
    timestamp_s: float
    teams: list[TeamShape]
    # EXTENSION POINT: optional registered metrics land here as a free-form map
    extra: dict[str, float | dict] = {}   # e.g. {"pitch_control": {...}}

class MetricsArtifact(ArtifactHeader):
    metrics_enabled: list[str]
    frames: list[TeamShapeFrame]
```

---

## 7. Stage specifications

Each stage: **read artifact → work → write artifact → emit quality signals.**
CLI form (see §9): `deepcoach run <stage> --config <clip.yaml>`.

### 7.0 S0 ingest — `stages/s0_ingest.py`
- Input: `config.clip.source` = a local file **or** a YouTube URL.
- Work: get the source to disk (yt-dlp for YouTube; ffmpeg to transcode/merge when
  available, plain copy otherwise), normalize to `data/in/<name>.mp4`, probe it
  (fps/resolution/duration), and **scan for scene cuts** (HSV-histogram distance
  between consecutive frames — a hard cut spikes the distance).
- Output: the normalized clip + `IngestArtifact` (`data/out/<name>/ingest.json`)
  carrying `detected_cuts` and `n_shots`.
- Why it exists: it applies the wrong-dot principle **at the door**. A highlight
  reel is an edited montage; each scene cut changes the camera and invalidates the
  per-shot homography, so dots after a cut would be wrong. S0 *detects* this and
  tells you to pick one continuous shot via `clip.frame_range` (or, later, to run
  the shot-segmenter). v1 still analyzes only ONE continuous shot — S0 just makes
  the multi-shot reality visible instead of silently producing wrong dots.
- Quality: `n_shots`, per-cut score, ffmpeg-availability notes.
- **`# EXTENSION POINT: the source seam.`** A real upload/web layer feeds files
  here unchanged; the future broadcast shot-segmenter splits a multi-shot source
  into per-shot clips that each re-enter the **same S1 input contract** (a clip at
  `data/in/`). Everything downstream only ever sees "a clip" — never the source.

### 7.1 S1 detect — `stages/s1_detect.py`
- Input: clip (via `io/clip.py`) + config.
- Work: run YOLO per frame, keep `player` and `ball` detections above
  `model.conf_threshold`. Map YOLO's COCO `person`→`player`, `sports ball`→`ball`
  (v1; a soccer-specific fine-tune later improves this — that's the S1 seam).
- Output: `DetectionArtifact` → `data/out/<clip>/detections.json`.
- Quality: detections/frame (mean, min, max), confidence distribution, frames
  with zero player detections, frames with >0 / >1 ball detections.
- **`# EXTENSION POINT: swap detector / fine-tuned weights`** — config `model.weights`
  selects the model; the stable contract is `DetectionArtifact`.

### 7.2 S2 teams — `stages/s2_teams.py` (runs AFTER S3 in v1)
- Input: `TrackArtifact` (from S3) + clip frames.
- Work: for each player track, sample frames across its life, crop the torso region,
  extract a jersey color feature (mean HSV), and average per track. k-means with
  `k=n_teams` over the per-track features assigns each track to a team.
  - **GK & referee are explicit edge cases.** They will not fit the two outfield
    clusters cleanly. Detect via distance to the assigned cluster center beyond a
    threshold (mean + 2σ) → flag `role=unknown`, `team=None` (excluded from team
    shape). Distinguishing gk-vs-ref by color alone is unreliable; honest `unknown`
    beats a wrong assignment. **Never silently force an outlier into team 0/1.**
- Output: `tracks.json` enriched in place with `team` / `role` per track.
- Quality: cluster separation score (silhouette), per-team track counts, count
  flagged gk/ref/unknown.
- **`# EXTENSION POINT: swap team-assignment strategy`** — e.g. a jersey-classifier
  CNN, or temporal voting. Stable contract: `Track.team` / `Track.role`.

### 7.3 S3 track — `stages/s3_track.py`
- Input: `DetectionArtifact`.
- Work: ByteTrack over player detections → persistent `track_id`s, maintained
  through short occlusions. team/role left unset here (S2 fills them next). The
  ball is carried as a single object (the most-confident ball detection per frame)
  with a reserved `track_id`.
- Output: `TrackArtifact` → `data/out/<clip>/tracks.json` (team=None, role=unknown).
- Quality: track count, mean/median track length (frames), short-track count as a
  **fragmentation proxy**. NOTE: true ID-switch counting needs ground truth, which
  v1 lacks — fragmentation is the observable stand-in, and the doc says so rather
  than reporting a switch count it can't honestly compute.
- **`# EXTENSION POINT: swap tracker / add re-id model`** — stable contract:
  `TrackArtifact` (persistent `track_id`). (Also where the supervision `ByteTrack`
  deprecation gets addressed when we move off the pinned version.)

### 7.4 S4 homography — `stages/s4_homography.py` (the crux)
- Input: clip keyframe(s) + config `homography.landmarks` (or interactive picking).
- Work: **MANUAL.** You click known pitch landmarks (corners, penalty-box corners,
  center spot/circle, halfway-line ends) on a keyframe. We solve the 3×3 homography
  (`cv2.findHomography`) mapping pixel→pitch-meters, and compute RMS reprojection
  error over the landmarks. Static-camera assumption documented; if the camera
  pans, recompute per keyframe and store multiple `Homography` entries.
  - Two input modes: (a) interactive OpenCV window writes the clicked points back
    into the config / a sidecar; (b) landmarks already present in config → solve
    headless. Both produce the same `HomographyArtifact`.
- Output: `HomographyArtifact` → `data/out/<clip>/homography.json`.
- Quality: reprojection_error_px per homography, number of landmarks used,
  whether error exceeds a warn threshold.
- **`# EXTENSION POINT: automatic pitch-line detection replaces the manual picker`**
  — behind the same `Homography` contract. Auto-detection emits the identical
  `HomographyArtifact`; S5 never knows the difference.

### 7.5 S5 project — `stages/s5_project.py`
- Input: `TrackArtifact` + `HomographyArtifact`.
- Work: for each track in each frame, take the bbox **ground point** (§5.1),
  apply the homography → `PitchCoordinate` in meters. **Pure, deterministic
  geometry — no model, no randomness, kept boring and stable.**
- `projection_confidence` (§ wrong-dot principle) is a function of:
  (1) the homography's reprojection error, and
  (2) the point's distance from the **convex hull of the landmarks** — points
      projected far outside the landmark hull are extrapolated and untrustworthy.
  Confidence decays from 1.0 toward 0.0 as either factor worsens.
- Output: `ProjectionArtifact` → `data/out/<clip>/projected.json`.
- Quality: fraction of entries with low projection_confidence, count of points
  falling outside pitch bounds (a strong wrong-dot signal).

### 7.6 S6 metrics — `stages/s6_metrics.py`
- Input: `ProjectionArtifact`.
- **Confidence-gating (wrong-dot guard):** only dots with
  `projection_confidence >= metrics.min_confidence` feed the metrics; the run
  reports used-vs-dropped and warns when most are dropped. Metrics from
  untrustworthy positions are themselves untrustworthy, so we exclude rather than
  average them in.
- Work: per frame compute the enabled metrics via **two registries**:
  - per-team `fn(points, ctx)` — `centroid, compactness, width, depth, def_line`
    and `hull_area` (space occupied = convex-hull area).
  - frame-level `fn(by_team, ctx)` — `pitch_control` (territorial dominance via a
    nearest-player / Voronoi partition), needing all teams at once; results land
    in `TeamShapeFrame.extra`.
  `pitch_control` and `hull_area` are per-frame / position-only, so they survive
  the track fragmentation broadcast footage causes. GK/ref excluded; each team
  records `n_players` so thin frames are visible.
- Output: `MetricsArtifact` → `data/out/<clip>/metrics.json`.
- Quality: used-vs-dropped dot counts, thin-team-frame count, inferred defending
  side per team (a heuristic — surfaced because it can be wrong on poor footage).
- **`# EXTENSION POINT: register new metrics here`** — add a function + `@register`
  (per-team) or `@register_frame` (needs all teams); enable it in
  `metrics.enabled`. Far later: event metrics. No pipeline edit required.
- Render: `render/pitchcontrol.py` turns `pitch_control` into an aggregate
  territorial-dominance map (PNG), confidence-gated like the metric.

---

## 8. Extension points — the "strings ready to bind"

For each future capability: the seam, and the **contract that stays stable** while
the implementation changes behind it.

| Future capability | Seam (where it attaches) | Stable contract |
|---|---|---|
| Better / fine-tuned detector | `s1_detect` behind `model.weights` config | `DetectionArtifact` |
| Auto team assignment (jersey CNN, temporal vote) | `s2_teams` strategy | `Track.team` / `Track.role` |
| Re-identification model | `s3_track` tracker swap | `TrackArtifact` (persistent `track_id`) |
| **Automatic pitch-line detection** | `s4_homography` picker swap | `Homography` / `HomographyArtifact` |
| File upload / YouTube ingest | `s0_ingest` source seam (`clip.source.kind`) | normalized clip at `data/in/` + `IngestArtifact` |
| Broadcast multi-shot handling | extends `s0_ingest`: segment shots → per-shot clips | feeds the **same S1 input contract** (a clip) |
| New metrics (incl. eventually events) | `s6_metrics` registry | registry signature `fn(ProjectedFrame, ctx) -> value`; lands in `TeamShapeFrame.extra` |
| Web / DB / visualization layer | consumes artifacts | `ProjectionArtifact` + `MetricsArtifact`; pipeline never knows it exists |

The pipeline core depends only on the contracts in the middle column. Each swap is
"replace the implementation behind the seam, leave the contract alone."

---

## 9. Config-driven design

One YAML per clip (`configs/<clip>.yaml`), validated by a pydantic `ClipConfig`
(`io/config.py`). Template documented in `configs/clip_template.yaml`:

```yaml
clip:        { path, fps_override?, frame_range?: [start, end] }
pitch:       { length_m: 105, width_m: 68 }
model:       { weights, conf_threshold, device: mps }
teams:       { n_teams: 2, sample_frames: [...] }
homography:  { mode: manual, landmarks: [{name, pixel_xy, pitch_xy}, ...] }
             # EXTENSION POINT: mode: auto
metrics:     { enabled: [centroid, compactness, width, depth, def_line] }
render:      { radar: true, overlay: true, heatmap: true }
```

`config_hash` = sha256 of the canonical (sorted-key) serialization, truncated to
16 hex chars. Stamped into every artifact so output is traceable to its config and
stale artifacts are detectable.

---

## 10. Artifacts, versioning, idempotency

- Location: `data/out/<clip_name>/<stage>.json`. (`data/` is gitignored.)
- Format: **JSON** for v1 — human-readable and trivially diffable, which matters
  while we're still eyeballing trust. (Parquet is an easy later swap for the
  large per-frame artifacts behind `io/artifacts.py`; the contract is unchanged.)
- `io/artifacts.py` provides typed `save_artifact(model, path)` /
  `load_artifact(path, ModelClass)`. Load refuses a major `schema_version` mismatch
  and warns on `config_hash` mismatch (stale-vs-current).
- Idempotency: a stage's output is a pure function of (its input artifacts +
  config). Re-running overwrites with identical content.

---

## 11. Known failure modes (named, so the quality report can watch them)

- **Occlusion** — players overlap; detections merge/drop. Mitigation: ByteTrack
  carries IDs through short gaps; quality report shows track-presence fraction.
- **ID switches** — track identity swaps between players. Every switch logged;
  switch count is a headline quality number.
- **GK / referee jersey color** — do not fit two outfield clusters; forcing them in
  pollutes team shape. Handled as explicit `role` flags; never silently assigned.
- **Homography error** — the dominant risk. Few/poor landmarks or a moving camera
  inflate reprojection error and produce **wrong dots**. Surfaced as
  `reprojection_error_px` and `projection_confidence`; points outside the landmark
  hull / pitch bounds are flagged.
- **Perspective foot-point error** — distant players' feet are ambiguous; small
  pixel error → large meter error far from the camera. Folded into
  `projection_confidence` via distance-from-hull.
- **Thin frames** — too few field players visible to compute meaningful team shape;
  metrics record `n_players` so these frames are visible, not silently averaged in.

---

## 12. Test clip recommendation & expected input format

**Expected S1 input:** a single video file (`.mp4`/`.mov`, H.264) of **one
continuous shot, no scene cuts**, ideally from a **wide/tactical** vantage where
most of the pitch and both teams are visible. 10–30 seconds at 25–30 fps is plenty
for the backbone. Known/declared `fps`. Place at `data/in/<clip>.mp4`.

**Legally-clean clip options (you choose; I'll target whichever you pick):**
1. **SoccerNet** sample clips — research dataset, widely used for exactly this task
   (tracking/homography). Good landmarks, tactical-ish broadcast. Requires a quick
   registration/NDA for research use.
2. **Record your own** — a phone on a tripod at a local/amateur match or training
   from an elevated, static, wide position. Zero licensing questions, and a static
   camera is the *ideal* v1 case (one homography for the whole clip).
3. **Roboflow / Ultralytics community soccer sample clips** — small permissively-
   licensed snippets used in their tutorials; convenient for a first smoke test.

My recommendation for development: **start with option 2 or 3** (static, no
licensing friction) to validate the backbone end-to-end, then move to SoccerNet
for a more realistic tactical clip. A static camera means S4 runs once and S5 is
maximally stable — the cleanest path to a trustworthy first result.

---

## 13. What this session delivers (and the pause)

Delivered now: this doc, `pyproject.toml` (pinned, Python 3.14), `README.md`,
`configs/clip_template.yaml`, all `contracts/` models, the `io/` layer
(`clip.py`, `artifacts.py`, `config.py`), the `cli.py` dispatcher, **a fully
implemented `S0 ingest`** (local file + YouTube + scene-cut inspection), S1–S6
stage stubs with their extension-point seams, and contract + scene-cut tests.

S0 is implemented now (not stubbed) because it's the tool we use to inspect an
incoming highlight video and decide which single continuous segment to analyze.

**We pause for review before implementing S1–S6 logic.**
