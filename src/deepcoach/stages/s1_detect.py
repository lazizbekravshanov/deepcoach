"""S1 detect — YOLO per-frame player + ball detection -> DetectionArtifact.

Reads the clip (via io/clip.py), writes data/out/<clip>/detections.json. Idempotent.

# EXTENSION POINT: swap detector / fine-tuned weights.
#   The detector is selected by config.model.weights. A soccer-specific fine-tune
#   plugs in here. The stable contract downstream depends on is DetectionArtifact.

NOTE: implementation deferred — backbone review pause (ARCHITECTURE.md §13).
"""

from __future__ import annotations

from ..contracts.common import BBox, ClassLabel
from ..contracts.detections import Detection, DetectionArtifact, DetectionFrame
from ..io.artifacts import now_utc_iso, out_dir, save_artifact
from ..io.clip import ClipReader
from ..io.config import ClipConfig

STAGE = "s1_detect"

# COCO class ids emitted by stock YOLO weights. A soccer fine-tune (the extension)
# may use its own ids; that mapping lives here, behind the DetectionArtifact contract.
COCO_PERSON = 0
COCO_SPORTS_BALL = 32


def run(config: ClipConfig) -> DetectionArtifact:
    from ultralytics import YOLO  # local import: heavy dep, only needed to detect

    model = YOLO(config.model.weights)
    conf = config.model.conf_threshold
    device = config.model.device

    frames_out: list[DetectionFrame] = []
    n_players = n_balls = zero_player_frames = 0
    conf_sum = conf_count = 0.0

    with ClipReader(config) as reader:
        fps, w, h = reader.fps, reader.width, reader.height
        for fr in reader:
            res = model.predict(fr.image, conf=conf, device=device, verbose=False)[0]
            dets: list[Detection] = []
            frame_players = 0
            for box in res.boxes:
                cls_id = int(box.cls)
                if cls_id == COCO_PERSON:
                    label = ClassLabel.player
                    frame_players += 1
                    n_players += 1
                elif cls_id == COCO_SPORTS_BALL:
                    label = ClassLabel.ball
                    n_balls += 1
                else:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                c = float(box.conf)
                conf_sum += c
                conf_count += 1
                dets.append(Detection(frame_idx=fr.frame_idx, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), cls=label, confidence=c))
            if frame_players == 0:
                zero_player_frames += 1
            frames_out.append(
                DetectionFrame(
                    frame_idx=fr.frame_idx,
                    timestamp_s=fr.timestamp_s,
                    detections=dets,
                    source_meta={"frame_w": w, "frame_h": h},
                )
            )

    art = DetectionArtifact(
        config_hash=config.config_hash(),
        stage=STAGE,
        created_utc=now_utc_iso(),
        fps=fps,
        frame_count=len(frames_out),
        frames=frames_out,
    )
    save_artifact(art, out_dir(config.clip_name()) / "detections.json")
    nf = max(1, len(frames_out))
    mean_conf = (conf_sum / conf_count) if conf_count else 0.0
    print(
        f"[s1_detect] {config.clip_name()}: {len(frames_out)} frames, "
        f"{n_players} player dets ({n_players/nf:.1f}/frame), {n_balls} ball dets, "
        f"mean conf={mean_conf:.2f}, {zero_player_frames} frames with no players"
    )
    return art
