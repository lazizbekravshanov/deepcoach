"""S4 homography (the crux) — MANUAL pitch-landmark picking -> HomographyArtifact.

You click known pitch landmarks on a keyframe; we solve the 3x3 pixel->meters
homography (cv2.findHomography) and record RMS reprojection error. Static-camera
assumption documented; recompute per keyframe if the camera pans.

# EXTENSION POINT: automatic pitch-line detection replaces the manual picker.
#   The auto picker emits the identical Homography / HomographyArtifact contract;
#   S5 never knows the difference.

NOTE: implementation deferred — backbone review pause (ARCHITECTURE.md §13).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..contracts.pitch import Homography, HomographyArtifact, PitchLandmark
from ..io.artifacts import now_utc_iso, out_dir, save_artifact
from ..io.config import ClipConfig

STAGE = "s4_homography"

# Reprojection error above this is loudly flagged — the dominant wrong-dot risk.
WARN_REPROJ_PX = 15.0


def solve_homography(landmarks: list[PitchLandmark]) -> tuple[list[list[float]], float]:
    """Solve the 3x3 pixel->pitch(meters) homography and its RMS reprojection error.

    Pure function (no I/O): the integration point for both the manual picker and a
    future automatic pitch-line detector. Reprojection error is reported in PIXELS
    (known pitch points mapped back through H^-1, compared to the clicked pixels) to
    match the `reprojection_error_px` contract field.
    """
    if len(landmarks) < 4:
        raise ValueError(f"homography needs >= 4 landmarks, got {len(landmarks)}")
    src = np.array([lm.pixel_xy for lm in landmarks], dtype=np.float64)  # pixels
    dst = np.array([lm.pitch_xy for lm in landmarks], dtype=np.float64)  # meters
    H, _ = cv2.findHomography(src, dst, method=0)  # 0 = least-squares over all points
    if H is None:
        raise ValueError("homography solve failed (degenerate landmark configuration?)")

    Hinv = np.linalg.inv(H)
    back = cv2.perspectiveTransform(dst.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    err_px = float(np.sqrt(np.mean(np.sum((back - src) ** 2, axis=1))))
    return H.tolist(), err_px


def _landmarks_are_provided(landmarks: list[PitchLandmark]) -> bool:
    """True if every landmark has a real (non-placeholder) pixel click in the config."""
    return len(landmarks) >= 4 and all((lm.pixel_xy[0] or lm.pixel_xy[1]) for lm in landmarks)


def pick_landmarks_interactive(config: ClipConfig) -> list[PitchLandmark]:
    """Open the keyframe and let the user click each named landmark in turn.

    Requires a display (cv2 GUI), so this is run on the user's machine — not in a
    headless run. The pitch_xy / names come from config; the user fills pixel_xy.
    # EXTENSION POINT: automatic pitch-line detection replaces this picker, emitting
    #   the same PitchLandmark list / Homography contract.
    """
    from ..io.clip import ClipReader  # local import: only needed for interactive picking

    specs = config.homography.landmarks
    if len(specs) < 4:
        raise ValueError("config.homography.landmarks must list >= 4 named pitch points to click")

    with ClipReader(config) as reader:
        frame = reader.read_frame(config.homography.keyframe_idx).image

    picked: list[PitchLandmark] = []
    state = {"current": 0, "xy": None}

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["xy"] = (float(x), float(y))

    win = "deepcoach S4 — ENTER=confirm  u=undo  s=skip(not visible)  ESC=abort"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _on_mouse)
    while state["current"] < len(specs):
        spec = specs[state["current"]]
        disp = frame.copy()
        for p in picked:
            cv2.circle(disp, (int(p.pixel_xy[0]), int(p.pixel_xy[1])), 5, (0, 255, 0), -1)
        if state["xy"] is not None:
            cv2.circle(disp, (int(state["xy"][0]), int(state["xy"][1])), 5, (0, 0, 255), 2)
        cv2.putText(disp, f"[{state['current']+1}/{len(specs)}] click {spec.name} @ {spec.pitch_xy}m",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp, f"confirmed: {len(picked)} (need >=4)", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            cv2.destroyWindow(win)
            raise KeyboardInterrupt("landmark picking aborted")
        if key == ord("s"):  # skip a landmark not visible in this frame
            state["current"] += 1
            state["xy"] = None
        if key == ord("u") and picked:
            picked.pop()
            state["current"] = max(0, state["current"] - 1)
            state["xy"] = None
        if key in (13, 10) and state["xy"] is not None:  # ENTER confirms the click
            picked.append(PitchLandmark(name=spec.name, pixel_xy=state["xy"], pitch_xy=spec.pitch_xy))
            state["current"] += 1
            state["xy"] = None
    cv2.destroyWindow(win)
    if len(picked) < 4:
        raise ValueError(f"need >= 4 clicked landmarks, got {len(picked)}")
    return picked


def run(config: ClipConfig) -> HomographyArtifact:
    name = config.clip_name()
    landmarks = config.homography.landmarks

    if _landmarks_are_provided(landmarks):
        used = landmarks  # headless solve from config-provided clicks
        mode = "config"
    else:
        used = pick_landmarks_interactive(config)  # GUI picking on the user's machine
        mode = "interactive"

    matrix, err = solve_homography(used)
    art = HomographyArtifact(
        config_hash=config.config_hash(),
        stage=STAGE,
        created_utc=now_utc_iso(),
        static_camera=True,
        homographies=[
            Homography(
                matrix=matrix,
                source_keyframe_idx=config.homography.keyframe_idx,
                reprojection_error_px=err,
                pitch_landmarks_used=used,
            )
        ],
    )
    save_artifact(art, out_dir(name) / "homography.json")
    print(f"[s4_homography] {name}: {len(used)} landmarks ({mode}), reprojection_error={err:.2f}px")
    if err > WARN_REPROJ_PX:
        print(
            f"[s4_homography] WARNING: reprojection error {err:.1f}px > {WARN_REPROJ_PX}px — "
            "dots will be unreliable. Re-click landmarks with wider, more accurate spread."
        )
    return art
