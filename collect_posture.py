"""Record REAL landmark data from your webcam for one posture class.

Run once per class (waist-up framing, good front lighting):

    .venv\\Scripts\\python.exe collect_posture.py correct_posture --seconds 60
    .venv\\Scripts\\python.exe collect_posture.py slouching      --seconds 60
    .venv\\Scripts\\python.exe collect_posture.py neck_forward   --seconds 60
    .venv\\Scripts\\python.exe collect_posture.py lean           --seconds 60

Then retrain on the real data:

    .venv\\Scripts\\python.exe train_real.py

Each run writes data/raw_landmarks/CV/<label>_<timestamp>.csv with the exact
landmark_<idx>_<x|y|z|v> columns that build_dataset.py / csv_to_features.py
already expect. Press 'q' in the preview window to stop early.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

from src.utils.camera import open_working_camera
from src.cv.pose_extractor import PoseExtractor
from src.cv.landmarks import ORDERED_NAMES, LANDMARK_INDICES
from src.utils.config import mediapipe_complexity

CLASSES = ["correct_posture", "slouching", "neck_forward", "lean"]
OUT_DIR = Path("data/raw_landmarks/CV")


def _row_from_landmarks(data, label: str, reliable: bool) -> dict:
    """data is the (9, 4) array in ORDERED_NAMES order."""
    row = {}
    for i, name in enumerate(ORDERED_NAMES):
        idx = LANDMARK_INDICES[name]
        x, y, z, v = data[i]
        row[f"landmark_{idx}_x"] = float(x)
        row[f"landmark_{idx}_y"] = float(y)
        row[f"landmark_{idx}_z"] = float(z)
        row[f"landmark_{idx}_v"] = float(v)
    row["label"] = label
    row["is_reliable"] = bool(reliable)
    return row


def main():
    ap = argparse.ArgumentParser(description="Collect real posture landmark data.")
    ap.add_argument("label", choices=CLASSES, help="Posture class you will hold.")
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="How long to record (default 60).")
    ap.add_argument("--countdown", type=int, default=5,
                    help="Seconds to get into position before recording.")
    ap.add_argument("--camera", type=int, default=None,
                    help="Force a camera index (else auto-detect / G2B_CAMERA_INDEX).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap, cam_idx = open_working_camera(preferred=args.camera)
    # Collect at the SAME MediaPipe complexity the live app uses on this device
    # (Full=1 on PC, Lite=0 on the Pi). This keeps training data consistent with
    # inference, so a retrain done on the Pi matches how the Pi actually runs.
    complexity = mediapipe_complexity()
    print(f"Using camera index {cam_idx}. Class = {args.label}. "
          f"MediaPipe complexity = {complexity}.")
    pose = PoseExtractor(model_complexity=complexity)

    rows = []
    win = f"Collecting: {args.label}  (press q to stop)"

    # --- Countdown so you can get into the posture ---
    cd_end = time.time() + args.countdown
    while time.time() < cd_end:
        ok, frame = cap.read()
        if not ok:
            continue
        left = cd_end - time.time()
        cv2.putText(frame, f"Get into '{args.label}' posture: {left:0.0f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Record ---
    rec_end = time.time() + args.seconds
    reliable_count = 0
    while time.time() < rec_end:
        ok, frame = cap.read()
        if not ok:
            continue
        la = pose.extract(frame)
        captured = False
        if la is not None:
            rows.append(_row_from_landmarks(la.data, args.label, la.is_reliable))
            captured = True
            if la.is_reliable:
                reliable_count += 1

        left = rec_end - time.time()
        color = (0, 200, 0) if (la is not None and la.is_reliable) else (0, 0, 255)
        status = "OK" if (la is not None and la.is_reliable) else "ADJUST FRAMING"
        cv2.putText(frame, f"{args.label}  {left:0.0f}s  frames={len(rows)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
        cv2.putText(frame, status, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    if not rows:
        print("No frames captured (no pose detected). Check lighting / framing.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"{args.label}_{ts}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved {len(rows)} frames ({reliable_count} reliable) -> {out_path}")
    if reliable_count < 0.5 * len(rows):
        print("WARNING: many frames were NOT reliable (hips/landmarks low "
              "visibility). Re-record framed waist-up with better lighting for "
              "best accuracy.")


if __name__ == "__main__":
    main()
