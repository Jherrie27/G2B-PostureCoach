"""Run for 30 seconds against your webcam, print classifications.

This is a MANUAL test — run it yourself in front of the camera:
    python tests/test_live_classifier.py
It cannot be run headless/CI (needs a real webcam + a person).
"""
import time
import cv2
import numpy as np

from src.cv.pose_extractor import PoseExtractor
from src.cv.normalizer import normalize
from src.cv.features import extract_features
from src.cv.classifier import PostureClassifier


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    pe = PoseExtractor(model_complexity=1)
    clf = PostureClassifier()

    start = time.time()
    n_frames = 0
    while time.time() - start < 30:
        ok, frame = cap.read()
        if not ok:
            break
        n_frames += 1
        la = pe.extract(frame)
        if la is None or not la.is_reliable:
            print("[unreliable]")
            continue
        normed = normalize(la.data)
        if normed is None:
            continue
        feats = extract_features(normed)
        if feats is None:
            continue
        label, conf, _ = clf.predict(feats)
        print(f"{label:18s}  conf={conf:.2f}  "
              f"ear_x={feats['ear_shoulder_offset_x']:+.2f}  "
              f"sh_z={feats['shoulder_roll_z']:+.2f}  "
              f"tilt={feats['shoulder_tilt_angle']:+.1f}")
    cap.release()
    pe.close()
    print(f"\n{n_frames} frames in 30s = {n_frames / 30:.1f} fps")


if __name__ == "__main__":
    main()
