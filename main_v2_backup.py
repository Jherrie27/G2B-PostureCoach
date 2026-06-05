"""
main.py
Posture Correction Coach — Core CV Application
Camera + MediaPipe + Classifier + Session Tracker

Run this on Raspberry Pi 5:
    python main.py

Or on laptop for testing:
    python main.py --laptop
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import threading
import argparse
import queue
import sys

from session import SessionTracker
from rag_query import get_coaching_advice, answer_user_question

# ── ARGS ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--laptop", action="store_true",
                    help="Use laptop webcam instead of Pi camera")
args = parser.parse_args()

# ── CONFIG ────────────────────────────────────────────────────────
MODEL_PATH        = "posture_classifier_v2.pkl"
SOFT_ALERT_SEC    = 120    # 2 minutes
HARD_ALERT_SEC    = 600    # 10 minutes
FRAME_WIDTH       = 640
FRAME_HEIGHT      = 480

LABEL_COLORS = {
    "correct":      (34,  197, 94),
    "slouching":    (239, 68,  68),
    "neck_forward": (234, 179, 8),
    "leaning":      (168, 85,  247),
}

# ── LOAD CLASSIFIER ───────────────────────────────────────────────
print("Loading posture classifier...")
clf = joblib.load(MODEL_PATH)
print(f"Classifier loaded: {MODEL_PATH}")

# ── MEDIAPIPE ─────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose    = mp_pose.Pose(
    min_detection_confidence = 0.6,
    min_tracking_confidence  = 0.6,
)

# ── SESSION TRACKER ───────────────────────────────────────────────
tracker = SessionTracker(
    soft_alert_sec = SOFT_ALERT_SEC,
    hard_alert_sec = HARD_ALERT_SEC,
)

# ── SHARED STATE ──────────────────────────────────────────────────
llm_response    = ""        # latest LLM coaching text
llm_loading     = False     # True while waiting for Groq
advice_queue    = queue.Queue()  # thread-safe advice updates


def call_llm_async(posture_label, streak_min, session_total_min):
    """Calls LLM in a background thread so camera never freezes."""
    global llm_response, llm_loading
    llm_loading = True
    try:
        advice = get_coaching_advice(
            posture_label, streak_min, session_total_min)
        advice_queue.put(advice)
    except Exception as e:
        advice_queue.put(f"Error getting advice: {str(e)}")
    finally:
        llm_loading = False


# ── CAMERA SETUP ──────────────────────────────────────────────────
def get_camera():
    if args.laptop:
        print("Using laptop webcam (cv2.VideoCapture(0))")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        return cap, "laptop"
    else:
        # Raspberry Pi 5 — try picamera2 first
        try:
            from picamera2 import Picamera2
            import libcamera
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT),
                      "format": "RGB888"}
            )
            picam2.configure(config)
            picam2.start()
            print("Using Raspberry Pi Camera Module (picamera2)")
            return picam2, "picamera2"
        except Exception as e:
            print(f"picamera2 failed ({e}), falling back to cv2")
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            return cap, "laptop"


# ── FEATURE EXTRACTION ────────────────────────────────────────────
def get_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def compute_angle(v1, v2):
    d = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / d, -1, 1))))


def extract_features(landmarks, w, h):
    """
    Extracts 3 angle features from MediaPipe landmarks.
    Same computation as collect_posture.py — guaranteed match.
    """
    nose   = get_xy(landmarks, 0,  w, h)
    l_sh   = get_xy(landmarks, 11, w, h)
    r_sh   = get_xy(landmarks, 12, w, h)
    l_hip  = get_xy(landmarks, 23, w, h)
    r_hip  = get_xy(landmarks, 24, w, h)

    sh_mid  = (l_sh  + r_sh)  / 2
    hip_mid = (l_hip + r_hip) / 2
    up      = np.array([0, -1])

    neck_angle    = compute_angle(nose   - sh_mid,  up)
    spine_angle   = compute_angle(sh_mid - hip_mid, up)
    shoulder_tilt = abs(l_sh[1] - r_sh[1])

    return np.array([[neck_angle, spine_angle, shoulder_tilt]])


# ── OVERLAY DRAWING ───────────────────────────────────────────────
def draw_overlay(frame, label, stats, llm_loading):
    h, w = frame.shape[:2]
    color = LABEL_COLORS.get(label, (200, 200, 200))

    # posture label — top left
    cv2.putText(frame, label.upper().replace("_", " "),
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, color, 3)

    # streak timer
    streak = stats["current_streak_sec"]
    streak_str = f"Streak: {int(streak)//60:02d}:{int(streak)%60:02d}"
    cv2.putText(frame, streak_str,
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)

    # session stats — bottom left
    poor_pct = stats["poor_posture_pct"]
    session_min = stats["session_duration_sec"] / 60
    cv2.putText(frame,
                f"Session: {session_min:.1f}min  Poor: {poor_pct:.1f}%",
                (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (180, 180, 180), 1)

    # alert border — red pulsing when bad posture streak active
    if label != "correct" and streak > 30:
        thickness = 8
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)

    # LLM loading indicator
    if llm_loading:
        cv2.putText(frame, "Getting coaching advice...",
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (234, 179, 8), 1)

    return frame


# ── MAIN LOOP ─────────────────────────────────────────────────────
def main():
    global llm_response, llm_loading

    camera, cam_type = get_camera()
    print("\n=== POSTURE COACH STARTED ===")
    print("Press Q to quit")
    print("Press S to get session summary now")
    print("=" * 30)

    frame_count   = 0
    fps_timer     = time.time()
    fps           = 0
    current_label = "correct"

    while True:
        # ── capture frame ──
        if cam_type == "picamera2":
            frame = camera.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = camera.read()
            if not ret:
                print("Camera read failed")
                break

        h, w = frame.shape[:2]
        frame_count += 1

        # ── fps calc ──
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_timer)
            fps_timer = time.time()

        # ── mediapipe pose ──
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            # draw skeleton
            mp_draw.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0),
                                    thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 180, 0),
                                    thickness=2),
            )

            # extract features + classify
            features      = extract_features(landmarks, w, h)
            current_label = clf.predict(features)[0]
            confidence    = clf.predict_proba(features).max()

            # show confidence
            cv2.putText(frame,
                        f"conf: {confidence:.0%}",
                        (20, 115), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (150, 150, 150), 1)

            # update session tracker
            events = tracker.update(current_label)
            stats  = tracker.get_stats()

            # soft alert — print to terminal
            if events["soft_alert"]:
                streak_min = stats["current_streak_sec"] / 60
                print(f"\n[SOFT ALERT] {current_label} for "
                      f"{streak_min:.1f} min — consider adjusting")

            # hard alert — call LLM in background thread
            if events["hard_alert"] and not llm_loading:
                streak_min = stats["current_streak_sec"] / 60
                total_min  = stats["total_poor_sec"] / 60
                print(f"\n[HARD ALERT] Calling LLM for {current_label}...")
                t = threading.Thread(
                    target=call_llm_async,
                    args=(current_label, streak_min, total_min),
                    daemon=True,
                )
                t.start()

        else:
            # no person detected
            cv2.putText(frame, "No person detected",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (100, 100, 100), 2)
            stats = tracker.get_stats()

        # ── check advice queue ──
        try:
            llm_response = advice_queue.get_nowait()
            print(f"\n[COACHING ADVICE]\n{llm_response}\n")
        except queue.Empty:
            pass

        # ── draw overlay ──
        frame = draw_overlay(frame, current_label, stats, llm_loading)

        # fps display
        cv2.putText(frame, f"FPS: {fps:.0f}",
                    (w - 80, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (100, 100, 100), 1)

        cv2.imshow("Posture Coach", frame)

        # ── key handling ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nEnding session...")
            break

        if key == ord('s'):
            # manual session summary
            from rag_query import get_session_summary
            stats = tracker.get_stats()
            print("\n[SESSION SUMMARY]")
            print(get_session_summary(stats))

    # ── cleanup ──
    if cam_type == "picamera2":
        camera.stop()
    else:
        camera.release()

    cv2.destroyAllWindows()
    pose.close()

    # print final stats
    stats = tracker.get_stats()
    print("\n=== SESSION COMPLETE ===")
    print(f"Duration:         {stats['session_duration_sec']/60:.1f} min")
    print(f"Total poor posture: {stats['total_poor_sec']/60:.1f} min "
          f"({stats['poor_posture_pct']:.1f}%)")
    print(f"Streaks logged:   {stats['num_streaks']}")
    print("=" * 25)


if __name__ == "__main__":
    main()
