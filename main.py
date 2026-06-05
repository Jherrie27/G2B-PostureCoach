"""Headless mode: prints PostureState updates to stdout. No UI.

Previous (v2) CV+session loop is preserved in main_v2_backup.py.
    python main.py
"""
from src.cv.pipeline import PosturePipeline
from src.utils.config import mediapipe_complexity, camera_resolution
from src.utils.camera import open_working_camera


def main():
    w, h = camera_resolution()
    cap, cam_idx = open_working_camera(width=w, height=h)
    print(f"Using camera index {cam_idx}")
    pipe = PosturePipeline(model_complexity=mediapipe_complexity())
    last = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            s = pipe.step(frame)
            if s is None:
                continue
            if s.posture_class != last:
                print(f"[{s.timestamp.strftime('%H:%M:%S')}] {s.posture_class}  "
                      f"conf={s.confidence:.2f}  primary={s.primary_issue}")
                last = s.posture_class
    finally:
        cap.release()
        pipe.close()


if __name__ == "__main__":
    main()
