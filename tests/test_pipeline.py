"""Live test of the full CV pipeline. Smoothing visible. Run for 60s.

MANUAL test — run yourself in front of the webcam:
    python tests/test_pipeline.py
Not collected as an automated test (no test_* functions); needs a camera.
"""
import time
import cv2

from src.cv.pipeline import PosturePipeline


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    pipe = PosturePipeline(model_complexity=1)
    last_label = None
    start = time.time()

    while time.time() - start < 60:
        ok, frame = cap.read()
        if not ok:
            break
        state = pipe.step(frame)
        if state is None:
            print("[no detection]")
            continue
        if state.posture_class != last_label:
            print(f"\n>>> CHANGED to {state.posture_class}  "
                  f"(conf={state.confidence:.2f}, primary={state.primary_issue})")
            last_label = state.posture_class
        else:
            print(".", end="", flush=True)
    cap.release()
    pipe.close()


if __name__ == "__main__":
    main()
