"""Robust webcam opening.

On Windows some indices (often a virtual/IR camera) open but can't grab frames
(MSMF error -1072875772). This picks the first index that actually yields a frame.
Override with env var G2B_CAMERA_INDEX to force a specific index.
"""
import os
from typing import Optional, Tuple
import cv2


def _configure(cap, width: int, height: int):
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def _yields_frame(cap, tries: int = 5) -> bool:
    for _ in range(tries):
        ok, frame = cap.read()
        if ok and frame is not None:
            return True
    return False


def open_working_camera(width: int = 640, height: int = 480,
                        preferred: Optional[int] = None,
                        max_index: int = 4) -> Tuple[cv2.VideoCapture, int]:
    """Return (cap, index) for the first camera that actually delivers frames.

    Order: G2B_CAMERA_INDEX (if set) or `preferred` first, then 0..max_index-1.
    Raises RuntimeError if none work.
    """
    env = os.environ.get("G2B_CAMERA_INDEX")
    forced = int(env) if env is not None else preferred

    order = []
    if forced is not None:
        order.append(forced)
    order += [i for i in range(max_index) if i != forced]

    for idx in order:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            _configure(cap, width, height)
            if _yields_frame(cap):
                return cap, idx
        cap.release()
    raise RuntimeError(
        f"No working camera found (tried indices {order}). "
        f"Set G2B_CAMERA_INDEX to the right one, or check the camera isn't in use."
    )
