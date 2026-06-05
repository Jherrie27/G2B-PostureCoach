"""MediaPipe Pose wrapper that returns only the landmarks we use."""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import mediapipe as mp

# Landmark constants live in landmarks.py (no mediapipe import) and are
# re-exported here for backward compatibility.
from .landmarks import (LANDMARK_INDICES, ORDERED_NAMES,
                        KEY_FOR_RELIABILITY, VISIBILITY_MIN)


@dataclass
class LandmarkArray:
    data: np.ndarray   # shape (9, 4) — x, y, z, visibility for each named landmark
    is_reliable: bool


class PoseExtractor:
    def __init__(self, model_complexity: int = 1, static_image_mode: bool = False):
        # model_complexity 0 = Lite (fastest, use on Pi 5)
        # model_complexity 1 = Full (use on dev machine)
        self.pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            static_image_mode=static_image_mode,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=False,
        )

    def extract(self, bgr_frame: np.ndarray) -> Optional[LandmarkArray]:
        rgb = bgr_frame[..., ::-1]
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return None
        all_lm = result.pose_landmarks.landmark
        rows = []
        for name in ORDERED_NAMES:
            lm = all_lm[LANDMARK_INDICES[name]]
            rows.append([lm.x, lm.y, lm.z, lm.visibility])
        data = np.array(rows, dtype=np.float32)

        # Reliability check on key landmarks
        idxs = [ORDERED_NAMES.index(n) for n in KEY_FOR_RELIABILITY]
        reliable = bool(np.all(data[idxs, 3] >= VISIBILITY_MIN))
        return LandmarkArray(data=data, is_reliable=reliable)

    def close(self):
        self.pose.close()
