"""Normalize a (9, 4) landmark array.

Origin: hip midpoint.
Scale:  shoulder width in the xy plane.
No rotation (we need shoulder tilt as a signal).
"""
from typing import Optional
import numpy as np
# Import index constants from the mediapipe-free constants module.
# (Re-exported here so features.py can keep importing them from normalizer.)
from .landmarks import (ORDERED_NAMES, NOSE, LE, RE, LS, RS, LEL, REL, LH, RH)

MIN_SHOULDER_WIDTH = 0.02   # in MediaPipe's normalized coords


def normalize(landmarks: np.ndarray) -> Optional[np.ndarray]:
    """landmarks shape (9, 4); returns shape (9, 4) or None if invalid."""
    if landmarks is None or landmarks.shape != (9, 4):
        return None
    hip_mid = 0.5 * (landmarks[LH, :3] + landmarks[RH, :3])
    centered_xyz = landmarks[:, :3] - hip_mid

    # Shoulder width in xy only (z is depth, different scale)
    sw = np.linalg.norm(landmarks[LS, :2] - landmarks[RS, :2])
    if sw < MIN_SHOULDER_WIDTH:
        return None
    scaled_xyz = centered_xyz / sw

    out = landmarks.copy()
    out[:, :3] = scaled_xyz
    return out
