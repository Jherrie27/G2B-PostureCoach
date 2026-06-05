"""14 posture features. Pure functions, fully deterministic."""
from typing import Dict, Optional
import numpy as np
from .normalizer import (NOSE, LE, RE, LS, RS, LEL, REL, LH, RH)

FEATURE_ORDER = [
    "ear_shoulder_offset_x",
    "craniovertebral_angle",
    "head_forward_offset_z",
    "nose_shoulder_offset_x",
    "shoulder_roll_z",
    "torso_compression_ratio",
    "elbow_forward_offset_z",
    "spine_angle_3d",
    "shoulder_tilt_angle",
    "hip_tilt_angle",
    "midline_deviation_angle",
    "nose_centerline_offset_x",
    "lateral_asymmetry_index",
    "landmark_confidence_mean",
]


def _deviation_from_vertical_deg(vec: np.ndarray) -> float:
    """Deviation of `vec` from the vertical AXIS, in degrees, in [0, 90].

    0 deg = aligned with vertical (either pole), growing as the vector tips
    toward horizontal. We fold the raw [0,180] angle with min(a, 180-a) because
    a torso/head vector points "up" (-y in image coords), so an upright posture
    must read ~0 deg deviation, not ~180. This is the convention the relabel
    thresholds (small = upright) and PostureState expect.
    """
    vertical = np.array([0.0, 1.0, 0.0])
    v = vec / (np.linalg.norm(vec) + 1e-9)
    cosang = float(np.clip(np.dot(v[:3], vertical), -1.0, 1.0))
    a = float(np.degrees(np.arccos(cosang)))
    return min(a, 180.0 - a)


def _line_tilt_from_horizontal_deg(p_from: np.ndarray, p_to: np.ndarray) -> float:
    """Signed tilt of the line p_from↔p_to from horizontal, folded to (-90, 90].

    A line and its reverse describe the same tilt, so we fold arctan2 (which
    lives in (-180, 180]) into (-90, 90]. This makes a LEVEL line read 0°
    (not 180°) and the magnitude grow with the tilt — which is what the
    'lean' detection needs. 0° = horizontal, sign = direction of tilt.
    """
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    ang = float(np.degrees(np.arctan2(dy, dx)))
    if ang > 90.0:
        ang -= 180.0
    elif ang <= -90.0:
        ang += 180.0
    return ang


def extract_features(n: np.ndarray) -> Optional[Dict[str, float]]:
    """Compute 14 features. `n` must be the normalized (9, 4) array."""
    if n is None or n.shape != (9, 4):
        return None

    nose, le, re_, ls, rs, lel, rel, lh, rh = (n[NOSE], n[LE], n[RE], n[LS],
                                               n[RS], n[LEL], n[REL], n[LH], n[RH])
    ear_mid = 0.5 * (le[:3] + re_[:3])
    sh_mid = 0.5 * (ls[:3] + rs[:3])
    hip_mid = 0.5 * (lh[:3] + rh[:3])  # this is now ≈ (0, 0, 0) after normalization
    elbow_mid = 0.5 * (lel[:3] + rel[:3])

    # 1. ear_shoulder_offset_x (positive = ears in front of shoulders → forward head)
    ear_shoulder_offset_x = float(ear_mid[2] - sh_mid[2])
    # NOTE: we use z (depth) for "forward" because MediaPipe x is left-right.
    # If your camera is side-on, you'd flip this — but for a frontal webcam,
    # forward head shows mostly as z-shift. We also include x-offset as feature 4.

    # 2. craniovertebral_angle (in sagittal-ish plane: yz)
    #    Vector from sh_mid to ear_mid in yz; angle from vertical (+y)
    yz_vec = np.array([0.0, ear_mid[1] - sh_mid[1], ear_mid[2] - sh_mid[2]])
    craniovertebral_angle = _deviation_from_vertical_deg(yz_vec)

    # 3. head_forward_offset_z (raw z delta)
    head_forward_offset_z = float(ear_mid[2] - sh_mid[2])

    # 4. nose_shoulder_offset_x (lateral head shift)
    nose_shoulder_offset_x = float(nose[0] - sh_mid[0])

    # 5. shoulder_roll_z (positive = shoulders in front of hips → roll forward)
    shoulder_roll_z = float(sh_mid[2] - hip_mid[2])

    # 6. torso_compression_ratio: |sh_y - hip_y| (because we normalized by sw)
    torso_compression_ratio = float(abs(sh_mid[1] - hip_mid[1]))

    # 7. elbow_forward_offset_z
    elbow_forward_offset_z = float(elbow_mid[2] - sh_mid[2])

    # 8. spine_angle_3d: angle of shoulder_mid relative to hip_mid (origin) in 3D vs +y
    spine_angle_3d = _deviation_from_vertical_deg(sh_mid)

    # 9. shoulder_tilt_angle: tilt of left↔right shoulder line vs horizontal,
    #    folded to (-90, 90] so a level line reads 0° and lean grows the magnitude.
    shoulder_tilt_angle = _line_tilt_from_horizontal_deg(ls[:3], rs[:3])

    # 10. hip_tilt_angle (same folded convention)
    hip_tilt_angle = _line_tilt_from_horizontal_deg(lh[:3], rh[:3])

    # 11. midline_deviation_angle: angle of sh_mid→hip_mid in xy vs vertical
    xy_vec = np.array([sh_mid[0] - hip_mid[0], sh_mid[1] - hip_mid[1], 0.0])
    midline_deviation_angle = _deviation_from_vertical_deg(xy_vec)

    # 12. nose_centerline_offset_x: nose vs shoulder midline x
    nose_centerline_offset_x = float(nose[0] - sh_mid[0])

    # 13. lateral_asymmetry_index: |left ear-to-shoulder| - |right ear-to-shoulder|
    left_es = float(np.linalg.norm(le[:3] - ls[:3]))
    right_es = float(np.linalg.norm(re_[:3] - rs[:3]))
    lateral_asymmetry_index = left_es - right_es

    # 14. landmark_confidence_mean
    landmark_confidence_mean = float(n[:, 3].mean())

    out = {
        "ear_shoulder_offset_x": ear_shoulder_offset_x,
        "craniovertebral_angle": craniovertebral_angle,
        "head_forward_offset_z": head_forward_offset_z,
        "nose_shoulder_offset_x": nose_shoulder_offset_x,
        "shoulder_roll_z": shoulder_roll_z,
        "torso_compression_ratio": torso_compression_ratio,
        "elbow_forward_offset_z": elbow_forward_offset_z,
        "spine_angle_3d": spine_angle_3d,
        "shoulder_tilt_angle": shoulder_tilt_angle,
        "hip_tilt_angle": hip_tilt_angle,
        "midline_deviation_angle": midline_deviation_angle,
        "nose_centerline_offset_x": nose_centerline_offset_x,
        "lateral_asymmetry_index": lateral_asymmetry_index,
        "landmark_confidence_mean": landmark_confidence_mean,
    }
    # Sanity: no NaNs
    if any(np.isnan(v) or np.isinf(v) for v in out.values()):
        return None
    return out


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert dict to ordered numpy vector for the classifier."""
    return np.array([features[k] for k in FEATURE_ORDER], dtype=np.float32)
