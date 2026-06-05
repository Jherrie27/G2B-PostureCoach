import numpy as np
from src.cv.pose_extractor import ORDERED_NAMES
from src.cv.normalizer import normalize
from src.cv.features import extract_features, FEATURE_ORDER


def _make_correct_landmarks():
    """Synthetic 'correct posture' landmark set."""
    # x, y, z, visibility
    n = np.array([
        [0.5, 0.20, 0.0, 1.0],   # nose
        [0.55, 0.22, 0.0, 1.0],  # left_ear
        [0.45, 0.22, 0.0, 1.0],  # right_ear
        [0.60, 0.35, 0.0, 1.0],  # left_shoulder
        [0.40, 0.35, 0.0, 1.0],  # right_shoulder
        [0.65, 0.50, 0.0, 1.0],  # left_elbow
        [0.35, 0.50, 0.0, 1.0],  # right_elbow
        [0.55, 0.65, 0.0, 1.0],  # left_hip
        [0.45, 0.65, 0.0, 1.0],  # right_hip
    ], dtype=np.float32)
    return n


def test_normalize_centers_hips():
    n = _make_correct_landmarks()
    out = normalize(n)
    hip_mid = 0.5 * (out[7, :3] + out[8, :3])
    assert np.allclose(hip_mid, 0, atol=1e-5)


def test_features_for_correct_posture():
    n = _make_correct_landmarks()
    norm = normalize(n)
    f = extract_features(norm)
    assert f is not None
    # For our synthetic-correct sample:
    assert abs(f["shoulder_tilt_angle"]) < 5, f
    assert abs(f["ear_shoulder_offset_x"]) < 0.1, f
    assert abs(f["shoulder_roll_z"]) < 0.05, f


def test_features_for_forward_head():
    n = _make_correct_landmarks()
    # Push ears forward in z
    n[1, 2] = 0.10
    n[2, 2] = 0.10
    norm = normalize(n)
    f = extract_features(norm)
    assert f["ear_shoulder_offset_x"] > 0.15, f["ear_shoulder_offset_x"]
    assert f["craniovertebral_angle"] > 10, f["craniovertebral_angle"]


def test_features_for_lean():
    n = _make_correct_landmarks()
    # Tilt shoulders
    n[3, 1] = 0.32  # left shoulder up
    n[4, 1] = 0.38  # right shoulder down
    norm = normalize(n)
    f = extract_features(norm)
    assert abs(f["shoulder_tilt_angle"]) > 5, f["shoulder_tilt_angle"]


def test_feature_order_matches():
    f = extract_features(normalize(_make_correct_landmarks()))
    assert list(f.keys()) == FEATURE_ORDER
