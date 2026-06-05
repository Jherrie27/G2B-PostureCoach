"""Soft-probability rule engine. Returns per-class probability based on
clinical thresholds. Used to fuse with LGBM output for robustness.
"""
from typing import Dict
import numpy as np

CLASSES = ["correct_posture", "slouching", "neck_forward", "lean"]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def rule_probs(f: Dict[str, float]) -> np.ndarray:
    """Returns [p_correct, p_slouching, p_neck_forward, p_lean]."""
    # Soft scores in [0, 1]
    lean_score = max(
        _sigmoid((abs(f["shoulder_tilt_angle"]) - 4.0) / 2.0),
        _sigmoid((f["midline_deviation_angle"] - 4.0) / 2.0),
    )
    neck_score = max(
        _sigmoid((f["ear_shoulder_offset_x"] - 0.25) / 0.10),
        _sigmoid((f["craniovertebral_angle"] - 12.0) / 4.0),
    )
    slouch_score = max(
        _sigmoid((f["shoulder_roll_z"] - 0.10) / 0.05),
        _sigmoid(((1.45 - f["torso_compression_ratio"])) / 0.10),
    )
    # "Correct" is the residual
    correct_score = 1.0 - max(lean_score, neck_score, slouch_score)
    correct_score = max(correct_score, 0.01)

    probs = np.array([correct_score, slouch_score, neck_score, lean_score],
                     dtype=np.float32)
    return probs / probs.sum()
