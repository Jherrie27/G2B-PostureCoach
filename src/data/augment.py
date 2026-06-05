"""Geometric perturbation of normalized landmark arrays.

The trick: we operate on landmarks (not features), regenerate features,
and label-preserve. Use after relabeling and before training.
"""
from typing import List
import numpy as np
import pandas as pd

from src.cv.landmarks import ORDERED_NAMES
from src.cv.normalizer import normalize
from src.cv.features import extract_features


def perturb_landmarks(n: np.ndarray, rng: np.random.Generator,
                      noise: float = 0.005,
                      rot_xy: float = 0.04,   # radians
                      rot_yz: float = 0.04
                      ) -> np.ndarray:
    """Apply small noise + small rotations. Returns new (9, 4) array."""
    out = n.copy()
    # Per-landmark noise
    out[:, :3] += rng.normal(0, noise, (9, 3))

    # Small camera roll (rotation in xy plane)
    theta_xy = rng.uniform(-rot_xy, rot_xy)
    Rxy = np.array([[np.cos(theta_xy), -np.sin(theta_xy), 0],
                    [np.sin(theta_xy),  np.cos(theta_xy), 0],
                    [0, 0, 1]], dtype=np.float32)
    # Small camera pitch (rotation in yz plane)
    theta_yz = rng.uniform(-rot_yz, rot_yz)
    Ryz = np.array([[1, 0, 0],
                    [0, np.cos(theta_yz), -np.sin(theta_yz)],
                    [0, np.sin(theta_yz),  np.cos(theta_yz)]], dtype=np.float32)
    out[:, :3] = (out[:, :3] @ Rxy.T) @ Ryz.T
    return out


def augment_features_row(landmarks: np.ndarray, label: str,
                         n_aug: int, rng: np.random.Generator) -> List[dict]:
    rows = []
    for _ in range(n_aug):
        perturbed = perturb_landmarks(landmarks, rng)
        normed = normalize(perturbed)
        if normed is None:
            continue
        feats = extract_features(normed)
        if feats is None:
            continue
        feats["label"] = label
        feats["source_file"] = "augmented"
        rows.append(feats)
    return rows
