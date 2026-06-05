"""Generate exaggerated landmark samples for under-represented classes.

Takes correct-posture landmark arrays and mathematically deforms them into
class-typical exaggerated versions. This is how you avoid recollecting data.

G2B fully-synthetic path: because this project has NO raw landmark dumps,
`make_correct_pool` synthesizes an anatomically-plausible pool of seated
'correct' posture landmark arrays from a canonical template. The other three
classes are then deformed from that pool.
"""
from typing import Iterator
import numpy as np

from src.cv.landmarks import ORDERED_NAMES

NOSE = ORDERED_NAMES.index("nose")
LE = ORDERED_NAMES.index("left_ear")
RE = ORDERED_NAMES.index("right_ear")
LS = ORDERED_NAMES.index("left_shoulder")
RS = ORDERED_NAMES.index("right_shoulder")
LEL = ORDERED_NAMES.index("left_elbow")
REL = ORDERED_NAMES.index("right_elbow")
LH = ORDERED_NAMES.index("left_hip")
RH = ORDERED_NAMES.index("right_hip")

# Canonical seated 'correct' posture in MediaPipe-like normalized image coords.
# x in [0,1] left→right, y in [0,1] top→bottom, z ≈ 0 (depth, + = away from camera).
# Order matches ORDERED_NAMES: nose, l_ear, r_ear, l_sh, r_sh, l_el, r_el, l_hip, r_hip.
_CANONICAL_CORRECT = np.array([
    [0.500, 0.190, 0.00, 1.0],   # nose
    [0.545, 0.205, 0.02, 1.0],   # left_ear  (slightly behind nose in z)
    [0.455, 0.205, 0.02, 1.0],   # right_ear
    [0.600, 0.335, 0.00, 1.0],   # left_shoulder  (gap to hips -> compression ~1.6)
    [0.400, 0.335, 0.00, 1.0],   # right_shoulder
    [0.640, 0.510, 0.01, 1.0],   # left_elbow
    [0.360, 0.510, 0.01, 1.0],   # right_elbow
    [0.560, 0.660, 0.00, 1.0],   # left_hip
    [0.440, 0.660, 0.00, 1.0],   # right_hip
], dtype=np.float32)


def make_correct_pool(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Synthesize a pool of plausible 'correct' posture landmark arrays.

    Returns shape (n_samples, 9, 4). Applies realistic per-sample variation:
    global translation/scale (camera distance/position), small body sway,
    mild left/right asymmetry, and per-landmark jitter.
    """
    pool = np.empty((n_samples, 9, 4), dtype=np.float32)
    for i in range(n_samples):
        s = _CANONICAL_CORRECT.copy()

        # Global scale (camera distance) and translation (camera framing)
        scale = rng.uniform(0.85, 1.15)
        cx, cy = 0.5, 0.5
        s[:, 0] = cx + (s[:, 0] - cx) * scale + rng.uniform(-0.05, 0.05)
        s[:, 1] = cy + (s[:, 1] - cy) * scale + rng.uniform(-0.05, 0.05)

        # Small natural sway/tilt of the whole upper body (in xy), well below
        # the 'lean' threshold so these stay clearly 'correct'.
        theta = rng.uniform(-0.03, 0.03)  # ~1.7 deg
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta),  np.cos(theta), 0],
                      [0, 0, 1]], dtype=np.float32)
        pivot = 0.5 * (s[LH, :3] + s[RH, :3])
        s[:, :3] = (s[:, :3] - pivot) @ R.T + pivot

        # Mild depth variation of the head (natural, kept sub-threshold so
        # 'correct' samples don't drift toward neck_forward)
        head_z = rng.uniform(-0.02, 0.02)
        s[[NOSE, LE, RE], 2] += head_z

        # Per-landmark jitter + tiny visibility noise
        s[:, :3] += rng.normal(0, 0.006, (9, 3)).astype(np.float32)
        s[:, 3] = np.clip(1.0 - np.abs(rng.normal(0, 0.03, 9)), 0.6, 1.0)

        pool[i] = s
    return pool


def synth_slouching(n: np.ndarray, severity: float, rng) -> np.ndarray:
    """Shoulders/elbows forward and slightly down. Head also drags forward."""
    out = n.copy()
    # Shoulders forward in z, slightly down in y
    for idx in [LS, RS]:
        out[idx, 2] += severity * 0.22 + rng.normal(0, 0.01)
        out[idx, 1] += severity * 0.04 + rng.normal(0, 0.005)
    # Elbows track shoulders forward
    for idx in [LEL, REL]:
        out[idx, 2] += severity * 0.15 + rng.normal(0, 0.01)
    # Head follows
    for idx in [NOSE, LE, RE]:
        out[idx, 2] += severity * 0.12 + rng.normal(0, 0.01)
        out[idx, 1] += severity * 0.02 + rng.normal(0, 0.005)
    return out


def synth_neck_forward(n: np.ndarray, severity: float, rng) -> np.ndarray:
    """Head forward in z; shoulders stay put."""
    out = n.copy()
    for idx in [NOSE, LE, RE]:
        out[idx, 2] += severity * 0.28 + rng.normal(0, 0.015)
        out[idx, 0] += rng.normal(0, 0.005)  # tiny lateral jitter
    return out


def synth_lean(n: np.ndarray, severity: float, rng,
               direction: str = "left") -> np.ndarray:
    """Tilt upper body in xy plane."""
    out = n.copy()
    sign = -1.0 if direction == "left" else 1.0
    angle = sign * severity * 0.18    # ~10° max
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle),  np.cos(angle), 0],
                  [0, 0, 1]], dtype=np.float32)
    pivot_idx_y = float(0.5 * (n[LS, 1] + n[RS, 1]))  # roughly chest height
    for idx in [NOSE, LE, RE, LS, RS, LEL, REL]:
        pt = out[idx, :3].copy()
        pt[1] -= pivot_idx_y
        pt = R @ pt
        pt[1] += pivot_idx_y
        out[idx, :3] = pt
    return out


def synthesize_from_correct(correct_landmarks_array: np.ndarray,
                            target_class: str,
                            n_samples: int,
                            rng: np.random.Generator
                            ) -> Iterator[np.ndarray]:
    """Yield n_samples synthesized landmark arrays for target_class."""
    sources = correct_landmarks_array  # shape (N, 9, 4) — pool of correct posture
    if sources.shape[0] == 0:
        raise ValueError("Need at least one correct-posture landmark sample")
    for _ in range(n_samples):
        base = sources[rng.integers(0, sources.shape[0])]
        severity = rng.uniform(0.4, 1.0)
        if target_class == "slouching":
            yield synth_slouching(base, severity, rng)
        elif target_class == "neck_forward":
            yield synth_neck_forward(base, severity, rng)
        elif target_class == "lean":
            direction = rng.choice(["left", "right"])
            yield synth_lean(base, severity, rng, direction=direction)
        else:
            raise ValueError(f"Cannot synthesize class {target_class}")
