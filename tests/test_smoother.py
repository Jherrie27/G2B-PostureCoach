import numpy as np
from src.cv.smoother import TemporalSmoother


def _probs(idx):
    p = np.array([0.05, 0.05, 0.05, 0.05])
    p[idx] = 0.85
    return p


def test_initial_state():
    s = TemporalSmoother()
    assert s.current_label == "correct_posture"


def test_single_flip_does_not_change_label():
    s = TemporalSmoother(hysteresis_frames=8)
    s.update(_probs(0))  # warm up to correct
    s.update(_probs(1))  # one slouch frame
    assert s.current_label == "correct_posture"


def test_sustained_change_flips_label():
    s = TemporalSmoother(hysteresis_frames=8)
    for _ in range(20):
        s.update(_probs(0))  # warm up to correct
    for _ in range(20):
        s.update(_probs(1))  # sustained slouch
    assert s.current_label == "slouching"


def test_ema_smooths_noise():
    s = TemporalSmoother(alpha=0.3)
    # Alternate noisy predictions
    for i in range(30):
        idx = 0 if i % 2 == 0 else 1
        s.update(_probs(idx))
    # Should hold initial label since neither class wins consistently
    assert s.current_label == "correct_posture"
