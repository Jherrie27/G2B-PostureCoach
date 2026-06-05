"""EMA on probabilities + hysteresis on label transitions."""
from typing import Optional, Tuple
import numpy as np

from src.cv.rule_engine import CLASSES


class TemporalSmoother:
    def __init__(self, alpha: float = 0.30, hysteresis_frames: int = 8):
        self.alpha = alpha
        self.hysteresis = hysteresis_frames
        self.smoothed: Optional[np.ndarray] = None
        self.current_label: str = "correct_posture"
        self.candidate_label: Optional[str] = None
        self.candidate_streak: int = 0

    def update(self, raw_probs: np.ndarray) -> Tuple[str, np.ndarray]:
        raw_probs = np.asarray(raw_probs, dtype=np.float32)
        # 1. EMA
        if self.smoothed is None:
            self.smoothed = raw_probs.copy()
        else:
            self.smoothed = (self.alpha * raw_probs +
                             (1.0 - self.alpha) * self.smoothed)

        # 2. Argmax + hysteresis
        argmax_idx = int(np.argmax(self.smoothed))
        argmax_label = CLASSES[argmax_idx]

        if argmax_label == self.current_label:
            self.candidate_label = None
            self.candidate_streak = 0
        else:
            if argmax_label == self.candidate_label:
                self.candidate_streak += 1
            else:
                self.candidate_label = argmax_label
                self.candidate_streak = 1
            if self.candidate_streak >= self.hysteresis:
                self.current_label = argmax_label
                self.candidate_label = None
                self.candidate_streak = 0

        return self.current_label, self.smoothed.copy()

    def reset(self):
        self.smoothed = None
        self.current_label = "correct_posture"
        self.candidate_label = None
        self.candidate_streak = 0
