"""Tracks session-level aggregates from a stream of (label, timestamp) updates."""
from collections import defaultdict, deque
from datetime import datetime
from typing import Optional, Dict
import time

from src.state.posture_state import CLASSES


class SessionTracker:
    def __init__(self):
        self.session_start: float = time.time()
        self.current_label: Optional[str] = None
        self.current_label_start: float = self.session_start
        self.time_per_class: Dict[str, float] = defaultdict(float)
        self.correction_events: int = 0
        self.longest_bad_streak: float = 0.0
        self._last_bad_streak_start: Optional[float] = None

    def update(self, label: str) -> dict:
        now = time.time()

        if self.current_label is None:
            # First call
            self.current_label = label
            self.current_label_start = now
            if label != "correct_posture":
                self._last_bad_streak_start = now
            return self._snapshot(now, label)

        if label != self.current_label:
            # Class transition: bank time on previous class
            elapsed = now - self.current_label_start
            self.time_per_class[self.current_label] += elapsed

            # Correction-to-correct event
            if (label == "correct_posture"
                    and self.current_label != "correct_posture"):
                self.correction_events += 1
                if self._last_bad_streak_start is not None:
                    streak = now - self._last_bad_streak_start
                    self.longest_bad_streak = max(self.longest_bad_streak, streak)
                    self._last_bad_streak_start = None

            # Entering a bad posture
            if label != "correct_posture" and self.current_label == "correct_posture":
                self._last_bad_streak_start = now

            self.current_label = label
            self.current_label_start = now

        return self._snapshot(now, label)

    def _snapshot(self, now: float, label: str) -> dict:
        # Add in-progress time for the current label
        live_per_class = dict(self.time_per_class)
        live_per_class[label] = (live_per_class.get(label, 0.0) +
                                 now - self.current_label_start)
        total = sum(live_per_class.values()) or 1.0
        distribution = {c: live_per_class.get(c, 0.0) / total for c in CLASSES}
        return {
            "posture_duration_sec": now - self.current_label_start,
            "session_duration_sec": now - self.session_start,
            "posture_distribution": distribution,
            "correction_events": self.correction_events,
            "longest_bad_posture_streak_sec": self.longest_bad_streak,
        }
