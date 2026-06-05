"""
session.py
Posture Correction Coach — Session Tracker
Tracks bad posture streaks and cumulative session stats.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SessionTracker:
    # thresholds
    soft_alert_sec: int = 120    # 2 min  → soft ping
    hard_alert_sec: int = 600    # 10 min → LLM coaching trigger

    # session totals
    session_start:       float = field(default_factory=time.time)
    session_poor_total:  float = 0.0   # cumulative bad posture seconds

    # current streak
    current_label:       str   = "correct"
    bad_posture_start:   Optional[float] = None
    current_streak_sec:  float = 0.0
    last_alert_time:     float = 0.0   # prevents repeated alerts

    # history
    streak_log: list = field(default_factory=list)
    alert_log:  list = field(default_factory=list)

    # flags
    soft_alert_fired: bool = False
    hard_alert_fired: bool = False

    def update(self, label: str) -> dict:
        """
        Call once per frame with the current posture label.
        Returns a dict of events that fired this frame.
        """
        now    = time.time()
        events = {
            "soft_alert": False,
            "hard_alert": False,
            "streak_closed": False,
        }

        if label != "correct":
            # bad posture detected
            if self.bad_posture_start is None:
                self.bad_posture_start  = now
                self.soft_alert_fired   = False
                self.hard_alert_fired   = False

            self.current_streak_sec = now - self.bad_posture_start

            # soft alert at 2 minutes
            if (self.current_streak_sec >= self.soft_alert_sec
                    and not self.soft_alert_fired):
                self.soft_alert_fired  = True
                events["soft_alert"]   = True
                self.alert_log.append({
                    "type":     "soft",
                    "label":    label,
                    "streak":   self.current_streak_sec,
                    "time":     now,
                })

            # hard alert at 10 minutes
            if (self.current_streak_sec >= self.hard_alert_sec
                    and not self.hard_alert_fired):
                self.hard_alert_fired  = True
                events["hard_alert"]   = True
                self.alert_log.append({
                    "type":     "hard",
                    "label":    label,
                    "streak":   self.current_streak_sec,
                    "time":     now,
                })

        else:
            # good posture — close any active streak
            if self.bad_posture_start is not None:
                duration = now - self.bad_posture_start
                self.session_poor_total += duration
                self.streak_log.append({
                    "label":       self.current_label,
                    "start":       self.bad_posture_start,
                    "duration_sec": round(duration, 1),
                })
                self.bad_posture_start  = None
                self.current_streak_sec = 0.0
                self.soft_alert_fired   = False
                self.hard_alert_fired   = False
                events["streak_closed"] = True

        self.current_label = label
        return events

    def get_stats(self) -> dict:
        now              = time.time()
        session_duration = now - self.session_start

        # include current active streak in total
        active_streak = self.current_streak_sec if self.bad_posture_start else 0.0
        total_poor    = self.session_poor_total + active_streak

        poor_pct = (total_poor / session_duration * 100) if session_duration > 0 else 0.0

        return {
            "session_duration_sec": round(session_duration, 1),
            "total_poor_sec":       round(total_poor, 1),
            "poor_posture_pct":     round(poor_pct, 1),
            "current_streak_sec":   round(self.current_streak_sec, 1),
            "current_label":        self.current_label,
            "num_streaks":          len(self.streak_log),
            "streak_log":           self.streak_log,
            "alert_log":            self.alert_log,
        }

    def format_time(self, seconds: float) -> str:
        """Converts seconds to MM:SS string."""
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m:02d}:{s:02d}"
