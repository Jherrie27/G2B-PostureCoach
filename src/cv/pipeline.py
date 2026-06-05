"""Orchestrates: frame in -> PostureState out."""
from datetime import datetime
from typing import Optional
import numpy as np

from src.cv.pose_extractor import PoseExtractor
from src.cv.normalizer import normalize
from src.cv.features import extract_features
from src.cv.classifier import PostureClassifier
from src.cv.smoother import TemporalSmoother
from src.cv.rule_engine import CLASSES
from src.state.posture_state import PostureState
from src.state.session_tracker import SessionTracker


class PosturePipeline:
    def __init__(self,
                 model_complexity: int = 1,
                 model_path: str = "models/posture_lgbm_v3.txt"):
        self.pose = PoseExtractor(model_complexity=model_complexity)
        self.classifier = PostureClassifier(model_path=model_path)
        self.smoother = TemporalSmoother(alpha=0.30, hysteresis_frames=8)
        self.session = SessionTracker()
        self._last_state: Optional[PostureState] = None

    def step(self, bgr_frame: np.ndarray) -> Optional[PostureState]:
        la = self.pose.extract(bgr_frame)
        if la is None:
            return self._hold_state(is_reliable=False)
        if not la.is_reliable:
            return self._hold_state(is_reliable=False)

        normed = normalize(la.data)
        if normed is None:
            return self._hold_state(is_reliable=False)

        feats = extract_features(normed)
        if feats is None:
            return self._hold_state(is_reliable=False)

        _, _, probs = self.classifier.predict(feats)
        probs_vec = np.array([probs[c] for c in CLASSES])
        smoothed_label, smoothed_probs = self.smoother.update(probs_vec)
        session = self.session.update(smoothed_label)

        state = PostureState(
            posture_class=smoothed_label,
            confidence=float(smoothed_probs[CLASSES.index(smoothed_label)]),
            class_probabilities={c: float(p)
                                 for c, p in zip(CLASSES, smoothed_probs)},
            **feats,
            is_reliable=True,
            timestamp=datetime.now(),
            posture_duration_sec=session["posture_duration_sec"],
            session_duration_sec=session["session_duration_sec"],
            posture_distribution=session["posture_distribution"],
            correction_events=session["correction_events"],
            longest_bad_posture_streak_sec=session["longest_bad_posture_streak_sec"],
        )
        self._last_state = state
        return state

    def _hold_state(self, is_reliable: bool) -> Optional[PostureState]:
        if self._last_state is None:
            return None
        # Return last state but flag unreliable
        prev = self._last_state
        prev.is_reliable = is_reliable
        prev.timestamp = datetime.now()
        return prev

    def close(self):
        self.pose.close()
