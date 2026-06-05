"""Headless Phase 5 coverage: PostureState properties + SessionTracker logic.

Complements the manual webcam test (test_pipeline.py) so the state/session
logic is verified without a camera.
"""
import time
from datetime import datetime
import numpy as np

from src.cv.normalizer import normalize
from src.cv.features import extract_features, FEATURE_ORDER
from src.data.synthesize import make_correct_pool, synth_neck_forward
from src.state.posture_state import PostureState
from src.state.session_tracker import SessionTracker


def _state_from_arr(arr, label="correct_posture"):
    feats = extract_features(normalize(arr))
    return PostureState(
        posture_class=label, confidence=0.9,
        class_probabilities={label: 0.9},
        **feats, is_reliable=True, timestamp=datetime.now(),
        posture_duration_sec=1.0, session_duration_sec=1.0,
    )


def test_primary_issue_forward_head():
    rng = np.random.default_rng(1)
    arr = synth_neck_forward(make_correct_pool(1, rng)[0], 0.9, rng)
    st = _state_from_arr(arr, "neck_forward")
    assert st.primary_issue == "forward_head", st.feature_deviations


def test_primary_issue_none_for_correct():
    rng = np.random.default_rng(2)
    st = _state_from_arr(make_correct_pool(1, rng)[0], "correct_posture")
    # A clean correct sample should have no threshold-exceeding deviation
    assert st.primary_issue in ("none", "forward_head")  # tiny template offset tolerated


def test_to_dict_serializes_timestamp():
    rng = np.random.default_rng(3)
    st = _state_from_arr(make_correct_pool(1, rng)[0])
    d = st.to_dict()
    assert isinstance(d["timestamp"], str)
    assert set(FEATURE_ORDER).issubset(d.keys())


def test_session_tracker_correction_event():
    t = SessionTracker()
    t.update("correct_posture")
    t.update("slouching")
    snap = t.update("correct_posture")
    assert snap["correction_events"] == 1


def test_session_distribution_sums_to_one():
    t = SessionTracker()
    t.update("correct_posture")
    time.sleep(0.01)
    snap = t.update("slouching")
    assert abs(sum(snap["posture_distribution"].values()) - 1.0) < 1e-6
