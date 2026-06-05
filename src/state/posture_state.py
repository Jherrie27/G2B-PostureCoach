"""The structured object describing one moment of observed posture."""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Optional, Literal

PostureClass = Literal["correct_posture", "slouching", "neck_forward", "lean"]
CLASSES = ["correct_posture", "slouching", "neck_forward", "lean"]

# Normal thresholds for each measured value (matches rule_engine)
NORMAL_THRESHOLDS = {
    "ear_shoulder_offset_x": 0.20,
    "shoulder_roll_z": 0.08,
    "torso_compression_min": 1.55,
    "shoulder_tilt_abs_max": 3.0,
    "midline_deviation_max": 3.0,
    "craniovertebral_max": 12.0,
}


@dataclass
class PostureState:
    # --- Classification ---
    posture_class: PostureClass
    confidence: float
    class_probabilities: Dict[str, float]

    # --- Features ---
    ear_shoulder_offset_x: float
    craniovertebral_angle: float
    head_forward_offset_z: float
    nose_shoulder_offset_x: float
    shoulder_roll_z: float
    torso_compression_ratio: float
    elbow_forward_offset_z: float
    spine_angle_3d: float
    shoulder_tilt_angle: float
    hip_tilt_angle: float
    midline_deviation_angle: float
    nose_centerline_offset_x: float
    lateral_asymmetry_index: float
    landmark_confidence_mean: float

    # --- Quality ---
    is_reliable: bool

    # --- Timing ---
    timestamp: datetime
    posture_duration_sec: float
    session_duration_sec: float

    # --- Session aggregates ---
    posture_distribution: Dict[str, float] = field(default_factory=dict)
    correction_events: int = 0
    longest_bad_posture_streak_sec: float = 0.0

    @property
    def feature_deviations(self) -> Dict[str, float]:
        return {
            "forward_head": max(0.0, self.ear_shoulder_offset_x -
                                NORMAL_THRESHOLDS["ear_shoulder_offset_x"]),
            "shoulder_roll": max(0.0, self.shoulder_roll_z -
                                 NORMAL_THRESHOLDS["shoulder_roll_z"]),
            "torso_compression": max(
                0.0,
                NORMAL_THRESHOLDS["torso_compression_min"] - self.torso_compression_ratio),
            "shoulder_tilt": max(0.0, abs(self.shoulder_tilt_angle) -
                                 NORMAL_THRESHOLDS["shoulder_tilt_abs_max"]),
            "midline_deviation": max(0.0, abs(self.midline_deviation_angle) -
                                     NORMAL_THRESHOLDS["midline_deviation_max"]),
        }

    @property
    def primary_issue(self) -> str:
        devs = self.feature_deviations
        if not any(devs.values()):
            return "none"
        return max(devs, key=devs.get)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d
