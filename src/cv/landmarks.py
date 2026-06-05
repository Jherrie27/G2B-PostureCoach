"""Landmark index constants — NO heavy imports.

Kept separate from pose_extractor (which imports mediapipe) so that the
feature / data / training code can use these constants without loading the
MediaPipe native DLLs. This avoids both slow startup and the intermittent
'_framework_bindings DLL load failed' issue on Windows for non-CV code paths.
"""

# Indices in MediaPipe's 33-landmark layout
LANDMARK_INDICES = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_hip": 23,
    "right_hip": 24,
}
ORDERED_NAMES = list(LANDMARK_INDICES.keys())  # fixed order for consistent rows
KEY_FOR_RELIABILITY = ["left_ear", "right_ear", "left_shoulder",
                       "right_shoulder", "left_hip", "right_hip"]
VISIBILITY_MIN = 0.5

# Pre-resolved row indices into the (9, 4) array (order == ORDERED_NAMES)
NOSE = ORDERED_NAMES.index("nose")
LE = ORDERED_NAMES.index("left_ear")
RE = ORDERED_NAMES.index("right_ear")
LS = ORDERED_NAMES.index("left_shoulder")
RS = ORDERED_NAMES.index("right_shoulder")
LEL = ORDERED_NAMES.index("left_elbow")
REL = ORDERED_NAMES.index("right_elbow")
LH = ORDERED_NAMES.index("left_hip")
RH = ORDERED_NAMES.index("right_hip")
