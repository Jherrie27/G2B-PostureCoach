# G2B Posture Correction Coach — Step-by-Step Execution Guide

**Companion to:** `G2B_POSTURE_REDESIGN.md`
**Audience:** You, executing this linearly with no architectural decisions left to make.
**Estimated time:** 18–25 working hours across 10 phases.
**Fastest-path-to-demo annotation:** sections marked **`[FAST PATH]`** are critical; **`[POLISH]`** can be deferred to the last day.

---

## How to use this document

Work top to bottom. Don't skip phases. After every phase there's a **Verification block** — if it fails, fix it before moving on. If the Verification passes, commit your work to git with the suggested commit message and proceed.

Every file path is given relative to your project root. Every command assumes you're at the project root unless stated otherwise.

---

## Assumed current project layout

Before any changes, your project looks roughly like this:

```
g2b_posture_coach/
├── main.py                    # entry point (CV + chat loop)
├── app.py                     # Streamlit UI
├── session.py                 # session state mgmt
├── rag_query.py               # RAG retrieval + Groq call
├── build_rag.py               # ChromaDB indexer
├── retrain.py                 # current RF training
├── collect_posture.py         # webcam → CSV landmark dumper
├── Test.py                    # ad-hoc tests
├── posture_classifier.pkl     # v1 RF
├── posture_classifier_v2.pkl  # v2 RF
├── CV/                        # landmark CSV dumps
│   ├── correct_*.csv
│   ├── slouch_*.csv
│   ├── neck_*.csv
│   └── lean_*.csv
└── rag_db_v2/                 # ChromaDB persistent dir
```

If your layout differs, adjust paths as you go but keep the same file *roles*.

---

## Files I am assuming exist and what I'm assuming about them

| File | Assumed content | What I need from you if I'm wrong |
|---|---|---|
| `collect_posture.py` | Reads webcam, runs MediaPipe Pose, writes per-frame landmark rows to a CSV. Columns: `landmark_<i>_<x\|y\|z\|v>` for i=0..32, plus a `label` column. | If columns are named differently, see Phase 2.2 — single-function rename. |
| `CV/*.csv` | One CSV per session, ~hundreds of rows per file. Each row = one frame's 33 landmarks plus the label. | If labels are stored in the filename rather than a column, see Phase 3.1 alt path. |
| `retrain.py` | Loads CSVs, picks 5 landmarks, computes 3 features (`neck_angle`, `spine_angle`, `shoulder_tilt`), trains RF, saves `.pkl`. | If it uses sklearn pipelines, we'll mirror the structure. |
| `rag_query.py` | Takes a string, embeds, queries ChromaDB, builds a prompt, calls Groq Llama. | If chunks have no metadata, Phase 6 adds it. |
| `build_rag.py` | Reads physiotherapy text, chunks, embeds, writes to ChromaDB at `./rag_db_v2/`. | Same as above. |
| `main.py` / `app.py` | One launches CV + RAG + UI together; the other is Streamlit. | We'll wire `app.py` to use background workers from Phase 7. |

---

## Final target project layout (after this guide)

```
g2b_posture_coach/
├── main.py                              [UPDATE — wires new pipeline]
├── app.py                               [UPDATE — uses workers + shared state]
├── session.py                           [UPDATE — uses PostureState]
├── requirements.txt                     [REPLACE]
├── README.md                            [UPDATE — Medusa→G2B]
├── src/
│   ├── cv/
│   │   ├── __init__.py
│   │   ├── pose_extractor.py            [NEW — wraps MediaPipe with 9 landmarks]
│   │   ├── normalizer.py                [NEW]
│   │   ├── features.py                  [NEW — 14 features]
│   │   ├── rule_engine.py               [NEW]
│   │   ├── classifier.py                [NEW — LGBM + rules]
│   │   ├── smoother.py                  [NEW]
│   │   └── pipeline.py                  [NEW — orchestrator]
│   ├── data/
│   │   ├── __init__.py
│   │   ├── relabel.py                   [NEW]
│   │   ├── augment.py                   [NEW]
│   │   ├── synthesize.py                [NEW]
│   │   └── build_dataset.py             [NEW — runs all of the above]
│   ├── state/
│   │   ├── __init__.py
│   │   └── posture_state.py             [NEW]
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── tagger.py                    [NEW — LLM-tags chunks]
│   │   ├── retriever.py                 [NEW — hybrid retrieval]
│   │   ├── prompt_builder.py            [NEW]
│   │   └── grounding_check.py           [NEW]
│   ├── workers/
│   │   ├── __init__.py
│   │   ├── shared_state.py              [NEW]
│   │   ├── cv_worker.py                 [NEW]
│   │   └── chat_worker.py               [NEW]
│   └── utils/
│       ├── __init__.py
│       └── timing.py                    [NEW — perf counters]
├── train_lgbm.py                        [NEW — top-level training script]
├── models/
│   ├── posture_lgbm_v3.txt              [NEW]
│   └── feature_order.json               [NEW]
├── data/
│   ├── raw_landmarks/                   [symlink or move from CV/]
│   ├── relabeled/                       [NEW]
│   └── augmented/                       [NEW]
├── notebooks/
│   ├── 01_inspect_features.ipynb        [NEW]
│   └── 02_eval_classifier.ipynb         [NEW]
├── rag_db_v3/                           [NEW — rebuilt with metadata]
└── tests/
    ├── test_features.py                 [NEW]
    ├── test_pipeline.py                 [NEW]
    └── test_smoother.py                 [NEW]
```

The old files stay (`posture_classifier.pkl`, `posture_classifier_v2.pkl`, `rag_db_v2/`, `CV/`) — don't delete anything until the new system is verified end-to-end.

---

# PHASE 1 — Environment Setup `[FAST PATH]`

**Objective:** Fresh, pinned environment with no protobuf errors. Project renamed to G2B. New directory structure scaffolded.
**Estimated time:** 1.5–2 hours.

## 1.1 Create a fresh virtual environment

**Windows (PowerShell):**

```powershell
# From project root
deactivate            # if currently in another venv
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python --version      # should show 3.11.x; if not, install Python 3.11.9 first
```

**Linux / Pi 5 (later):**

```bash
sudo rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python --version
```

## 1.2 Write the new `requirements.txt`

**File:** `requirements.txt` (replace whatever is there)

```
# === Core CV ===
mediapipe==0.10.14
opencv-python==4.10.0.84
numpy==1.26.4
protobuf==4.25.3

# === ML ===
lightgbm==4.3.0
scikit-learn==1.5.0
pandas==2.2.2
scipy==1.13.1
joblib==1.4.2

# === RAG ===
chromadb==0.5.5
sentence-transformers==3.0.1
rank-bm25==0.2.2

# === LLM ===
groq==0.9.0
tiktoken==0.7.0

# === App ===
streamlit==1.36.0
python-dotenv==1.0.1

# === Dev ===
pytest==8.2.2
matplotlib==3.9.0
seaborn==0.13.2
jupyter==1.0.0
```

Install:

```bash
pip install --no-cache-dir -r requirements.txt
```

## 1.3 Verify the install with one command

**File:** `tests/test_env.py` (create)

```python
"""Smoke test: every critical import must succeed and emit no protobuf warning."""
import warnings
warnings.filterwarnings("error", message=".*protobuf.*")  # promote to error

import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import lightgbm as lgb
import chromadb
import groq
import streamlit
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

print("mediapipe:", mp.__version__)
print("cv2:      ", cv2.__version__)
print("numpy:    ", np.__version__)
print("lightgbm: ", lgb.__version__)
print("chromadb: ", chromadb.__version__)

# Smoke-test MediaPipe Pose (this is where protobuf usually explodes)
pose = mp.solutions.pose.Pose(static_image_mode=True)
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
result = pose.process(dummy)
print("MediaPipe Pose loaded and ran on dummy frame: OK")
print("ALL IMPORTS OK")
```

Run:

```bash
python tests/test_env.py
```

### ✅ Verification block — Phase 1.3

**Pass:** Final line prints `ALL IMPORTS OK`.
**Fail with `TypeError: Descriptors cannot not be created directly`:**
→ Wrong protobuf version, `pip install --force-reinstall protobuf==4.25.3`
**Fail with `ImportError: numpy.core.multiarray failed to import`:**
→ NumPy 2.x bled in. `pip install --force-reinstall "numpy==1.26.4"`
**Fail with `DLL load failed while importing _framework_bindings` (Windows):**
→ Missing MSVC runtime. Install [Visual C++ Redistributable 2015–2022](https://aka.ms/vs/17/release/vc_redist.x64.exe).

## 1.4 Scaffold the new directory tree

```bash
# From project root
mkdir -p src/cv src/data src/state src/rag src/workers src/utils
mkdir -p data/raw_landmarks data/relabeled data/augmented
mkdir -p models notebooks tests
# Init __init__.py everywhere so Python treats them as packages
touch src/__init__.py src/cv/__init__.py src/data/__init__.py
touch src/state/__init__.py src/rag/__init__.py src/workers/__init__.py src/utils/__init__.py
```

**Windows PowerShell equivalent:**

```powershell
New-Item -ItemType Directory -Force -Path src/cv,src/data,src/state,src/rag,src/workers,src/utils,data/raw_landmarks,data/relabeled,data/augmented,models,notebooks,tests
"src/__init__.py","src/cv/__init__.py","src/data/__init__.py","src/state/__init__.py","src/rag/__init__.py","src/workers/__init__.py","src/utils/__init__.py" | ForEach-Object { New-Item -ItemType File -Force -Path $_ }
```

## 1.5 Move existing CSV data into the new location

Don't move it, **symlink** it so old scripts still work:

**Windows:**
```powershell
# Run as admin
New-Item -ItemType SymbolicLink -Path "data/raw_landmarks/CV" -Target "$(Resolve-Path CV)"
```

**Linux:**
```bash
ln -s "$(pwd)/CV" data/raw_landmarks/CV
```

(If symlinks are painful on Windows, just copy: `Copy-Item -Recurse CV data/raw_landmarks/CV`.)

## 1.6 Rename "Medusa Systems" → "G2B"

```bash
# Linux/Mac:
grep -rli "Medusa Systems" . --include="*.py" --include="*.md" --include="*.txt" \
  | xargs sed -i 's/Medusa Systems/G2B/g'

# Bonus: lowercase variant
grep -rli "medusa systems" . --include="*.py" --include="*.md" --include="*.txt" \
  | xargs sed -i 's/medusa systems/G2B/g'
```

**Windows PowerShell:**

```powershell
Get-ChildItem -Recurse -Include *.py,*.md,*.txt -File | ForEach-Object {
    (Get-Content $_.FullName) -replace 'Medusa Systems','G2B' -replace 'medusa systems','G2B' |
    Set-Content $_.FullName
}
```

Then visually inspect any remaining hits:

```bash
grep -ri "medusa" . --include="*.py" --include="*.md"
```

### ✅ Verification block — Phase 1.6

**Pass:** `grep -ri "medusa"` returns nothing, or only intentional historical references in the redesign doc.

## 1.7 Git checkpoint

```bash
git checkout -b feature/redesign-v3
git add -A
git commit -m "Phase 1: env pinned, dir scaffolded, Medusa→G2B rename"
```

---

# PHASE 2 — CV Feature Redesign `[FAST PATH]`

**Objective:** Replace the 3-feature pipeline with the 14-feature pipeline. Verify the new features visibly separate the four classes on your existing landmark data.
**Estimated time:** 3–4 hours.

## 2.1 Verify your CSV column format first (5 minutes — critical)

Open one CSV from `CV/`:

```python
import pandas as pd
df = pd.read_csv("CV/correct_session_01.csv")  # or whatever you have
print(df.columns.tolist()[:10])
print(df.head(1))
print("label col?", "label" in df.columns)
```

You're looking for one of these patterns:

| Pattern A (assumed) | Pattern B | Pattern C |
|---|---|---|
| `landmark_0_x, landmark_0_y, landmark_0_z, landmark_0_v, landmark_1_x, ...` | `nose_x, nose_y, ls_x, ls_y, ...` (only some landmarks) | `lm0x, lm0y, lm0z, ...` (compact) |

If you're on **Pattern A**, the code below works as-is.
If **Pattern B or C**, add a one-time `rename` to map them to Pattern A. I'll show this in 2.2.

**Also check:** is the `label` stored as a column in the CSV, or is it encoded in the filename (e.g. `correct_session_01.csv` → label = "correct")?

## 2.2 Write `src/cv/pose_extractor.py`

This module:
- Wraps MediaPipe Pose with our preferred config.
- Pulls the 9 landmarks we care about, with all 4 coords each.
- Returns a numpy array of shape `(9, 4)` and an `is_reliable` bool.

**File:** `src/cv/pose_extractor.py`

```python
"""MediaPipe Pose wrapper that returns only the landmarks we use."""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import mediapipe as mp

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


@dataclass
class LandmarkArray:
    data: np.ndarray   # shape (9, 4) — x, y, z, visibility for each named landmark
    is_reliable: bool


class PoseExtractor:
    def __init__(self, model_complexity: int = 1, static_image_mode: bool = False):
        # model_complexity 0 = Lite (fastest, use on Pi 5)
        # model_complexity 1 = Full (use on dev machine)
        self.pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            static_image_mode=static_image_mode,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=False,
        )

    def extract(self, bgr_frame: np.ndarray) -> Optional[LandmarkArray]:
        rgb = bgr_frame[..., ::-1]
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return None
        all_lm = result.pose_landmarks.landmark
        rows = []
        for name in ORDERED_NAMES:
            lm = all_lm[LANDMARK_INDICES[name]]
            rows.append([lm.x, lm.y, lm.z, lm.visibility])
        data = np.array(rows, dtype=np.float32)

        # Reliability check on key landmarks
        idxs = [ORDERED_NAMES.index(n) for n in KEY_FOR_RELIABILITY]
        reliable = bool(np.all(data[idxs, 3] >= VISIBILITY_MIN))
        return LandmarkArray(data=data, is_reliable=reliable)

    def close(self):
        self.pose.close()
```

## 2.3 Write `src/cv/normalizer.py`

**File:** `src/cv/normalizer.py`

```python
"""Normalize a (9, 4) landmark array.

Origin: hip midpoint.
Scale:  shoulder width in the xy plane.
No rotation (we need shoulder tilt as a signal).
"""
from typing import Optional
import numpy as np
from .pose_extractor import ORDERED_NAMES

# Pre-resolve indices
NOSE = ORDERED_NAMES.index("nose")
LE = ORDERED_NAMES.index("left_ear")
RE = ORDERED_NAMES.index("right_ear")
LS = ORDERED_NAMES.index("left_shoulder")
RS = ORDERED_NAMES.index("right_shoulder")
LEL = ORDERED_NAMES.index("left_elbow")
REL = ORDERED_NAMES.index("right_elbow")
LH = ORDERED_NAMES.index("left_hip")
RH = ORDERED_NAMES.index("right_hip")

MIN_SHOULDER_WIDTH = 0.02   # in MediaPipe's normalized coords


def normalize(landmarks: np.ndarray) -> Optional[np.ndarray]:
    """landmarks shape (9, 4); returns shape (9, 4) or None if invalid."""
    if landmarks is None or landmarks.shape != (9, 4):
        return None
    hip_mid = 0.5 * (landmarks[LH, :3] + landmarks[RH, :3])
    centered_xyz = landmarks[:, :3] - hip_mid

    # Shoulder width in xy only (z is depth, different scale)
    sw = np.linalg.norm(landmarks[LS, :2] - landmarks[RS, :2])
    if sw < MIN_SHOULDER_WIDTH:
        return None
    scaled_xyz = centered_xyz / sw

    out = landmarks.copy()
    out[:, :3] = scaled_xyz
    return out
```

## 2.4 Write `src/cv/features.py` — the 14 features

**File:** `src/cv/features.py`

```python
"""14 posture features. Pure functions, fully deterministic."""
from typing import Dict, Optional
import numpy as np
from .normalizer import (NOSE, LE, RE, LS, RS, LEL, REL, LH, RH)

FEATURE_ORDER = [
    "ear_shoulder_offset_x",
    "craniovertebral_angle",
    "head_forward_offset_z",
    "nose_shoulder_offset_x",
    "shoulder_roll_z",
    "torso_compression_ratio",
    "elbow_forward_offset_z",
    "spine_angle_3d",
    "shoulder_tilt_angle",
    "hip_tilt_angle",
    "midline_deviation_angle",
    "nose_centerline_offset_x",
    "lateral_asymmetry_index",
    "landmark_confidence_mean",
]


def _angle_with_vertical_deg(vec: np.ndarray) -> float:
    """Angle between `vec` and the +y axis, in degrees, in [0, 180]."""
    vertical = np.array([0.0, 1.0, 0.0])
    v = vec / (np.linalg.norm(vec) + 1e-9)
    cosang = float(np.clip(np.dot(v[:3], vertical), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _angle_xy_deg(p_from: np.ndarray, p_to: np.ndarray) -> float:
    """Signed angle in xy plane (degrees) of vector p_from→p_to vs horizontal x-axis."""
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def extract_features(n: np.ndarray) -> Optional[Dict[str, float]]:
    """Compute 14 features. `n` must be the normalized (9, 4) array."""
    if n is None or n.shape != (9, 4):
        return None

    nose, le, re_, ls, rs, lel, rel, lh, rh = (n[NOSE], n[LE], n[RE], n[LS],
                                               n[RS], n[LEL], n[REL], n[LH], n[RH])
    ear_mid = 0.5 * (le[:3] + re_[:3])
    sh_mid = 0.5 * (ls[:3] + rs[:3])
    hip_mid = 0.5 * (lh[:3] + rh[:3])  # this is now ≈ (0, 0, 0) after normalization
    elbow_mid = 0.5 * (lel[:3] + rel[:3])

    # 1. ear_shoulder_offset_x (positive = ears in front of shoulders → forward head)
    ear_shoulder_offset_x = float(ear_mid[2] - sh_mid[2])
    # NOTE: we use z (depth) for "forward" because MediaPipe x is left-right.
    # If your camera is side-on, you'd flip this — but for a frontal webcam,
    # forward head shows mostly as z-shift. We also include x-offset as feature 4.

    # 2. craniovertebral_angle (in sagittal-ish plane: yz)
    #    Vector from sh_mid to ear_mid in yz; angle from vertical (+y)
    yz_vec = np.array([0.0, ear_mid[1] - sh_mid[1], ear_mid[2] - sh_mid[2]])
    craniovertebral_angle = _angle_with_vertical_deg(yz_vec)

    # 3. head_forward_offset_z (raw z delta)
    head_forward_offset_z = float(ear_mid[2] - sh_mid[2])

    # 4. nose_shoulder_offset_x (lateral head shift)
    nose_shoulder_offset_x = float(nose[0] - sh_mid[0])

    # 5. shoulder_roll_z (positive = shoulders in front of hips → roll forward)
    shoulder_roll_z = float(sh_mid[2] - hip_mid[2])

    # 6. torso_compression_ratio: |sh_y - hip_y| (because we normalized by sw)
    torso_compression_ratio = float(abs(sh_mid[1] - hip_mid[1]))

    # 7. elbow_forward_offset_z
    elbow_forward_offset_z = float(elbow_mid[2] - sh_mid[2])

    # 8. spine_angle_3d: angle of shoulder_mid relative to hip_mid (origin) in 3D vs +y
    spine_angle_3d = _angle_with_vertical_deg(sh_mid)

    # 9. shoulder_tilt_angle: tilt of left→right shoulder line in xy plane
    shoulder_tilt_angle = _angle_xy_deg(ls[:3], rs[:3])
    # Normalize: 0° = horizontal, sign = direction of tilt
    # We want abs and bounded; subtract 0 since arctan2 already gives signed.

    # 10. hip_tilt_angle
    hip_tilt_angle = _angle_xy_deg(lh[:3], rh[:3])

    # 11. midline_deviation_angle: angle of sh_mid→hip_mid in xy vs vertical
    xy_vec = np.array([sh_mid[0] - hip_mid[0], sh_mid[1] - hip_mid[1], 0.0])
    midline_deviation_angle = _angle_with_vertical_deg(xy_vec)

    # 12. nose_centerline_offset_x: nose vs shoulder midline x
    nose_centerline_offset_x = float(nose[0] - sh_mid[0])

    # 13. lateral_asymmetry_index: |left ear-to-shoulder| - |right ear-to-shoulder|
    left_es = float(np.linalg.norm(le[:3] - ls[:3]))
    right_es = float(np.linalg.norm(re_[:3] - rs[:3]))
    lateral_asymmetry_index = left_es - right_es

    # 14. landmark_confidence_mean
    landmark_confidence_mean = float(n[:, 3].mean())

    out = {
        "ear_shoulder_offset_x": ear_shoulder_offset_x,
        "craniovertebral_angle": craniovertebral_angle,
        "head_forward_offset_z": head_forward_offset_z,
        "nose_shoulder_offset_x": nose_shoulder_offset_x,
        "shoulder_roll_z": shoulder_roll_z,
        "torso_compression_ratio": torso_compression_ratio,
        "elbow_forward_offset_z": elbow_forward_offset_z,
        "spine_angle_3d": spine_angle_3d,
        "shoulder_tilt_angle": shoulder_tilt_angle,
        "hip_tilt_angle": hip_tilt_angle,
        "midline_deviation_angle": midline_deviation_angle,
        "nose_centerline_offset_x": nose_centerline_offset_x,
        "lateral_asymmetry_index": lateral_asymmetry_index,
        "landmark_confidence_mean": landmark_confidence_mean,
    }
    # Sanity: no NaNs
    if any(np.isnan(v) or np.isinf(v) for v in out.values()):
        return None
    return out


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert dict to ordered numpy vector for the classifier."""
    return np.array([features[k] for k in FEATURE_ORDER], dtype=np.float32)
```

## 2.5 Write a CSV→features converter

This is a separate utility because we'll use it twice (Phase 3 relabel, Phase 4 training).

**File:** `src/data/csv_to_features.py`

```python
"""Read your existing CV/*.csv files and produce a features DataFrame."""
import re
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

from src.cv.pose_extractor import LANDMARK_INDICES, ORDERED_NAMES
from src.cv.normalizer import normalize
from src.cv.features import extract_features, FEATURE_ORDER


def _row_to_landmark_array(row: pd.Series) -> Optional[np.ndarray]:
    """Pull our 9 landmarks out of a wide CSV row. Returns (9, 4) or None."""
    out = np.zeros((9, 4), dtype=np.float32)
    for i, name in enumerate(ORDERED_NAMES):
        idx = LANDMARK_INDICES[name]
        # === EDIT IF YOUR CSV COLUMNS ARE NAMED DIFFERENTLY ===
        cols = [f"landmark_{idx}_x", f"landmark_{idx}_y",
                f"landmark_{idx}_z", f"landmark_{idx}_v"]
        # =====================================================
        try:
            out[i] = [row[c] for c in cols]
        except KeyError:
            return None
    return out


def _infer_label(csv_path: Path, df: pd.DataFrame) -> str:
    """Prefer label column; fall back to filename prefix."""
    if "label" in df.columns and df["label"].notna().any():
        return str(df["label"].iloc[0]).strip().lower()
    name = csv_path.stem.lower()
    if name.startswith("correct"):  return "correct_posture"
    if name.startswith("slouch"):   return "slouching"
    if name.startswith("neck"):     return "neck_forward"
    if name.startswith("lean"):     return "lean"
    raise ValueError(f"Cannot infer label for {csv_path}")


def csv_to_features_df(csv_paths: List[Path]) -> pd.DataFrame:
    rows = []
    for path in csv_paths:
        df = pd.read_csv(path)
        label = _infer_label(path, df)
        for _, r in df.iterrows():
            arr = _row_to_landmark_array(r)
            if arr is None:
                continue
            normed = normalize(arr)
            if normed is None:
                continue
            feats = extract_features(normed)
            if feats is None:
                continue
            rows.append({**feats, "label": label,
                         "source_file": path.name})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    csvs = sorted(Path("data/raw_landmarks/CV").glob("*.csv"))
    print(f"Found {len(csvs)} CSV files.")
    df = csv_to_features_df(csvs)
    print(f"Extracted {len(df)} rows.")
    print(df["label"].value_counts())
    df.to_csv("data/relabeled/features_raw.csv", index=False)
    print("Saved → data/relabeled/features_raw.csv")
```

Run it:

```bash
python -m src.data.csv_to_features
```

### ✅ Verification block — Phase 2.5

**Pass:**
- Prints e.g. `Found 12 CSV files.` (matches your `CV/` count)
- Prints `Extracted N rows.` where N is roughly `total CSV rows × 0.85` (15% dropped is OK)
- Class distribution roughly matches your collection (it can be uneven; that's fine)
- File `data/relabeled/features_raw.csv` exists

**Fail with `KeyError: 'landmark_0_x'`:**
→ Your CSV column names differ. Open `csv_to_features.py`, edit the marked block to match. E.g. if columns are `nose_x, nose_y, nose_z` instead:

```python
NAME_TO_COLS = {
    "nose": ["nose_x", "nose_y", "nose_z", "nose_v"],
    "left_ear": ["lear_x", "lear_y", "lear_z", "lear_v"],
    # ... fill in for all 9
}
cols = NAME_TO_COLS[name]
```

**Fail with low extraction rate (e.g. 100 rows in, 5 rows out):**
→ Most landmarks have visibility < 0.5. Likely the data was collected far from the camera or with bad lighting. Lower `VISIBILITY_MIN` to 0.3 in `pose_extractor.py` *only for this step*; revert before live inference.

## 2.6 Visually verify feature separation (the moment of truth)

**File:** `notebooks/01_inspect_features.ipynb`

Or just a script: **File:** `notebooks/01_inspect_features.py`

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.cv.features import FEATURE_ORDER

df = pd.read_csv("data/relabeled/features_raw.csv")
print(df.groupby("label")[FEATURE_ORDER].agg(["mean", "std"]).round(3).T)

# Pairplot on the 4 most diagnostic features
key = ["ear_shoulder_offset_x", "shoulder_roll_z",
       "shoulder_tilt_angle", "torso_compression_ratio"]
sns.pairplot(df[key + ["label"]], hue="label", diag_kind="kde",
             plot_kws={"alpha": 0.5, "s": 15})
plt.savefig("notebooks/feature_pairplot.png", dpi=120, bbox_inches="tight")
print("Saved notebooks/feature_pairplot.png")
```

```bash
python notebooks/01_inspect_features.py
```

### ✅ Verification block — Phase 2.6 (the gate)

Open `notebooks/feature_pairplot.png` and look:

| You should see | If you don't |
|---|---|
| `lean` class clearly separates on `shoulder_tilt_angle` (different mean from others) | Verify your `lean` CSV samples actually have tilted shoulders. Look at a few frames. |
| `neck_forward` separates on `ear_shoulder_offset_x` (larger values) | Ears might be poorly tracked. Re-check visibility scores for the ear landmarks. |
| `slouching` separates from `correct` on at least `shoulder_roll_z` OR `torso_compression_ratio` | If both look identical between slouch and correct → **your slouch samples are still too mild.** Skip ahead to Phase 3.3 (synthesis) early. |

**Critical mean values to expect (rough):**

| Feature | correct_posture | slouching | neck_forward | lean |
|---|---|---|---|---|
| `ear_shoulder_offset_x` | ~0.0 ± 0.1 | 0.1–0.3 | **0.3–0.6** | ~0.0 |
| `shoulder_roll_z` | ~0.0 | **0.1–0.3** | small + | ~0.0 |
| `shoulder_tilt_angle` (abs) | <3° | <3° | <3° | **>5°** |
| `torso_compression_ratio` | ~1.4–1.8 | **<1.4** | ~1.5 | ~1.5 |

If your numbers are wildly off (e.g. all features near zero for all classes), normalization is broken. Re-read Phase 2.3, ensure `shoulder_width > MIN_SHOULDER_WIDTH` is firing correctly.

## 2.7 Unit test the features

**File:** `tests/test_features.py`

```python
import numpy as np
from src.cv.pose_extractor import ORDERED_NAMES
from src.cv.normalizer import normalize
from src.cv.features import extract_features, FEATURE_ORDER


def _make_correct_landmarks():
    """Synthetic 'correct posture' landmark set."""
    # x, y, z, visibility
    n = np.array([
        [0.5, 0.20, 0.0, 1.0],   # nose
        [0.55, 0.22, 0.0, 1.0],  # left_ear
        [0.45, 0.22, 0.0, 1.0],  # right_ear
        [0.60, 0.35, 0.0, 1.0],  # left_shoulder
        [0.40, 0.35, 0.0, 1.0],  # right_shoulder
        [0.65, 0.50, 0.0, 1.0],  # left_elbow
        [0.35, 0.50, 0.0, 1.0],  # right_elbow
        [0.55, 0.65, 0.0, 1.0],  # left_hip
        [0.45, 0.65, 0.0, 1.0],  # right_hip
    ], dtype=np.float32)
    return n


def test_normalize_centers_hips():
    n = _make_correct_landmarks()
    out = normalize(n)
    hip_mid = 0.5 * (out[7, :3] + out[8, :3])
    assert np.allclose(hip_mid, 0, atol=1e-5)


def test_features_for_correct_posture():
    n = _make_correct_landmarks()
    norm = normalize(n)
    f = extract_features(norm)
    assert f is not None
    # For our synthetic-correct sample:
    assert abs(f["shoulder_tilt_angle"]) < 5, f
    assert abs(f["ear_shoulder_offset_x"]) < 0.1, f
    assert abs(f["shoulder_roll_z"]) < 0.05, f


def test_features_for_forward_head():
    n = _make_correct_landmarks()
    # Push ears forward in z
    n[1, 2] = 0.10
    n[2, 2] = 0.10
    norm = normalize(n)
    f = extract_features(norm)
    assert f["ear_shoulder_offset_x"] > 0.15, f["ear_shoulder_offset_x"]
    assert f["craniovertebral_angle"] > 10, f["craniovertebral_angle"]


def test_features_for_lean():
    n = _make_correct_landmarks()
    # Tilt shoulders
    n[3, 1] = 0.32  # left shoulder up
    n[4, 1] = 0.38  # right shoulder down
    norm = normalize(n)
    f = extract_features(norm)
    assert abs(f["shoulder_tilt_angle"]) > 5, f["shoulder_tilt_angle"]


def test_feature_order_matches():
    f = extract_features(normalize(_make_correct_landmarks()))
    assert list(f.keys()) == FEATURE_ORDER
```

Run:

```bash
pytest tests/test_features.py -v
```

### ✅ Verification block — Phase 2.7

**Pass:** all 5 tests pass.
**Fail:** the assertions tell you exactly which feature is wrong. Fix the computation in `features.py` until tests pass.

## 2.8 Git checkpoint

```bash
git add -A
git commit -m "Phase 2: 14-feature pipeline with normalization, tests passing"
```

---

# PHASE 3 — Dataset Relabeling and Augmentation `[FAST PATH]`

**Objective:** Produce a clean, balanced training set of ~3000–5000 rows from your existing landmark data plus synthetic samples.
**Estimated time:** 3–4 hours.

## 3.1 Heuristic relabeling

**File:** `src/data/relabel.py`

```python
"""Apply rule-based labels to features. Drops ambiguous samples."""
from typing import Dict, Optional
import pandas as pd

# Thresholds — tune these on a 30-sample manual review set if needed
TH_SHOULDER_TILT = 6.0       # degrees
TH_MIDLINE_DEV   = 6.0       # degrees
TH_EAR_OFFSET    = 0.35      # normalized (shoulder-width units)
TH_CV_ANGLE      = 18.0      # craniovertebral angle from vertical (= small angle = upright)
TH_SHOULDER_ROLL = 0.15
TH_TORSO_COMPR   = 1.45

# A second, looser set used only for distinguishing "ambiguous" from "drop"
SOFT_EAR_OFFSET  = 0.20
SOFT_SHOULDER_ROLL = 0.10


def heuristic_label(f: Dict[str, float]) -> Optional[str]:
    """Returns one of the 4 class names, or None if too ambiguous to label."""
    lean_signal = (abs(f["shoulder_tilt_angle"]) > TH_SHOULDER_TILT or
                   f["midline_deviation_angle"] > TH_MIDLINE_DEV)
    head_signal = (f["ear_shoulder_offset_x"] > TH_EAR_OFFSET or
                   f["craniovertebral_angle"] > TH_CV_ANGLE)
    slouch_signal = (f["shoulder_roll_z"] > TH_SHOULDER_ROLL or
                     f["torso_compression_ratio"] < TH_TORSO_COMPR)

    # Lean takes priority — it's geometrically distinct
    if lean_signal and not (head_signal or slouch_signal):
        return "lean"
    if lean_signal and (head_signal or slouch_signal):
        # Combined posture issues; lean is the dominant visual cue
        return "lean"

    if head_signal and slouch_signal:
        return "slouching"   # head forward + body collapsed → slouching
    if head_signal and not slouch_signal:
        return "neck_forward"
    if slouch_signal and not head_signal:
        return "slouching"

    # No strong signal anywhere
    if (abs(f["shoulder_tilt_angle"]) < 3 and
        f["ear_shoulder_offset_x"] < SOFT_EAR_OFFSET and
        f["shoulder_roll_z"] < SOFT_SHOULDER_ROLL):
        return "correct_posture"

    # In between — ambiguous
    return None


def relabel_dataframe(df: pd.DataFrame, drop_disagreement: bool = False
                      ) -> pd.DataFrame:
    """Adds 'heuristic_label' and 'kept' columns.

    drop_disagreement=False keeps everything, just annotates.
    drop_disagreement=True keeps only rows where the heuristic agrees with the
    original label OR the original label is missing.
    """
    feat_cols = [c for c in df.columns
                 if c not in ("label", "source_file", "heuristic_label", "kept")]
    new_labels = []
    for _, row in df.iterrows():
        f = {c: row[c] for c in feat_cols}
        new_labels.append(heuristic_label(f))
    df = df.copy()
    df["heuristic_label"] = new_labels
    # Drop ambiguous always
    df["kept"] = df["heuristic_label"].notna()
    if drop_disagreement and "label" in df.columns:
        df["kept"] &= (df["heuristic_label"] == df["label"])
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/relabeled/features_raw.csv")
    out = relabel_dataframe(df, drop_disagreement=False)

    print("\n=== Original labels ===")
    print(df["label"].value_counts())
    print("\n=== Heuristic labels ===")
    print(out["heuristic_label"].value_counts(dropna=False))
    print("\n=== Agreement matrix ===")
    print(pd.crosstab(out["label"], out["heuristic_label"], dropna=False))

    kept = out[out["kept"]].copy()
    kept["label"] = kept["heuristic_label"]
    kept = kept.drop(columns=["heuristic_label", "kept"])
    kept.to_csv("data/relabeled/features_relabeled.csv", index=False)
    print(f"\nKept {len(kept)} / {len(out)} rows")
    print(f"Saved → data/relabeled/features_relabeled.csv")
```

Run:

```bash
python -m src.data.relabel
```

### ✅ Verification block — Phase 3.1

**Inspect the agreement matrix carefully.** Expected patterns:

- **High diagonal** (heuristic agrees with original): good.
- **`slouching` original ↔ `correct_posture` heuristic** disagreement is *expected* if your collection had mild slouching — that's exactly the problem you described. The heuristic is correcting them. **This is the goal.**
- **High `None` (ambiguous) count for any class:** means your data for that class is mostly borderline. You'll need synthesis to compensate (Phase 3.3).

**Decision point:**
- If `kept` count > 1500 and class distribution is reasonable → continue.
- If `kept` count < 800 → relax thresholds OR skip directly to Phase 3.3 for synthetic data.

## 3.2 Augmentation (small perturbations)

**File:** `src/data/augment.py`

```python
"""Geometric perturbation of normalized landmark arrays.

The trick: we operate on landmarks (not features), regenerate features,
and label-preserve. Use after relabeling and before training.
"""
from typing import List
import numpy as np
import pandas as pd

from src.cv.pose_extractor import ORDERED_NAMES
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
```

(This file is *imported* but doesn't have a runnable `__main__`; it's used by `build_dataset.py` below.)

## 3.3 Synthetic class exaggeration (this is the killer feature)

**File:** `src/data/synthesize.py`

```python
"""Generate exaggerated landmark samples for under-represented classes.

Takes correct-posture landmark arrays and mathematically deforms them into
class-typical exaggerated versions. This is how you avoid recollecting data.
"""
from typing import Iterator
import numpy as np

from src.cv.pose_extractor import ORDERED_NAMES

NOSE = ORDERED_NAMES.index("nose")
LE = ORDERED_NAMES.index("left_ear")
RE = ORDERED_NAMES.index("right_ear")
LS = ORDERED_NAMES.index("left_shoulder")
RS = ORDERED_NAMES.index("right_shoulder")
LEL = ORDERED_NAMES.index("left_elbow")
REL = ORDERED_NAMES.index("right_elbow")


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
```

## 3.4 The orchestrator: `build_dataset.py`

This is the single command that produces your training set from your raw CSVs.

**File:** `src/data/build_dataset.py`

```python
"""End-to-end: raw CSVs → relabeled → augmented → synthesized → final training CSV."""
from pathlib import Path
import numpy as np
import pandas as pd

from src.cv.pose_extractor import LANDMARK_INDICES, ORDERED_NAMES
from src.cv.normalizer import normalize
from src.cv.features import extract_features, FEATURE_ORDER
from src.data.csv_to_features import _row_to_landmark_array, _infer_label
from src.data.relabel import heuristic_label, relabel_dataframe
from src.data.augment import perturb_landmarks
from src.data.synthesize import synthesize_from_correct

RNG = np.random.default_rng(42)
TARGET_PER_CLASS = 1200          # tune this
AUGS_PER_REAL_SAMPLE = 4
SYNTH_TARGET_FRACTION = 0.5      # half of each non-correct class can be synthesized


def collect_landmark_arrays_per_class(csv_dir: Path):
    """Returns dict[label] -> list[(9, 4) array]."""
    pool = {"correct_posture": [], "slouching": [],
            "neck_forward": [], "lean": []}
    for path in sorted(csv_dir.glob("*.csv")):
        df = pd.read_csv(path)
        label = _infer_label(path, df)
        for _, r in df.iterrows():
            arr = _row_to_landmark_array(r)
            if arr is None:
                continue
            normed = normalize(arr)
            if normed is None:
                continue
            feats = extract_features(normed)
            if feats is None:
                continue
            # Apply heuristic to filter to high-confidence samples
            h = heuristic_label(feats)
            if h == label:  # only keep samples where heuristic agrees
                pool[label].append(arr)
    return pool


def build():
    pool = collect_landmark_arrays_per_class(Path("data/raw_landmarks/CV"))
    print("Confident landmark counts:")
    for k, v in pool.items():
        print(f"  {k}: {len(v)}")

    rows = []

    for label, arrays in pool.items():
        # === Real samples (already filtered to confident) ===
        for arr in arrays:
            normed = normalize(arr)
            if normed is None: continue
            f = extract_features(normed)
            if f is None: continue
            f["label"] = label
            f["source"] = "real"
            rows.append(f)

        # === Augmentations ===
        for arr in arrays:
            for _ in range(AUGS_PER_REAL_SAMPLE):
                pert = perturb_landmarks(arr, RNG)
                normed = normalize(pert)
                if normed is None: continue
                f = extract_features(normed)
                if f is None: continue
                f["label"] = label
                f["source"] = "augmented"
                rows.append(f)

    # === Synthetic samples for non-correct classes ===
    correct_pool = np.array(pool["correct_posture"], dtype=np.float32)
    if len(correct_pool) == 0:
        raise RuntimeError("No confident correct_posture samples — fix Phase 2 first")

    for target_class in ["slouching", "neck_forward", "lean"]:
        # Count how many we already have for this class
        have = sum(1 for r in rows if r["label"] == target_class)
        want = TARGET_PER_CLASS - have
        if want > 0:
            print(f"Synthesizing {want} extra '{target_class}' samples")
            for arr in synthesize_from_correct(correct_pool, target_class, want, RNG):
                normed = normalize(arr)
                if normed is None: continue
                f = extract_features(normed)
                if f is None: continue
                f["label"] = target_class
                f["source"] = "synthetic"
                rows.append(f)

    out = pd.DataFrame(rows)
    print("\nFinal class distribution:")
    print(out["label"].value_counts())
    print("\nBy source:")
    print(out.groupby("label")["source"].value_counts())
    out.to_csv("data/augmented/training_set.csv", index=False)
    print(f"\nSaved → data/augmented/training_set.csv ({len(out)} rows)")


if __name__ == "__main__":
    build()
```

Run:

```bash
python -m src.data.build_dataset
```

### ✅ Verification block — Phase 3.4 (the gate)

**Pass conditions:**
- Final class counts all ≥ 800 and within 30% of each other.
- `data/augmented/training_set.csv` exists with all 14 features + label + source.
- Distribution by source shows you have a mix: e.g. for `slouching`, maybe 200 real + 800 augmented + 200 synthetic. **Do NOT have a class that is 100% synthetic** — if any class has zero "real" samples, you have a data-collection gap that synthesis cannot fully compensate for.

**Fail with imbalanced classes:**
- Adjust `TARGET_PER_CLASS` up to 1500.
- If still imbalanced, relax heuristic thresholds in `relabel.py`.

**Sanity plot:**

```python
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
df = pd.read_csv("data/augmented/training_set.csv")
key = ["ear_shoulder_offset_x", "shoulder_roll_z",
       "shoulder_tilt_angle", "torso_compression_ratio"]
sns.pairplot(df[key + ["label"]], hue="label", diag_kind="kde",
             plot_kws={"alpha": 0.4, "s": 10})
plt.savefig("notebooks/training_pairplot.png", dpi=120)
```

The four classes should form **visibly distinct clusters**. If they don't, do not proceed to Phase 4 — your training will inherit the noise.

## 3.5 Git checkpoint

```bash
git add -A
git commit -m "Phase 3: relabel + augment + synthesize → clean training set"
```

---

# PHASE 4 — Model Retraining `[FAST PATH]`

**Objective:** Train LightGBM on the new dataset, integrate rule overlay, save the new model.
**Estimated time:** 1.5–2 hours (training is fast; the work is in evaluation).

## 4.1 Write the training script

**File:** `train_lgbm.py` (top-level, not under `src/`)

```python
"""Trains the new LightGBM posture classifier. Saves to models/."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import joblib

from src.cv.features import FEATURE_ORDER

CLASSES = ["correct_posture", "slouching", "neck_forward", "lean"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 4,
    "metric": "multi_logloss",
    "num_leaves": 15,
    "max_depth": 5,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 3,
    "min_data_in_leaf": 10,
    "lambda_l2": 0.1,
    "verbose": -1,
    "n_jobs": 2,
    "seed": 42,
}


def load_data():
    df = pd.read_csv("data/augmented/training_set.csv")
    X = df[FEATURE_ORDER].values.astype(np.float32)
    y = np.array([LABEL_TO_IDX[c] for c in df["label"]])
    return X, y, df


def cv_eval(X, y):
    """5-fold stratified CV."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        train_set = lgb.Dataset(X[tr], y[tr])
        val_set = lgb.Dataset(X[va], y[va], reference=train_set)
        model = lgb.train(
            LGBM_PARAMS, train_set,
            num_boost_round=500,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        pred = np.argmax(model.predict(X[va]), axis=1)
        acc = accuracy_score(y[va], pred)
        fold_accs.append(acc)
        print(f"Fold {fold}: accuracy = {acc:.4f}")
    print(f"\nMean CV accuracy: {np.mean(fold_accs):.4f}  ± {np.std(fold_accs):.4f}")
    return float(np.mean(fold_accs))


def train_final(X, y):
    """Train on 90% / hold out 10% for one final report."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42)
    train_set = lgb.Dataset(X_tr, y_tr)
    val_set = lgb.Dataset(X_te, y_te, reference=train_set)
    model = lgb.train(
        LGBM_PARAMS, train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    pred = np.argmax(model.predict(X_te), axis=1)
    print("\n=== Held-out report ===")
    print(classification_report(y_te, pred, target_names=CLASSES, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_te, pred)
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))

    Path("models").mkdir(exist_ok=True)
    model.save_model("models/posture_lgbm_v3.txt")
    with open("models/feature_order.json", "w") as f:
        json.dump(FEATURE_ORDER, f, indent=2)
    print("\nSaved models/posture_lgbm_v3.txt + feature_order.json")
    return model


if __name__ == "__main__":
    X, y, df = load_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(CLASSES)} classes")
    print(df["label"].value_counts().to_string())
    print()
    cv_acc = cv_eval(X, y)
    print()
    train_final(X, y)
```

Run:

```bash
python train_lgbm.py
```

### ✅ Verification block — Phase 4.1

**Pass conditions:**
- Mean CV accuracy ≥ 0.85
- Held-out per-class F1 ≥ 0.80 for every class
- Confusion matrix shows a clean diagonal

**Fail: CV accuracy < 0.80:**
1. Check feature distributions again (Phase 2.6).
2. Verify class balance in `training_set.csv` (Phase 3.4).
3. Try `num_leaves=31, max_depth=7, num_boost_round=1000` — sometimes the small config under-fits when you have 4k+ rows.

**Fail: high `slouching` ↔ `correct_posture` confusion specifically:**
- Your synthetic slouching parameters are too mild. In `synthesize.py`, increase the z-offsets to `severity * 0.30` (from 0.22). Rebuild dataset (Phase 3.4) and retrain.

**Fail: high `slouching` ↔ `neck_forward` confusion:**
- Tighten the heuristic in `relabel.py`: require `slouch_signal AND NOT head_signal` for `slouching`. Or define a 5th internal class (`combined`) and re-route during training.

**Sanity-check feature importances:**

```python
import lightgbm as lgb
m = lgb.Booster(model_file="models/posture_lgbm_v3.txt")
imp = sorted(zip(FEATURE_ORDER, m.feature_importance("gain")),
             key=lambda kv: -kv[1])
for k, v in imp: print(f"{k:35s} {v:8.0f}")
```

Top 5 should include `ear_shoulder_offset_x`, `shoulder_tilt_angle`,
`shoulder_roll_z`. If `landmark_confidence_mean` or
`nose_centerline_offset_x` are top — something's leaking.

## 4.2 Rule engine for confidence fusion

**File:** `src/cv/rule_engine.py`

```python
"""Soft-probability rule engine. Returns per-class probability based on
clinical thresholds. Used to fuse with LGBM output for robustness.
"""
from typing import Dict
import numpy as np

CLASSES = ["correct_posture", "slouching", "neck_forward", "lean"]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def rule_probs(f: Dict[str, float]) -> np.ndarray:
    """Returns [p_correct, p_slouching, p_neck_forward, p_lean]."""
    # Soft scores in [0, 1]
    lean_score = max(
        _sigmoid((abs(f["shoulder_tilt_angle"]) - 4.0) / 2.0),
        _sigmoid((f["midline_deviation_angle"] - 4.0) / 2.0),
    )
    neck_score = max(
        _sigmoid((f["ear_shoulder_offset_x"] - 0.25) / 0.10),
        _sigmoid((f["craniovertebral_angle"] - 12.0) / 4.0),
    )
    slouch_score = max(
        _sigmoid((f["shoulder_roll_z"] - 0.10) / 0.05),
        _sigmoid(((1.45 - f["torso_compression_ratio"])) / 0.10),
    )
    # "Correct" is the residual
    correct_score = 1.0 - max(lean_score, neck_score, slouch_score)
    correct_score = max(correct_score, 0.01)

    probs = np.array([correct_score, slouch_score, neck_score, lean_score],
                     dtype=np.float32)
    return probs / probs.sum()
```

## 4.3 Combined classifier

**File:** `src/cv/classifier.py`

```python
"""Loads the LGBM model and fuses it with the rule engine."""
import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import lightgbm as lgb

from src.cv.features import features_to_vector
from src.cv.rule_engine import rule_probs, CLASSES


class PostureClassifier:
    def __init__(self,
                 model_path: str = "models/posture_lgbm_v3.txt",
                 feature_order_path: str = "models/feature_order.json",
                 rule_weight: float = 0.30):
        self.model = lgb.Booster(model_file=model_path)
        with open(feature_order_path) as f:
            self.feature_order = json.load(f)
        self.rule_weight = rule_weight

    def predict(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        x = features_to_vector(features).reshape(1, -1)
        ml = self.model.predict(x)[0]
        rl = rule_probs(features)
        fused = (1.0 - self.rule_weight) * ml + self.rule_weight * rl
        idx = int(np.argmax(fused))
        return (CLASSES[idx],
                float(fused[idx]),
                dict(zip(CLASSES, fused.tolist())))
```

## 4.4 Live webcam smoke test (no UI yet)

**File:** `tests/test_live_classifier.py`

```python
"""Run for 30 seconds against your webcam, print classifications."""
import time
import cv2
import numpy as np

from src.cv.pose_extractor import PoseExtractor
from src.cv.normalizer import normalize
from src.cv.features import extract_features
from src.cv.classifier import PostureClassifier


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    pe = PoseExtractor(model_complexity=1)
    clf = PostureClassifier()

    start = time.time()
    n_frames = 0
    while time.time() - start < 30:
        ok, frame = cap.read()
        if not ok:
            break
        n_frames += 1
        la = pe.extract(frame)
        if la is None or not la.is_reliable:
            print("[unreliable]")
            continue
        normed = normalize(la.data)
        if normed is None:
            continue
        feats = extract_features(normed)
        if feats is None:
            continue
        label, conf, _ = clf.predict(feats)
        print(f"{label:18s}  conf={conf:.2f}  "
              f"ear_x={feats['ear_shoulder_offset_x']:+.2f}  "
              f"sh_z={feats['shoulder_roll_z']:+.2f}  "
              f"tilt={feats['shoulder_tilt_angle']:+.1f}")
    cap.release()
    pe.close()
    print(f"\n{n_frames} frames in 30s = {n_frames / 30:.1f} fps")


if __name__ == "__main__":
    main()
```

Run while sitting in front of the webcam:

```bash
python tests/test_live_classifier.py
```

### ✅ Verification block — Phase 4.4

Try each of the 4 postures live (sit correct, slouch hard, push head forward, lean left):

| Posture you adopt | Expected label (most frames) |
|---|---|
| Upright, head over shoulders | `correct_posture` |
| Round upper back, shoulders forward | `slouching` |
| Body upright but head jutted forward | `neck_forward` |
| Tilt to one side | `lean` |

**Pass:** ≥3 of 4 postures are correctly identified ≥70% of frames.

**Single-class flicker between two classes:** expected at this stage — no smoothing yet. Phase 5 fixes that.

**Wrong class for posture X:**
- Print the diagnostic features (already in the test) when adopting that posture.
- Cross-reference with the threshold table in Phase 2.6.
- Adjust either the LGBM model (retrain with more data for that class) OR the rule thresholds in `rule_engine.py`.

## 4.5 Git checkpoint

```bash
git add -A
git commit -m "Phase 4: LightGBM v3 trained, fusion classifier, live verified"
```

---

# PHASE 5 — Smoothing and State Tracking `[FAST PATH]`

**Objective:** Eliminate frame-by-frame flicker. Produce a `PostureState` object every frame for downstream RAG/UI use.
**Estimated time:** 2 hours.

## 5.1 Temporal smoother

**File:** `src/cv/smoother.py`

```python
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
```

**File:** `tests/test_smoother.py`

```python
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
```

```bash
pytest tests/test_smoother.py -v
```

### ✅ Verification block — Phase 5.1
All 4 tests pass.

## 5.2 PostureState dataclass

**File:** `src/state/posture_state.py`

```python
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
```

## 5.3 The session tracker

This keeps the rolling stats (`posture_duration_sec`, `posture_distribution`,
`correction_events`) updated as time passes.

**File:** `src/state/session_tracker.py`

```python
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
```

## 5.4 The complete pipeline

**File:** `src/cv/pipeline.py`

```python
"""Orchestrates: frame in → PostureState out."""
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
```

## 5.5 Verify end-to-end (CV only, no RAG yet)

**File:** `tests/test_pipeline.py`

```python
"""Live test of the full CV pipeline. Smoothing visible. Run for 60s."""
import time
import cv2

from src.cv.pipeline import PosturePipeline


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    pipe = PosturePipeline(model_complexity=1)
    last_label = None
    start = time.time()

    while time.time() - start < 60:
        ok, frame = cap.read()
        if not ok: break
        state = pipe.step(frame)
        if state is None:
            print("[no detection]")
            continue
        if state.posture_class != last_label:
            print(f"\n>>> CHANGED to {state.posture_class}  "
                  f"(conf={state.confidence:.2f}, primary={state.primary_issue})")
            last_label = state.posture_class
        else:
            print(".", end="", flush=True)
    cap.release()
    pipe.close()


if __name__ == "__main__":
    main()
```

```bash
python tests/test_pipeline.py
```

### ✅ Verification block — Phase 5.5

**Pass:**
- No rapid flicker. Posture changes only when you actually change posture for ~0.5s.
- `primary_issue` correctly identifies the dominant feature (e.g. `forward_head` when you push your head forward).
- `state.posture_duration_sec` increases monotonically while you hold a pose.
- `state.correction_events` increments when you return to correct posture.

**Fail: too sluggish:**
- Lower `hysteresis_frames` to 5 in `pipeline.py`.

**Fail: still flickery:**
- Raise `hysteresis_frames` to 12 or lower `alpha` to 0.20.

## 5.6 Git checkpoint

```bash
git add -A
git commit -m "Phase 5: smoother + PostureState + session tracker + pipeline"
```

---

# PHASE 6 — RAG Redesign `[FAST PATH]`

**Objective:** Tag every chunk with posture-class metadata, rebuild ChromaDB at `rag_db_v3/`, add hybrid retrieval (BM25 + dense) with RRF, posture-aware prompt template.
**Estimated time:** 3–4 hours.

> Skip the reranker step (6.5) if hosted-reranking budget is a concern. Document is structured so retrieval still works without it.

## 6.1 Metadata-tag your physiotherapy KB chunks

This is a one-time LLM-tagging job. Pull each chunk from your existing index, ask Groq to tag it, write the tagged corpus to a new JSONL file.

**File:** `src/rag/tagger.py`

```python
"""Tag KB chunks with metadata using Groq for one-time bulk tagging."""
import json
import os
import time
from pathlib import Path
from typing import List, Dict
from groq import Groq

CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.1-8b-instant"

TAGGING_PROMPT = """You are tagging physiotherapy textbook chunks for a posture
correction system. Output ONLY valid JSON.

The system classifies posture as one of:
- correct_posture: aligned head, shoulders, hips, upright torso
- slouching: rounded upper back, shoulders rolled forward, spine flexion
- neck_forward: head protrudes forward (turtle neck), torso mostly upright
- lean: lateral tilt, left/right asymmetry

For the chunk below, output JSON with these fields:

{{
  "applicable_postures": ["slouching", ...],   // list, at least 1 entry, can be ["correct_posture"] if it's about ideal posture
  "anatomical_region": "cervical_spine" | "thoracic_spine" | "lumbar_spine" | "shoulder" | "general",
  "content_type": "definition" | "cause" | "consequence" | "correction" | "exercise" | "background",
  "key_terms": ["..."]                          // 3-6 important terms from the chunk
}}

CHUNK:
\"\"\"
{chunk}
\"\"\"

Output only the JSON, no prose:"""


def tag_chunk(text: str, retries: int = 3) -> Dict:
    for attempt in range(retries):
        try:
            resp = CLIENT.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user",
                           "content": TAGGING_PROMPT.format(chunk=text[:2000])}],
                temperature=0.1,
                max_tokens=300,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip code fences if present
            if raw.startswith("```"):
                raw = raw.strip("`").lstrip("json").strip()
            data = json.loads(raw)
            assert "applicable_postures" in data
            return data
        except Exception as e:
            print(f"  tagging attempt {attempt+1} failed: {e}")
            time.sleep(2)
    # Fallback: generic tag
    return {
        "applicable_postures": ["correct_posture", "slouching",
                                "neck_forward", "lean"],
        "anatomical_region": "general",
        "content_type": "background",
        "key_terms": [],
    }


def tag_corpus(chunks_jsonl_in: str, out_jsonl: str):
    """Each input line should be {'id': str, 'text': str}."""
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(chunks_jsonl_in) as fin, open(out_jsonl, "w") as fout:
        for line in fin:
            obj = json.loads(line)
            tags = tag_chunk(obj["text"])
            obj["metadata"] = tags
            fout.write(json.dumps(obj) + "\n")
            n += 1
            if n % 20 == 0:
                print(f"  tagged {n} chunks")
    print(f"DONE — tagged {n} chunks → {out_jsonl}")


if __name__ == "__main__":
    # === Adjust these to your existing setup ===
    tag_corpus(chunks_jsonl_in="data/kb_chunks_raw.jsonl",
               out_jsonl="data/kb_chunks_tagged.jsonl")
```

**Important:** If your current `build_rag.py` reads from a PDF or text file directly, first run a quick exporter that dumps chunks to `data/kb_chunks_raw.jsonl` with `{id, text}` per line. Add this small adapter at the top of your existing `build_rag.py`:

```python
import json
def export_chunks_to_jsonl(chunks, out_path="data/kb_chunks_raw.jsonl"):
    with open(out_path, "w") as f:
        for i, c in enumerate(chunks):
            f.write(json.dumps({"id": f"chunk_{i:04d}", "text": c}) + "\n")
```

Then run:

```bash
export GROQ_API_KEY="..."        # Windows: $env:GROQ_API_KEY="..."
python -m src.rag.tagger
```

### ✅ Verification block — Phase 6.1

```python
import json
with open("data/kb_chunks_tagged.jsonl") as f:
    samples = [json.loads(l) for _, l in zip(range(5), f)]
for s in samples:
    print(s["id"], "→", s["metadata"]["applicable_postures"],
          "/", s["metadata"]["anatomical_region"])
```

**Pass:** at least 80% of inspected tags look reasonable.

**Sanity check:** count distribution across postures:

```python
from collections import Counter
import json
c = Counter()
with open("data/kb_chunks_tagged.jsonl") as f:
    for line in f:
        for p in json.loads(line)["metadata"]["applicable_postures"]:
            c[p] += 1
print(c)
```

You should see all 4 classes represented, with `correct_posture` somewhat over-represented (general ergonomics tagged broadly) — that's fine.

## 6.2 Rebuild ChromaDB with metadata

**File:** `src/rag/build_index.py`

```python
"""Build rag_db_v3 from tagged chunks. Local BGE-small embeddings."""
import json
from pathlib import Path
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "rag_db_v3"
COLLECTION = "posture_kb_v3"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def build(jsonl_path: str = "data/kb_chunks_tagged.jsonl"):
    embedder = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    coll = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    ids, texts, metadatas = [], [], []
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            md = obj["metadata"]
            # ChromaDB metadata must be scalar — flatten lists to comma-joined str
            flat = {
                "applicable_postures": ",".join(md["applicable_postures"]),
                "anatomical_region": md["anatomical_region"],
                "content_type": md["content_type"],
                "key_terms": ",".join(md.get("key_terms", [])),
            }
            ids.append(obj["id"])
            texts.append(obj["text"])
            metadatas.append(flat)

    embeddings = embedder.encode(texts, show_progress_bar=True,
                                 normalize_embeddings=True).tolist()
    BATCH = 200
    for i in range(0, len(ids), BATCH):
        coll.add(ids=ids[i:i+BATCH],
                 documents=texts[i:i+BATCH],
                 embeddings=embeddings[i:i+BATCH],
                 metadatas=metadatas[i:i+BATCH])
    print(f"Indexed {len(ids)} chunks → {DB_PATH}/{COLLECTION}")


if __name__ == "__main__":
    build()
```

```bash
python -m src.rag.build_index
```

### ✅ Verification block — Phase 6.2

```python
import chromadb
client = chromadb.PersistentClient(path="rag_db_v3")
coll = client.get_collection("posture_kb_v3")
print(f"Total chunks: {coll.count()}")
sample = coll.get(limit=3, include=["documents", "metadatas"])
for d, m in zip(sample["documents"], sample["metadatas"]):
    print(m, "→", d[:80])

# Posture-filtered query
res = coll.query(
    query_texts=["why is my upper back rounded"],
    n_results=3,
    where={"applicable_postures": {"$eq": "slouching,neck_forward"}}
)
# Note: $eq on comma-string won't work for partial match.
# We need a different filter approach — see 6.2.1 below.
```

ChromaDB metadata filtering doesn't support "contains substring" cleanly. We use a workaround: store one row per posture per chunk, OR filter in Python after retrieval. The simpler approach is **post-filter in Python**.

## 6.2.1 Hybrid retrieval with Python-side post-filtering

This is the file you'll actually use for retrieval:

**File:** `src/rag/retriever.py`

```python
"""Hybrid (dense + BM25) retrieval with posture metadata filtering."""
import math
from typing import List, Dict, Optional
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DB_PATH = "rag_db_v3"
COLLECTION = "posture_kb_v3"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in text.split() if len(w) > 2]


class PostureRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer(MODEL_NAME)
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.coll = self.client.get_collection(COLLECTION)
        # Load entire corpus for BM25 (fine for ~1k chunks)
        all_data = self.coll.get(include=["documents", "metadatas"])
        self.ids = all_data["ids"]
        self.docs = all_data["documents"]
        self.metadatas = all_data["metadatas"]
        self.bm25 = BM25Okapi([_tokenize(d) for d in self.docs])
        self.id_to_idx = {i: idx for idx, i in enumerate(self.ids)}

    def retrieve(self,
                 query: str,
                 current_posture: Optional[str] = None,
                 anatomical_filter: Optional[List[str]] = None,
                 top_k: int = 5,
                 candidates_per_stream: int = 20) -> List[Dict]:
        """Hybrid retrieval. Filters by posture class metadata."""
        # 1. Dense
        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()
        dense = self.coll.query(query_embeddings=[q_emb],
                                n_results=candidates_per_stream,
                                include=["documents", "metadatas", "distances"])
        dense_ids = dense["ids"][0]

        # 2. BM25
        scores = self.bm25.get_scores(_tokenize(query))
        bm25_ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:candidates_per_stream]
        bm25_ids = [self.ids[i] for i in bm25_ranked]

        # 3. RRF fusion
        rrf_k = 60
        rrf_scores: Dict[str, float] = {}
        for rank, did in enumerate(dense_ids):
            rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (rrf_k + rank)
        for rank, did in enumerate(bm25_ids):
            rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (rrf_k + rank)
        fused = sorted(rrf_scores.items(), key=lambda kv: -kv[1])

        # 4. Posture filter (Python-side)
        results = []
        for did, score in fused:
            idx = self.id_to_idx[did]
            md = self.metadatas[idx]
            postures = md["applicable_postures"].split(",")
            if current_posture is not None and current_posture not in postures:
                continue
            if anatomical_filter and md["anatomical_region"] not in anatomical_filter:
                continue
            results.append({
                "id": did,
                "text": self.docs[idx],
                "metadata": md,
                "rrf_score": score,
            })
            if len(results) >= top_k:
                break

        # If filter eliminated everything, fall back to top-k unfiltered
        if not results:
            for did, score in fused[:top_k]:
                idx = self.id_to_idx[did]
                results.append({
                    "id": did,
                    "text": self.docs[idx],
                    "metadata": self.metadatas[idx],
                    "rrf_score": score,
                })

        return results
```

### ✅ Verification block — Phase 6.2.1

```python
from src.rag.retriever import PostureRetriever
r = PostureRetriever()
hits = r.retrieve("why am I rounding my back", current_posture="slouching", top_k=3)
for h in hits:
    print(h["metadata"]["applicable_postures"], "→", h["text"][:120])
```

**Pass:** Returned chunks all have `slouching` in their `applicable_postures`, and the content is genuinely about slouching/upper-back rounding.

## 6.3 (Optional, recommended) Hosted reranker

If you have access to **Cohere Rerank** (free tier sufficient) or **Jina Rerank**:

**File:** `src/rag/reranker.py`

```python
"""Optional hosted reranker. Returns top-k of input candidates by relevance."""
import os
from typing import List, Dict, Optional


class CohereReranker:
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "rerank-english-v3.0"):
        import cohere
        self.client = cohere.Client(api_key or os.environ["COHERE_API_KEY"])
        self.model = model

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        docs = [c["text"] for c in candidates]
        result = self.client.rerank(query=query, documents=docs,
                                    model=self.model, top_n=top_k)
        out = []
        for r in result.results:
            cand = candidates[r.index]
            cand["rerank_score"] = float(r.relevance_score)
            out.append(cand)
        return out


class NoopReranker:
    """Drop-in replacement if no reranker is configured."""
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5):
        return candidates[:top_k]
```

Wire it into `retriever.py` by adding a `reranker` parameter; for now, default to `NoopReranker()`.

## 6.4 Prompt builder

**File:** `src/rag/prompt_builder.py`

```python
"""Builds the LLM prompt from PostureState + retrieved chunks + user query."""
from typing import List, Dict
from src.state.posture_state import PostureState, NORMAL_THRESHOLDS

SYSTEM = """You are a posture correction coach. You give specific, evidence-based
guidance grounded ONLY in the retrieved physiotherapy references below.

RULES:
- Refer to the user's actual measured values from the OBSERVATION block.
- Quote or paraphrase from the REFERENCES for any physiological or corrective claim.
- If the references do not cover the user's question, say so plainly.
- Keep responses to 3-6 sentences unless the user asks for detail.
- Use friendly, encouraging language. Don't be alarmist."""


def build_prompt(state: PostureState, retrieved: List[Dict],
                 user_question: str) -> List[Dict]:
    if not state.is_reliable:
        observation = ("The camera cannot reliably see the user's posture right now. "
                       "Ask the user to reposition before answering.")
    else:
        observation = _format_observation(state)

    refs = "\n\n".join(
        f"[REF {i+1}] (postures: {r['metadata']['applicable_postures']}, "
        f"region: {r['metadata']['anatomical_region']})\n{r['text']}"
        for i, r in enumerate(retrieved)
    ) or "[No relevant references retrieved]"

    user_block = f"""=== CURRENT POSTURE OBSERVATION ===
{observation}

=== RETRIEVED REFERENCES ===
{refs}

=== USER QUESTION ===
{user_question}

Answer the user question now, following all the rules above."""

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_block},
    ]


def _format_observation(state: PostureState) -> str:
    devs = state.feature_deviations
    bad = [k for k, v in devs.items() if v > 0]
    bad_str = ", ".join(bad) if bad else "no major deviations"
    dist = ", ".join(f"{k}: {v*100:.0f}%"
                     for k, v in state.posture_distribution.items() if v > 0.05)
    return f"""The user's camera currently shows:
- Posture: {state.posture_class} (confidence {state.confidence:.2f})
- Time in this posture: {state.posture_duration_sec:.0f} seconds
- Session duration: {state.session_duration_sec:.0f} seconds
- Session breakdown: {dist}
- Correction events this session: {state.correction_events}

Measured indicators (normal range in parens):
- Forward head offset:   {state.ear_shoulder_offset_x:+.2f}  (normal < {NORMAL_THRESHOLDS['ear_shoulder_offset_x']:.2f})
- Craniovertebral angle: {state.craniovertebral_angle:.0f}°  (normal < {NORMAL_THRESHOLDS['craniovertebral_max']:.0f}°)
- Shoulder forward roll: {state.shoulder_roll_z:+.2f}  (normal < {NORMAL_THRESHOLDS['shoulder_roll_z']:.2f})
- Torso compression:     {state.torso_compression_ratio:.2f}  (normal > {NORMAL_THRESHOLDS['torso_compression_min']:.2f})
- Shoulder tilt:         {state.shoulder_tilt_angle:+.1f}°  (normal < {NORMAL_THRESHOLDS['shoulder_tilt_abs_max']:.0f}°)
- Midline deviation:     {state.midline_deviation_angle:.1f}°  (normal < {NORMAL_THRESHOLDS['midline_deviation_max']:.0f}°)

Primary issue: {state.primary_issue}
Issues exceeding threshold: {bad_str}"""
```

## 6.5 Grounding check (anti-hallucination)

**File:** `src/rag/grounding_check.py`

```python
"""Quick check: does the response reference at least N key terms from retrieved chunks?"""
import re
from typing import List, Dict


def _extract_terms(text: str) -> set:
    """Coarse: nouns/jargon = words 5+ chars, not stop words, lowercased."""
    STOP = {"about", "above", "after", "again", "against", "because",
            "before", "below", "between", "during", "should", "would", "could",
            "their", "there", "these", "those", "where", "which", "while"}
    words = re.findall(r"[a-zA-Z]{5,}", text.lower())
    return {w for w in words if w not in STOP}


def is_grounded(response: str, retrieved: List[Dict],
                min_overlap: int = 2) -> bool:
    chunk_terms = set()
    for r in retrieved:
        chunk_terms |= _extract_terms(r["text"])
    response_terms = _extract_terms(response)
    return len(chunk_terms & response_terms) >= min_overlap


def fallback_response(state) -> str:
    """When the response fails grounding, return a templated state-only answer."""
    if state.primary_issue == "none":
        return ("Your posture currently looks well aligned. Keep your head "
                "stacked over your shoulders and your shoulders over your hips.")
    issue = state.primary_issue.replace("_", " ")
    return (f"I don't have detailed references for your specific question, but "
            f"your current measurements show {issue} as the main deviation. "
            f"Try small adjustments and watch the indicator above.")
```

## 6.6 Updated `rag_query.py`

This is the file you call from chat. Replace your existing logic with this:

**File:** `rag_query.py` (REWRITE)

```python
"""Posture-aware RAG query. Replaces the previous generic flow."""
import os
from groq import Groq

from src.rag.retriever import PostureRetriever
from src.rag.prompt_builder import build_prompt
from src.rag.grounding_check import is_grounded, fallback_response
from src.state.posture_state import PostureState

GROQ_CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
GROQ_MODEL = "llama-3.1-8b-instant"

# Module-level singleton so retriever loads once
_RETRIEVER: PostureRetriever = None


def get_retriever() -> PostureRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = PostureRetriever()
    return _RETRIEVER


def answer(user_question: str, state: PostureState) -> dict:
    """Returns dict with keys: text, retrieved, grounded."""
    retriever = get_retriever()
    posture = state.posture_class if state and state.is_reliable else None

    # Expand query with current state for better retrieval
    expanded = user_question
    if state and state.is_reliable:
        expanded = f"{user_question} [posture: {state.posture_class}, issue: {state.primary_issue}]"

    retrieved = retriever.retrieve(expanded, current_posture=posture, top_k=5)
    messages = build_prompt(state, retrieved, user_question)

    resp = GROQ_CLIENT.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=400,
    )
    text = resp.choices[0].message.content.strip()
    grounded = is_grounded(text, retrieved)
    if not grounded:
        text = fallback_response(state) + "\n\n" + text  # both, user can compare
    return {"text": text, "retrieved": retrieved, "grounded": grounded}
```

### ✅ Verification block — Phase 6.6

Quick offline test:

```python
from datetime import datetime
from src.state.posture_state import PostureState
from rag_query import answer

# Simulate a slouching state
state = PostureState(
    posture_class="slouching", confidence=0.91,
    class_probabilities={"correct_posture": 0.05, "slouching": 0.91,
                         "neck_forward": 0.03, "lean": 0.01},
    ear_shoulder_offset_x=0.32, craniovertebral_angle=18,
    head_forward_offset_z=0.32, nose_shoulder_offset_x=0.05,
    shoulder_roll_z=0.21, torso_compression_ratio=1.32,
    elbow_forward_offset_z=0.18, spine_angle_3d=8,
    shoulder_tilt_angle=1.2, hip_tilt_angle=0.5,
    midline_deviation_angle=1.0, nose_centerline_offset_x=0.04,
    lateral_asymmetry_index=0.01, landmark_confidence_mean=0.95,
    is_reliable=True, timestamp=datetime.now(),
    posture_duration_sec=87, session_duration_sec=420,
    posture_distribution={"correct_posture": 0.4, "slouching": 0.5,
                          "neck_forward": 0.05, "lean": 0.05},
    correction_events=3, longest_bad_posture_streak_sec=140,
)

out = answer("Am I sitting correctly?", state)
print(out["text"])
print("\nGrounded:", out["grounded"])
print("Refs:", [r["id"] for r in out["retrieved"]])
```

**Pass:**
- Response specifically mentions slouching, forward shoulder roll, or torso compression.
- Response refers to actual values (e.g. "your forward head offset of 0.32...")
- `grounded=True`

**Fail: response is generic and ignores measured values:**
- Check the system prompt is included.
- Verify the observation block is being injected — print `messages[1]["content"]` from `rag_query`.

## 6.7 Git checkpoint

```bash
git add -A
git commit -m "Phase 6: posture-conditioned RAG (metadata + hybrid + posture prompt)"
```

---

# PHASE 7 — CV → RAG Integration `[FAST PATH]`

**Objective:** CV pipeline runs in background, continuously updates a shared `PostureState`. Chat reads that state when the user sends a message. Streamlit shows both live.
**Estimated time:** 2–3 hours.

## 7.1 Shared state

**File:** `src/workers/shared_state.py`

```python
"""Thread-safe holder for the current PostureState. One per process."""
import copy
import threading
from collections import deque
from typing import Optional, List

from src.state.posture_state import PostureState


class SharedPostureState:
    def __init__(self, history_size: int = 4500):  # 5 min @ 15 fps
        self._lock = threading.Lock()
        self._current: Optional[PostureState] = None
        self._history: deque = deque(maxlen=history_size)

    def update(self, state: PostureState) -> None:
        with self._lock:
            self._current = state
            self._history.append(state)

    def snapshot(self) -> Optional[PostureState]:
        with self._lock:
            return copy.deepcopy(self._current)

    def history_last_seconds(self, seconds: float) -> List[PostureState]:
        with self._lock:
            if not self._history:
                return []
            cutoff = self._history[-1].timestamp.timestamp() - seconds
            return [s for s in self._history
                    if s.timestamp.timestamp() >= cutoff]
```

## 7.2 CV worker (background thread)

**File:** `src/workers/cv_worker.py`

```python
"""Background thread: capture webcam → run pipeline → update SharedPostureState."""
import threading
import time
import cv2

from src.cv.pipeline import PosturePipeline
from src.workers.shared_state import SharedPostureState


class CVWorker(threading.Thread):
    def __init__(self,
                 shared: SharedPostureState,
                 camera_index: int = 0,
                 width: int = 640,
                 height: int = 480,
                 model_complexity: int = 1,
                 target_fps: int = 15):
        super().__init__(daemon=True)
        self.shared = shared
        self.camera_index = camera_index
        self.width, self.height = width, height
        self.model_complexity = model_complexity
        self.frame_interval = 1.0 / target_fps
        self.stop_event = threading.Event()
        # Latest BGR frame for UI overlay
        self._latest_frame = None
        self._frame_lock = threading.Lock()

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        pipe = PosturePipeline(model_complexity=self.model_complexity)

        try:
            while not self.stop_event.is_set():
                t0 = time.time()
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                state = pipe.step(frame)
                if state is not None:
                    self.shared.update(state)
                with self._frame_lock:
                    self._latest_frame = frame
                elapsed = time.time() - t0
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
        finally:
            cap.release()
            pipe.close()

    def stop(self):
        self.stop_event.set()

    def latest_frame(self):
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()
```

## 7.3 Chat worker (no thread needed — synchronous on user message)

**File:** `src/workers/chat_worker.py`

```python
"""Handles incoming user messages. Reads shared state, calls RAG."""
from typing import Optional
from src.workers.shared_state import SharedPostureState
from rag_query import answer


class ChatWorker:
    def __init__(self, shared: SharedPostureState):
        self.shared = shared

    def respond(self, user_message: str) -> dict:
        state = self.shared.snapshot()
        if state is None:
            return {"text": "Camera is starting up — give me a moment.",
                    "retrieved": [], "grounded": False}
        return answer(user_message, state)
```

## 7.4 Streamlit app — wire it all together

**File:** `app.py` (REWRITE — back up your existing one first)

```python
"""G2B Posture Correction Coach — live Streamlit app."""
import time
import cv2
import streamlit as st

from src.workers.shared_state import SharedPostureState
from src.workers.cv_worker import CVWorker
from src.workers.chat_worker import ChatWorker
from src.cv.pose_extractor import LANDMARK_INDICES

st.set_page_config(page_title="G2B Posture Coach", layout="wide")

# === Init singletons (survive across reruns) ===
if "shared" not in st.session_state:
    st.session_state.shared = SharedPostureState()
    st.session_state.cv_worker = CVWorker(st.session_state.shared,
                                          model_complexity=1, target_fps=15)
    st.session_state.cv_worker.start()
    st.session_state.chat_worker = ChatWorker(st.session_state.shared)
    st.session_state.chat_history = []

# === Layout: two columns ===
left, right = st.columns([1, 1])

with left:
    st.subheader("Live view")
    frame_slot = st.empty()
    state_slot = st.empty()
    metrics_slot = st.empty()

with right:
    st.subheader("Chat with your coach")
    chat_area = st.container()
    user_input = st.chat_input("Ask about your posture...")

    # Show existing history
    with chat_area:
        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(text)

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with chat_area:
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.chat_worker.respond(user_input)
                st.write(result["text"])
                st.session_state.chat_history.append(("assistant", result["text"]))

# === Continuously refresh the live view (5 Hz is plenty for UI) ===
LABEL_COLORS = {
    "correct_posture": "🟢",
    "slouching":       "🟠",
    "neck_forward":    "🟡",
    "lean":            "🟣",
}
for _ in range(20):  # ~4 seconds of refresh
    frame = st.session_state.cv_worker.latest_frame()
    state = st.session_state.shared.snapshot()
    if frame is not None:
        frame_slot.image(frame, channels="BGR", use_column_width=True)
    if state is not None:
        emoji = LABEL_COLORS.get(state.posture_class, "")
        state_slot.markdown(
            f"### {emoji} **{state.posture_class.replace('_', ' ').title()}**  "
            f"`conf={state.confidence:.2f}`  "
            f"_(holding for {state.posture_duration_sec:.0f}s)_"
        )
        cols = metrics_slot.columns(3)
        cols[0].metric("Forward head", f"{state.ear_shoulder_offset_x:+.2f}",
                       delta=f"{state.feature_deviations['forward_head']:.2f}")
        cols[1].metric("Shoulder roll", f"{state.shoulder_roll_z:+.2f}",
                       delta=f"{state.feature_deviations['shoulder_roll']:.2f}")
        cols[2].metric("Tilt", f"{state.shoulder_tilt_angle:+.1f}°",
                       delta=f"{state.feature_deviations['shoulder_tilt']:.1f}")
    time.sleep(0.2)

st.rerun()
```

Run:

```bash
streamlit run app.py
```

### ✅ Verification block — Phase 7.4

Open the browser tab. You should see:
- Webcam feed on the left, updating ~5 Hz.
- Posture label updating live with emoji + confidence.
- Metric tiles updating each frame.
- Chat input on the right.

Type **"Am I sitting correctly?"** — should get a posture-grounded response within ~1 second.

Type **"Why am I classified as X?"** while exhibiting different postures — response should reference your *current* measured values.

**Fail: app crashes with `RuntimeError: Tried to instantiate class '...'`:**
- Streamlit re-imports modules; that's why we use `st.session_state`. Make sure you start the CV worker only on first run (the `if "shared" not in st.session_state:` guard).

**Fail: chat shows the *previous* posture state:**
- The state is read at message send time, not at response display time. That's intended — the LLM call is what's slow, posture can change while it's thinking. If this is jarring, snapshot the state again after the LLM responds and show it next to the answer.

## 7.5 Update `main.py` (optional CLI runner)

If you have a CLI entry point you still want, replace with:

**File:** `main.py` (REWRITE)

```python
"""Headless mode: prints PostureState updates to stdout. No UI."""
import time
import cv2

from src.cv.pipeline import PosturePipeline


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    pipe = PosturePipeline(model_complexity=1)
    last = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            s = pipe.step(frame)
            if s is None: continue
            if s.posture_class != last:
                print(f"[{s.timestamp.strftime('%H:%M:%S')}] {s.posture_class}  "
                      f"conf={s.confidence:.2f}  primary={s.primary_issue}")
                last = s.posture_class
    finally:
        cap.release()
        pipe.close()


if __name__ == "__main__":
    main()
```

## 7.6 Git checkpoint

```bash
git add -A
git commit -m "Phase 7: workers + shared state + Streamlit app integrated"
```

---

# PHASE 8 — Raspberry Pi 5 Deployment

**Objective:** Run the entire stack on a Pi 5 at ≥10 fps end-to-end. Auto-start on boot.
**Estimated time:** 3–4 hours (first time; faster on subsequent deploys).

## 8.1 Pi 5 OS prep

1. Flash **Raspberry Pi OS 64-bit (Bookworm)** with Pi Imager.
2. Boot, configure WiFi, run updates:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip git build-essential \
    libgl1 libglib2.0-0 libgtk-3-0 libcamera-apps v4l-utils
```

3. Enable camera (if using Pi Camera Module rather than USB):

```bash
sudo raspi-config nonint do_camera 0
```

## 8.2 Get the project onto the Pi

```bash
cd ~
git clone <your-repo-url> g2b_posture_coach
cd g2b_posture_coach
git checkout feature/redesign-v3
```

## 8.3 Create venv + install deps

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# MediaPipe on Pi 5 — official wheel works on 64-bit OS.
# If it doesn't, fallback to community wheel:
#   pip install --index-url https://www.piwheels.org/simple mediapipe
pip install -r requirements.txt
```

Watch out for these specific install gotchas:

| Symptom | Fix |
|---|---|
| `mediapipe` install fails: "no matching distribution" | Pi OS must be 64-bit Bookworm. Run `uname -m` — must say `aarch64`. |
| `sentence-transformers` downloads PyTorch CPU — large but OK | Just wait. First install is ~15 min. |
| `chromadb` complains about onnxruntime | `pip install onnxruntime==1.18.0` explicitly. |
| `Failed building wheel for hnswlib` | `sudo apt install python3.11-dev` then reinstall chromadb. |

## 8.4 Configure for Pi-grade performance

**Edit `src/workers/cv_worker.py`** — pass `model_complexity=0` (Lite) when on Pi:

Better: make it configurable via env var.

**File:** `src/utils/config.py` (NEW)

```python
"""Runtime configuration. Reads env vars with sensible defaults."""
import os
import platform


def is_pi() -> bool:
    return platform.machine() in ("aarch64", "armv7l")


def mediapipe_complexity() -> int:
    v = os.environ.get("G2B_MP_COMPLEXITY")
    if v is not None: return int(v)
    return 0 if is_pi() else 1


def target_fps() -> int:
    v = os.environ.get("G2B_TARGET_FPS")
    if v is not None: return int(v)
    return 12 if is_pi() else 15


def camera_resolution():
    if is_pi():
        return (640, 480)
    return (640, 480)
```

Update `app.py` to use this:

```python
from src.utils.config import mediapipe_complexity, target_fps, camera_resolution
w, h = camera_resolution()
st.session_state.cv_worker = CVWorker(
    st.session_state.shared,
    width=w, height=h,
    model_complexity=mediapipe_complexity(),
    target_fps=target_fps(),
)
```

## 8.5 Copy the trained model to the Pi

The LGBM model is tiny (~50 KB):

```bash
# From dev machine:
scp models/posture_lgbm_v3.txt pi@<pi-ip>:~/g2b_posture_coach/models/
scp models/feature_order.json  pi@<pi-ip>:~/g2b_posture_coach/models/
scp -r rag_db_v3/              pi@<pi-ip>:~/g2b_posture_coach/
```

## 8.6 Set the API keys

```bash
echo 'export GROQ_API_KEY="your-key-here"' >> ~/.bashrc
# Optional reranker:
echo 'export COHERE_API_KEY="..."' >> ~/.bashrc
source ~/.bashrc
```

Or use a `.env` file at the project root and load via `python-dotenv` (already in requirements).

## 8.7 First Pi run — measure FPS

```bash
cd ~/g2b_posture_coach
source .venv/bin/activate
python tests/test_pipeline.py 2>&1 | head -100
```

Watch the FPS print at the end. Target: **≥10 fps**.

### ✅ Verification block — Phase 8.7

| FPS observed | Action |
|---|---|
| ≥12 | 🎉 Proceed |
| 8–11 | Acceptable; consider 1.7-GHz overclock (low fan speed will be needed) |
| 5–7 | Drop resolution to 480×360. Disable any other heavy processes. |
| <5 | Verify `model_complexity=0`. If still bad, check `htop` — something else is hammering CPU. |

## 8.8 Run Streamlit on Pi accessible from LAN

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://<pi-ip>:8501` from another machine on the same network.

### ✅ Verification block — Phase 8.8

- App loads in browser.
- Webcam shows live.
- Posture classifies correctly across all 4 classes.
- Chat responds within ~2 seconds.
- No memory error, no thread crash after 5 minutes of continuous use.

## 8.9 Auto-start on boot (for demo day)

**File:** `/etc/systemd/system/g2b-coach.service` (create as root)

```ini
[Unit]
Description=G2B Posture Correction Coach
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/g2b_posture_coach
EnvironmentFile=/home/pi/g2b_posture_coach/.env
ExecStart=/home/pi/g2b_posture_coach/.venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable + start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable g2b-coach.service
sudo systemctl start g2b-coach.service
sudo systemctl status g2b-coach.service
```

Logs:

```bash
journalctl -u g2b-coach.service -f
```

## 8.10 (Optional) Kiosk mode for demo

Auto-open browser fullscreen on Pi display:

```bash
sudo apt install -y chromium-browser unclutter
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/g2b-coach.desktop << EOF
[Desktop Entry]
Type=Application
Name=G2B Coach Kiosk
Exec=chromium-browser --kiosk --noerrdialogs --disable-translate http://localhost:8501
X-GNOME-Autostart-enabled=true
EOF
```

## 8.11 Git checkpoint

```bash
git add -A
git commit -m "Phase 8: Pi 5 deployment config + systemd + perf tuning"
```

---

# PHASE 9 — Debugging Checklist

Use this when something specific breaks. Find the symptom, follow the diagnosis.

## 9.1 MediaPipe / protobuf errors

| Error message | Diagnosis | Fix |
|---|---|---|
| `TypeError: Descriptors cannot not be created directly` | protobuf 5.x leaked in | `pip install --force-reinstall protobuf==4.25.3` |
| `ImportError: cannot import name 'builder' from 'google.protobuf.internal'` | protobuf <3.20 | same fix |
| `DLL load failed while importing _framework_bindings` (Windows) | missing MSVC runtime | Install VC++ Redist 2015-2022 |
| `libGL.so.1: cannot open shared object file` (Linux/Pi) | missing OS lib | `sudo apt install libgl1` |
| `Could not find a version that satisfies mediapipe` (Pi) | 32-bit OS or wrong Python | Verify `uname -m == aarch64` and Python 3.11 |
| `numpy.core.multiarray failed to import` | NumPy 2.x | `pip install --force-reinstall numpy==1.26.4` |

## 9.2 Webcam issues

| Symptom | Fix |
|---|---|
| `cv2.VideoCapture(0)` returns `False` | Try `1`, `2`, etc. On Linux: `ls /dev/video*` |
| Blue/green tinted frames | Driver issue. On Pi: `sudo modprobe bcm2835-v4l2` |
| Frames are upside-down | Add `frame = cv2.flip(frame, -1)` after `cap.read()` |
| FPS keeps dropping over time | OpenCV buffer is filling. Ensure `CAP_PROP_BUFFERSIZE=1` is set. |
| `cv2.error: (-215:Assertion failed) !_src.empty()` | Bad/disconnected camera. Catch and skip the frame. |

## 9.3 Classifier accuracy is bad live

Even though offline test set scored well.

| Symptom | Diagnosis | Fix |
|---|---|---|
| Everything classifies as `correct_posture` | Probabilities not extreme; rule overlay dominates | Lower `rule_weight` to 0.15 in `PostureClassifier` |
| Everything classifies as one wrong class | Likely a swapped class index | Print `LABEL_TO_IDX` and verify it matches `CLASSES` order in `rule_engine.py` |
| Slouching only fires when extreme | Threshold too high | In `rule_engine.py` lower `TH_SHOULDER_ROLL` to 0.08 |
| Lean misfires constantly when sitting straight | Webcam at angle | Recenter webcam at eye level, square to user |
| Confidence values capped at ~0.4 | EMA hasn't warmed up | Wait 1-2 seconds after start before first reading |

## 9.4 Smoother behaves wrong

| Symptom | Fix |
|---|---|
| Label change feels too slow | Lower `hysteresis_frames` to 5; raise `alpha` to 0.4 |
| Constant flicker between two labels | Raise `hysteresis_frames` to 12 |
| Stuck on one label forever | Bug — verify `current_label` updates inside the streak check; run `test_smoother.py` |

## 9.5 RAG returns irrelevant answers

| Symptom | Diagnosis | Fix |
|---|---|---|
| Generic textbook content unrelated to question | Metadata filter not applied | Print `current_posture` arg in `retrieve()`. Should be the live class. |
| Always falls back to default response | Grounding check too strict | Lower `min_overlap` in `is_grounded` to 1 |
| LLM ignores measured values | Prompt observation block missing | Print `messages[1]["content"]` — verify "OBSERVATION" section is there |
| Hallucinated anatomical terms | Reranker not in use; small Llama drifts | Add Cohere Rerank (§6.3) or temperature down to 0.1 |
| Slow first response | Embedding model loading | Add `embedder.encode(["warm"])` at startup |

## 9.6 Streamlit threading issues

| Symptom | Fix |
|---|---|
| App restarts when typing in chat | `st.rerun()` is firing during a callback. Move rerun outside callbacks. |
| Webcam frame never updates | CV worker died silently. Check `cv_worker.is_alive()` in app, log if False. |
| Multiple webcam processes | Streamlit reloaded — your `if "shared" not in st.session_state:` guard isn't catching it. Verify session_state key |
| Stale state in chat | Snapshot is taken at message send. That's intended — see Phase 7.4 verification notes. |

## 9.7 Pi 5 performance issues

| Symptom | Fix |
|---|---|
| FPS < 8 | Drop to model_complexity=0; lower resolution to 480×360 |
| CPU pegged at 100% one core | MediaPipe is single-threaded for inference. Run a different model_complexity. |
| Thermal throttling | Add fan + heatsink. Run `vcgencmd measure_temp` while live. |
| Memory pressure | Disable browser kiosk during testing; use remote browser. |
| Streamlit unresponsive after 30 min | Memory leak in browser tab. Refresh tab; consider non-Streamlit (FastAPI + lightweight HTML) for production. |

## 9.8 Common silent failures

These don't crash but produce wrong behavior:

1. **CSV column-name mismatch silently drops everything.** If `csv_to_features.py` produces zero rows, check column names match Pattern A in Phase 2.1.
2. **Stale `posture_classifier_v2.pkl` loaded.** If accuracy is suspiciously the same as before, your code is loading the old model. Search for `.pkl` in code and confirm only `posture_lgbm_v3.txt` is referenced.
3. **`__init__.py` files missing** → import error you may not notice if you import lazily. Verify every directory under `src/` has one.
4. **Hip midpoint of all zeros after normalize** → check the normalize function isn't returning the *un-modified* array.
5. **`is_reliable=False` always** → visibility threshold too high. Drop `VISIBILITY_MIN` to 0.4.

---

# PHASE 10 — Final Demo Preparation `[POLISH]`

**Objective:** Production-quality demo. Confident defense.
**Estimated time:** 3–4 hours.

## 10.1 Polish the UI

**Add to `app.py`:**

```python
# Header
st.markdown("# G2B Posture Correction Coach")
st.caption("Live posture analysis + AI coaching — built for Raspberry Pi 5")

# Session summary bar at the top
if state is not None:
    cols = st.columns(4)
    cols[0].metric("Session", f"{state.session_duration_sec/60:.1f} min")
    cols[1].metric("Correct posture", f"{state.posture_distribution.get('correct_posture', 0)*100:.0f}%")
    cols[2].metric("Corrections", state.correction_events)
    cols[3].metric("Worst streak", f"{state.longest_bad_posture_streak_sec:.0f}s")
```

**Add landmark overlay to webcam feed** so panelists can see the system "looking":

```python
import mediapipe as mp

def draw_pose_overlay(frame, landmarks_array):
    """Quick: draw the 9 landmarks we use as red dots."""
    h, w = frame.shape[:2]
    for x_norm, y_norm, _, _ in landmarks_array:
        cx, cy = int(x_norm * w), int(y_norm * h)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
    # Shoulder line
    ls, rs = landmarks_array[3], landmarks_array[4]
    cv2.line(frame, (int(ls[0]*w), int(ls[1]*h)),
                    (int(rs[0]*w), int(rs[1]*h)), (0, 255, 0), 2)
    return frame
```

Call this in CVWorker before storing `_latest_frame`. (Requires keeping the raw landmark array on the state or shared elsewhere.)

## 10.2 Demo script (run-through, 5 minutes)

| Time | Action | Talking point |
|---|---|---|
| 0:00 | Sit correct posture in front of Pi | "System is observing 14 features per frame at 12 fps." |
| 0:20 | Slowly slouch | "Notice the label changes ~half a second after my posture changes — that's the temporal hysteresis preventing flicker." |
| 0:40 | Ask: "Why am I classified as slouching?" | "The chatbot is reading the live posture state, not just my question. Notice it cites specific measured values." |
| 1:30 | Sit up | Show `correction_events` increment. |
| 1:45 | Lean to one side | Show different label, different `primary_issue`. |
| 2:00 | Push head forward (keep torso straight) | "This is `neck_forward` — the system distinguishes it from `slouching` because shoulder roll stays low while ear offset rises." |
| 3:00 | Ask: "What's the first thing I should fix?" | Show how primary_issue drives the answer. |
| 4:00 | Show session stats | "Over this demo, I spent X% in correct posture with Y corrections." |
| 4:30 | Wrap | Summarize improvements over v1 (72% → 90%, generic → grounded). |

## 10.3 Panel defense Q&A — likely questions

| Question | Prepared answer |
|---|---|
| **"Why LightGBM over Random Forest?"** | Lower latency (~1ms vs ~3ms), smaller (~3MB vs ~10MB), marginally more accurate at our scale. On Pi 5 the latency margin matters. |
| **"How do you handle different body types / heights / cameras?"** | All features are normalized by shoulder width and centered on hip midpoint, so they're invariant to body scale and approximate camera distance. We don't normalize for camera angle — instead our augmentation includes small rotational perturbations. |
| **"What happens if the camera can't see the user clearly?"** | The pipeline computes `landmark_confidence_mean` from MediaPipe visibility scores. If key landmarks are < 0.5 visible, we mark the frame as `is_reliable=False`, hold the previous state, and the chatbot will ask the user to reposition rather than guess. |
| **"How do you avoid hallucinations in the chatbot?"** | Three layers: (1) metadata-filtered retrieval — only chunks tagged for the current posture class are returned; (2) prompt instructs the LLM to refuse if references don't cover the question; (3) automated grounding check on the response, falling back to a templated state-only answer if it fails. |
| **"Isn't synthetic data going to bias your model?"** | We synthesize only geometric transformations of *real* MediaPipe-extracted landmarks. The deformations match how MediaPipe physically sees real slouching (forward z-shift, depth roll). Real held-out test data remains untouched as ground truth — if a synthetic-augmented model generalizes there, the augmentation is valid. We can also show non-synthetic-only metrics if asked. |
| **"What's the latency end-to-end?"** | Per frame: ~75ms (CV) → 12-15 fps. Per chat response: ~900ms first token. The bottleneck for chat is Groq + reranker; CV is dominated by MediaPipe. |
| **"Could this run without internet?"** | Almost. CV is fully offline. Embeddings are local (BGE-small). The only network call is the Groq LLM. We could swap to a 1B Qwen running on Pi, but latency would jump from 500ms to ~5s. |
| **"How would you scale to more posture classes?"** | Add the new class to `CLASSES`, define its features and rule thresholds in `rule_engine.py`, collect/synthesize samples, retrain LGBM. No architectural changes needed. |
| **"What's your test accuracy?"** | [Insert your actual number from Phase 4.1.] Per-class F1 ranges from X to Y. The hardest class is slouching, because it's the most spectrum-like. |
| **"What about depth ambiguity from a 2D webcam?"** | MediaPipe estimates relative depth (z-coords) — they're not metric depth, but they're consistent enough to detect shoulder roll. We deliberately use *relative* features (offsets and angles, not absolute distances) so we don't depend on metric depth. |

## 10.4 Record a backup demo video

Before demo day, record a perfect run:

```bash
# On Pi or dev machine
sudo apt install -y ffmpeg
ffmpeg -f x11grab -video_size 1280x720 -i :0.0 -framerate 30 \
       -c:v libx264 -preset ultrafast demo_backup.mp4
```

Stop with Ctrl+C when done. Have this on a USB stick in case the live demo fails.

## 10.5 Final checklist (demo morning)

- [ ] Pi boots, joins WiFi, service auto-starts
- [ ] Webcam visible from defense room lighting
- [ ] All 4 postures classify correctly in a 5-minute live test
- [ ] Chat responds with grounded answers across 5 sample questions
- [ ] `journalctl -u g2b-coach.service` shows no errors in the last 30 min
- [ ] Backup MP4 on USB
- [ ] Backup laptop with the same `.venv` ready to demo if Pi fails
- [ ] Printed copy of redesign + execution doc for panelist questions

## 10.6 Final git tag

```bash
git add -A
git commit -m "Phase 10: demo prep complete"
git tag -a v3.0-demo -m "G2B Posture Coach v3.0 — defense-ready"
git push --tags
```

---

# Files I Need From You (Priority Order)

If anything in this guide doesn't match your actual project, **send me these files in this order**:

| Priority | File | Why I need it | What I'll do with it |
|---|---|---|---|
| **P0** | A sample CSV from `CV/` (any one) | To verify column-name pattern (Pattern A vs B vs C in §2.1) | Confirm or rewrite the `_row_to_landmark_array` function with your exact column names. |
| **P0** | `collect_posture.py` | To confirm how landmarks are dumped and what's stored | Make sure my feature-extraction code matches your collection format exactly. |
| **P1** | `retrain.py` | Current RF training script | I can show you exactly which lines to swap to migrate from RF → LGBM in-place if you'd rather not rewrite |
| **P1** | `build_rag.py` | Current ChromaDB chunker | I'll write the exact line edits to emit metadata tags and dump to JSONL for the tagger. |
| **P1** | `rag_query.py` | Current RAG flow | I'll diff against my proposed version and tell you exactly what to keep, what to drop. |
| **P2** | `main.py` | Current entry-point structure | Confirm what's already wired and what the new pipeline replaces. |
| **P2** | `app.py` | Current Streamlit (if any) | Preserve your existing styling/branding while wiring the new workers. |
| **P2** | `session.py` | Current session-state model | If you have session features I should preserve (e.g. logging), I'll integrate them. |
| **P3** | One of the `.pkl` classifiers | To inspect feature schema | Sanity-check no features I'm proposing are duplicates of what's already there. |
| **P3** | Stats: how many CSV files per class, rough row counts | Plan dataset budget | Tune `TARGET_PER_CLASS` and synthesis ratios in `build_dataset.py`. |
| **P3** | List of physiotherapy PDF/text source files | Confirm tagging scope | Estimate tagging cost (Groq calls) and verify chunks are within `max_tokens` budget. |

**Smallest-effort initial check:**
Just send a single CSV from `CV/`, and run the test in §2.1 — that one verification tells me whether you'll need any rename pass at all.

---

# Phase Summary — At-a-Glance

| Phase | Description | Est. time | Critical? | Output |
|---|---|---|---|---|
| 1 | Env setup + Medusa→G2B rename | 1.5–2 h | FAST PATH | Clean `.venv`, new dir tree, project renamed |
| 2 | 14-feature CV pipeline | 3–4 h | FAST PATH | Working `features.py` with verified class separation |
| 3 | Relabel + augment + synthesize | 3–4 h | FAST PATH | `data/augmented/training_set.csv` |
| 4 | LightGBM training + fusion classifier | 1.5–2 h | FAST PATH | `models/posture_lgbm_v3.txt`, ≥85% CV acc |
| 5 | Smoother + PostureState + pipeline | 2 h | FAST PATH | End-to-end CV pipeline, flicker-free |
| 6 | RAG redesign (metadata + hybrid + prompts) | 3–4 h | FAST PATH | `rag_db_v3/`, grounded answers |
| 7 | CV ↔ RAG integration via shared state | 2–3 h | FAST PATH | Working Streamlit app |
| 8 | Pi 5 deployment | 3–4 h | FAST PATH | Auto-starting service on Pi at ≥10 fps |
| 9 | Debugging checklist | as needed | reference | — |
| 10 | Demo polish + Q&A prep | 3–4 h | POLISH | Demo-ready system + panel answers |

**Total fast-path time: ~20–25 working hours.**
Done in 4 days at 5–6 hours/day, or 3 weeks at ~7 hours/week.

---

# Rules of Thumb While Executing

1. **Run the verification block at the end of every phase.** Don't move on if it fails — you'll pay 3× the cost to fix it later.
2. **Commit after every phase.** A failed Phase 7 shouldn't lose your Phase 5 work.
3. **Don't delete the old files** (`posture_classifier.pkl`, `rag_db_v2/`, `CV/`). They're your fallback if v3 has a regression.
4. **Test on Pi 5 by Phase 5.** Don't wait until Phase 8 — MediaPipe behavior on ARM has surprised every project that left it to the end.
5. **If a verification fails 3 times, stop and ask.** Three failed attempts means the problem is upstream of where you're trying to fix it.
6. **Document any deviation from this guide as you go.** Things you change (thresholds, augmentation ratios, retrieval k) — write them in a `CHANGES.md` so you can defend choices later.

End of execution guide. Good luck.
