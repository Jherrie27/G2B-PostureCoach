"""Read existing CV/*.csv files and produce a features DataFrame.

NOTE (G2B fully-synthetic path): the existing CV/*.csv files in THIS project store
only the 3 legacy scalar features (neck_angle, spine_angle, shoulder_tilt) + label —
they do NOT contain raw 33-landmark dumps. This module therefore yields 0 rows on the
current data and is kept for the case where real landmark dumps are collected later
(via a future collect_posture.py that writes landmark_<i>_<x|y|z|v> columns).
The synthetic dataset is built by src/data/build_dataset.py instead.
"""
import re
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

from src.cv.landmarks import LANDMARK_INDICES, ORDERED_NAMES
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
    """Prefer label column; fall back to filename prefix.

    Maps legacy class names to the canonical G2B names.
    """
    raw = None
    if "label" in df.columns and df["label"].notna().any():
        raw = str(df["label"].iloc[0]).strip().lower()
    else:
        name = csv_path.stem.lower()
        if name.startswith("correct"):  raw = "correct"
        elif name.startswith("slouch"): raw = "slouching"
        elif name.startswith("neck"):   raw = "neck_forward"
        elif name.startswith("lean"):   raw = "leaning"
        else:
            raise ValueError(f"Cannot infer label for {csv_path}")
    return canonical_label(raw)


def canonical_label(raw: str) -> str:
    """Map legacy/short class names to the canonical 4 G2B class names."""
    raw = raw.strip().lower()
    mapping = {
        "correct": "correct_posture",
        "correct_posture": "correct_posture",
        "slouch": "slouching",
        "slouching": "slouching",
        "neck": "neck_forward",
        "neck_forward": "neck_forward",
        "lean": "lean",
        "leaning": "lean",
    }
    return mapping.get(raw, raw)


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
    if len(df) == 0:
        print("WARNING: 0 rows extracted. The CV/*.csv files contain no raw landmark "
              "columns (landmark_<i>_<x|y|z|v>). Use src/data/build_dataset.py for the "
              "fully-synthetic dataset path instead.")
    else:
        print(df["label"].value_counts())
        df.to_csv("data/relabeled/features_raw.csv", index=False)
        print("Saved -> data/relabeled/features_raw.csv")
