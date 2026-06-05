"""Build the training set.

G2B fully-synthetic path (chosen because this project has no raw landmark dumps):
  1. Try to load any REAL landmark arrays from data/raw_landmarks/CV (forward-compat;
     yields nothing for the current 3-feature CSVs).
  2. Synthesize a base pool of 'correct' posture landmark arrays.
  3. Generate balanced classes:
       - correct_posture : sampled from the correct pool (+ real, if any)
       - slouching / neck_forward / lean : deformed from the correct pool
  4. Extract the 14 features for every array and write data/augmented/training_set.csv

If real landmark dumps are added later, they are automatically mixed in and the
`source` column distinguishes real / augmented / synthetic.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from src.cv.normalizer import normalize
from src.cv.features import extract_features
from src.data.csv_to_features import _row_to_landmark_array, _infer_label
from src.data.relabel import heuristic_label
from src.data.augment import perturb_landmarks
from src.data.synthesize import (make_correct_pool, synthesize_from_correct)

RNG = np.random.default_rng(42)
TARGET_PER_CLASS = 1200
CORRECT_BASE_POOL = 500          # synthetic 'correct' arrays used as deformation source
AUGS_PER_REAL_SAMPLE = 4
NON_CORRECT = ["slouching", "neck_forward", "lean"]


def collect_real_landmark_arrays_per_class(csv_dir: Path):
    """Returns dict[label] -> list[(9,4) array] from REAL landmark dumps, if present.

    Filters to samples where the heuristic agrees with the file's label.
    Returns empty lists for the current project (no landmark columns in CSVs).
    """
    pool = {"correct_posture": [], "slouching": [],
            "neck_forward": [], "lean": []}
    if not csv_dir.exists():
        return pool
    for path in sorted(csv_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path)
            label = _infer_label(path, df)
        except Exception:
            continue
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
            if heuristic_label(feats) == label:  # keep confident agreement only
                pool[label].append(arr)
    return pool


def _features_row(arr, label, source, rng):
    normed = normalize(arr)
    if normed is None:
        return None
    f = extract_features(normed)
    if f is None:
        return None
    f["label"] = label
    f["source"] = source
    return f


def build():
    real = collect_real_landmark_arrays_per_class(Path("data/raw_landmarks/CV"))
    n_real = {k: len(v) for k, v in real.items()}
    print("Real landmark arrays found:", n_real)

    # Synthetic correct-posture base pool (also doubles as 'correct' class source)
    correct_pool = make_correct_pool(CORRECT_BASE_POOL, RNG)
    print(f"Synthesized correct base pool: {correct_pool.shape[0]} arrays")

    rows = []

    # === correct_posture ===
    # real correct (+ augmentations) first, then top up from the synthetic pool
    for arr in real["correct_posture"]:
        r = _features_row(arr, "correct_posture", "real", RNG)
        if r: rows.append(r)
        for _ in range(AUGS_PER_REAL_SAMPLE):
            r = _features_row(perturb_landmarks(arr, RNG), "correct_posture", "augmented", RNG)
            if r: rows.append(r)
    have = sum(1 for r in rows if r["label"] == "correct_posture")
    idx = 0
    while have < TARGET_PER_CLASS:
        r = _features_row(correct_pool[idx % correct_pool.shape[0]],
                          "correct_posture", "synthetic", RNG)
        idx += 1
        if r:
            rows.append(r); have += 1

    # === slouching / neck_forward / lean ===
    for cls in NON_CORRECT:
        # real (+ augmentations)
        for arr in real[cls]:
            r = _features_row(arr, cls, "real", RNG)
            if r: rows.append(r)
            for _ in range(AUGS_PER_REAL_SAMPLE):
                r = _features_row(perturb_landmarks(arr, RNG), cls, "augmented", RNG)
                if r: rows.append(r)
        have = sum(1 for r in rows if r["label"] == cls)
        want = TARGET_PER_CLASS - have
        if want > 0:
            print(f"Synthesizing {want} '{cls}' samples")
            for arr in synthesize_from_correct(correct_pool, cls, want, RNG):
                r = _features_row(arr, cls, "synthetic", RNG)
                if r: rows.append(r)

    out = pd.DataFrame(rows)
    print("\nFinal class distribution:")
    print(out["label"].value_counts())
    print("\nBy source:")
    print(out.groupby("label")["source"].value_counts())
    Path("data/augmented").mkdir(parents=True, exist_ok=True)
    out.to_csv("data/augmented/training_set.csv", index=False)
    print(f"\nSaved -> data/augmented/training_set.csv ({len(out)} rows)")


if __name__ == "__main__":
    build()
