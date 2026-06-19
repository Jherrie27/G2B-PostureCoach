"""Train the posture classifier on REAL webcam data collected via collect_posture.py.

Unlike build_dataset.py (which filters real samples through a heuristic and tops
up with synthetic data), this trains DIRECTLY on your real labeled frames so the
model matches your actual camera, framing and lighting. That is the real fix for
"the camera labels my posture wrong".

    1. Collect data:  python collect_posture.py <class> --seconds 60   (x4 classes)
    2. Retrain:       python train_real.py
    3. Run the app:   .venv\\Scripts\\python.exe -m streamlit run app.py

It saves models/posture_lgbm_v3.txt (+ feature_order.json), the same paths the
app loads, after backing up the previous model. Reports a REAL held-out accuracy.
"""
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.cv.features import FEATURE_ORDER
from src.data.csv_to_features import csv_to_features_df
from train_lgbm import LGBM_PARAMS, CLASSES, LABEL_TO_IDX

RAW_DIR = Path("data/raw_landmarks/CV")
MODEL_PATH = Path("models/posture_lgbm_v3.txt")
FEATURE_ORDER_PATH = Path("models/feature_order.json")
KEY_FEATURES = ["ear_shoulder_offset_x", "craniovertebral_angle",
                "shoulder_roll_z", "torso_compression_ratio",
                "shoulder_tilt_angle", "midline_deviation_angle"]


def load_real():
    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        raise SystemExit(
            f"No CSVs in {RAW_DIR}. Run collect_posture.py for each class first.")
    print(f"Found {len(csvs)} CSV file(s):")
    for c in csvs:
        print(f"  - {c.name}")
    df = csv_to_features_df(csvs)
    if df.empty:
        raise SystemExit("0 feature rows extracted. Check the CSVs have "
                         "landmark_<idx>_<x|y|z|v> columns.")
    return df


def main():
    df = load_real()
    print("\nClass distribution (real frames):")
    print(df["label"].value_counts().to_string())

    counts = df["label"].value_counts()
    missing = [c for c in CLASSES if c not in counts.index]
    if missing:
        print(f"\nWARNING: no data for {missing}. The model cannot learn these "
              f"classes. Collect them before relying on the model.")
    thin = [c for c in counts.index if counts[c] < 100]
    if thin:
        print(f"WARNING: few samples for {thin} (<100 frames). Consider "
              f"recording longer for these classes.")

    # Per-class feature means: confirm the classes actually separate on YOUR data.
    print("\nPer-class means of key features (sanity check separation):")
    print(df.groupby("label")[KEY_FEATURES].mean().round(3).to_string())

    X = df[FEATURE_ORDER].values.astype(np.float32)
    y = np.array([LABEL_TO_IDX[c] for c in df["label"]])

    present = sorted(set(y))
    if len(present) < 2:
        raise SystemExit("Need at least 2 classes with data to train.")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)

    params = dict(LGBM_PARAMS)
    params["num_class"] = 4  # keep full 4-class head; absent classes just stay low
    train_set = lgb.Dataset(X_tr, y_tr)
    val_set = lgb.Dataset(X_te, y_te, reference=train_set)
    model = lgb.train(
        params, train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    pred = np.argmax(model.predict(X_te), axis=1)
    acc = accuracy_score(y_te, pred)
    print(f"\n=== REAL held-out accuracy: {acc:.4f} ===")
    labels_present = sorted(set(y_te) | set(pred))
    names_present = [CLASSES[i] for i in labels_present]
    print(classification_report(y_te, pred, labels=labels_present,
                                target_names=names_present, digits=4,
                                zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_te, pred, labels=labels_present)
    print(pd.DataFrame(cm, index=names_present, columns=names_present))

    # Back up the old model, then save the new one to the paths the app loads.
    MODEL_PATH.parent.mkdir(exist_ok=True)
    if MODEL_PATH.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = MODEL_PATH.with_name(f"posture_lgbm_v3.synthetic_{ts}.txt.bak")
        shutil.copy2(MODEL_PATH, bak)
        print(f"\nBacked up previous model -> {bak.name}")
    model.save_model(str(MODEL_PATH))
    with open(FEATURE_ORDER_PATH, "w") as f:
        json.dump(FEATURE_ORDER, f, indent=2)
    print(f"Saved real-data model -> {MODEL_PATH}")
    print("Restart the app to use it.")


if __name__ == "__main__":
    main()
