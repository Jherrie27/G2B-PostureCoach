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
    print(f"\nMean CV accuracy: {np.mean(fold_accs):.4f}  +/- {np.std(fold_accs):.4f}")
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
