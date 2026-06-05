"""
retrain.py
Posture Correction Coach — Classifier Retraining
Uses all groupmate webcam CSV files in the CV/ folder

HOW TO USE:
1. Put all webcam_posture_data_*.csv files in a folder called CV/
   in the same directory as this script
2. Run: python retrain.py
3. Output: posture_classifier_v2.pkl
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ── 1. LOAD ALL CSV FILES ─────────────────────────────────────────
print("=" * 55)
print("POSTURE COACH — CLASSIFIER RETRAINING")
print("=" * 55)

# looks for all CSVs in CV/ subfolder
CSV_DIR = "CV"
csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))

if not csv_files:
    # fallback: look in current folder
    csv_files = glob.glob("webcam_posture_data_*.csv")

if not csv_files:
    raise FileNotFoundError(
        "No CSV files found. Put your webcam_posture_data_*.csv "
        "files in a folder called CV/ next to this script."
    )

print(f"\n[1/4] Loading {len(csv_files)} CSV files...")
all_dfs = []
for f in sorted(csv_files):
    name = os.path.basename(f).replace("webcam_posture_data_","").replace(".csv","")
    df_i = pd.read_csv(f)
    print(f"  {name:<12} {len(df_i):>5} samples")
    all_dfs.append(df_i)

df = pd.concat(all_dfs, ignore_index=True)
print(f"\n  Combined raw total: {len(df)} samples")
print(f"  {df['label'].value_counts().to_string()}")

# ── 2. CLEAN OUTLIERS ─────────────────────────────────────────────
print(f"\n[2/4] Cleaning outliers...")

before = len(df)
df = df[
    (df.neck_angle    >= 0)   &
    (df.neck_angle    <= 90)  &   # physically impossible above 90
    (df.spine_angle   >= 0)   &
    (df.spine_angle   <= 60)  &   # extreme upper bound
    (df.shoulder_tilt >= 0)   &
    (df.shoulder_tilt <= 100) &   # extreme upper bound
    # remove near-zero rows (MediaPipe lost tracking)
    ~((df.neck_angle < 0.5) & (df.spine_angle < 0.5) & (df.shoulder_tilt < 0.5))
].copy()

print(f"  Removed: {before - len(df)} outlier rows")
print(f"  Remaining: {len(df)} samples")
print(f"  {df['label'].value_counts().to_string()}")

# ── 3. AUGMENTATION ───────────────────────────────────────────────
print(f"\n[3/4] Applying Gaussian noise augmentation...")

TARGET    = df['label'].value_counts().max()   # match the largest class
NOISE_STD = 1.5   # degrees — realistic pose variation

augmented = [df]
for label in df['label'].unique():
    class_df = df[df['label'] == label]
    needed   = TARGET - len(class_df)
    if needed <= 0:
        print(f"  {label:<15} no augmentation needed ({len(class_df)} samples)")
        continue

    print(f"  {label:<15} adding {needed} synthetic samples → {TARGET} total")
    sampled = class_df.sample(n=needed, replace=True, random_state=42).copy()
    noise   = np.random.normal(0, NOISE_STD, (needed, 3))
    sampled[['neck_angle', 'spine_angle', 'shoulder_tilt']] += noise
    sampled['neck_angle']    = sampled['neck_angle'].clip(0, 90)
    sampled['spine_angle']   = sampled['spine_angle'].clip(0, 60)
    sampled['shoulder_tilt'] = sampled['shoulder_tilt'].clip(0, 100)
    augmented.append(sampled)

df_aug = pd.concat(augmented, ignore_index=True)
df_aug = df_aug.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n  After augmentation:")
print(f"  {df_aug['label'].value_counts().to_string()}")
print(f"  Total: {len(df_aug)}")

# save augmented dataset for reference
df_aug.to_csv("webcam_posture_combined_augmented.csv", index=False)
print(f"\n  Saved: webcam_posture_combined_augmented.csv")

# ── 4. TRAIN ──────────────────────────────────────────────────────
print(f"\n[4/4] Training Random Forest classifier...")

X = df_aug[['neck_angle', 'spine_angle', 'shoulder_tilt']].values
y = df_aug['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"  Train set: {len(X_train)} samples")
print(f"  Test set:  {len(X_test)} samples")

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
clf.fit(X_train, y_train)

# ── 5. EVALUATE ───────────────────────────────────────────────────
acc = clf.score(X_test, y_test)
preds = clf.predict(X_test)

print(f"\n{'='*55}")
print(f"RESULTS")
print(f"{'='*55}")
print(f"Overall accuracy: {acc:.2%}")
print()
print(classification_report(y_test, preds))

print("Confusion matrix (rows=actual, cols=predicted):")
labels = sorted(df_aug['label'].unique())
cm     = confusion_matrix(y_test, preds, labels=labels)
cm_df  = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df.to_string())
print()

# ── 6. FEATURE IMPORTANCE ─────────────────────────────────────────
importances = clf.feature_importances_
feat_names  = ['neck_angle', 'spine_angle', 'shoulder_tilt']
print("Feature importances:")
for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"  {name:<15} {imp:.3f}  {bar}")

# ── 7. SAVE ───────────────────────────────────────────────────────
joblib.dump(clf, 'posture_classifier_v2.pkl')
print(f"\n{'='*55}")
print(f"Saved: posture_classifier_v2.pkl")
print(f"Copy this to your Pi 5 alongside rag_db_v2/")
print(f"{'='*55}")

# ── 8. DIAGNOSIS IF ACCURACY IS LOW ──────────────────────────────
if acc < 0.80:
    print(f"\n⚠  ACCURACY IS BELOW 80% — diagnosis:")
    for label in labels:
        mask    = y_test == label
        correct = (preds[mask] == label).sum()
        total   = mask.sum()
        recall  = correct / total
        if recall < 0.70:
            print(f"  {label}: only {recall:.0%} recall — "
                  f"re-collect this posture more exaggerated")
    print()
    print("  Most common fix for slouching confusion:")
    sl = df[df['label']=='slouching']
    low = (sl.spine_angle < 3).sum()
    print(f"  {low}/{len(sl)} slouching samples have spine_angle < 3°")
    print(f"  → Groupmates did not slouch enough during collection")
    print(f"  → Re-collect slouching: hunch fully, round shoulders,")
    print(f"    drop chin toward chest — make it very obvious")
