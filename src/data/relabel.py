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
                 if c not in ("label", "source_file", "source",
                              "heuristic_label", "kept")]
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
    print(f"Saved -> data/relabeled/features_relabeled.csv")
