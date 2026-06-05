# G2B Posture Correction Coach — Technical Redesign & Implementation Plan

**Project:** Posture Correction Coach
**Group:** G2B (formerly Medusa Systems — all references renamed)
**Target Device:** Raspberry Pi 5
**Author of this plan:** Lead AI engineering review
**Status:** Pre-implementation — proposed redesign

---

## TL;DR — Top 5 Conclusions Before You Read 50 Pages

1. **Your CV system isn't broken because of the model. It's broken because `spine_angle` is the wrong feature for `slouching`.** Slouching is *thoracic kyphosis* (upper back rounding), not whole-torso lean. Your current feature measures whole-torso lean. The classes are nearly identical on this feature because *they really are* — the feature does not encode the visual difference between them. No amount of Random Forest tuning fixes this.

2. **The single highest-leverage change in the whole project is adding ear landmarks (7, 8) and computing `head_forward_offset` and the `craniovertebral_angle`.** This is the clinical gold standard for forward head posture and it cleanly separates `neck_forward` from everything else.

3. **You do not need to recollect a dataset. You need to (a) compute the new feature set on your existing landmark dumps, (b) heuristically relabel using rules, (c) augment via geometric perturbation of landmarks.** Relabel-and-perturb is ~1 day of work and should jump you from 72% to 85–92%.

4. **LightGBM + a thin rule overlay + EMA temporal smoothing** is the right model stack for Pi 5. Random Forest is fine but LightGBM is ~3× smaller and ~2× faster on ARM for the same accuracy on tabular features. The temporal smoothing matters more than the model choice.

5. **Make the RAG *posture-conditioned*, not posture-aware.** The chatbot should receive a structured `PostureState` object on every query and the retriever should filter chunks by `posture_class` metadata. Generic chunks are retrieval poison when a specific class is active.

---

## Part 1 — Root-Cause Analysis of Current CV Failures

### 1.1 The feature set is fundamentally too weak

Your three features (`neck_angle`, `spine_angle`, `shoulder_tilt`) collectively measure: head tilt vs vertical, torso tilt vs vertical, and shoulder line tilt. **None of them encode the dominant visual signature of slouching, which is rounded upper back combined with forward shoulder roll.**

Concretely:

- `spine_angle` is computed from the shoulder midpoint → hip midpoint line vs vertical. When a seated person slouches, **their hips stay put and their upper back rounds forward**. The shoulder midpoint moves slightly forward and slightly down — but the *angle* of the shoulder-mid → hip-mid line changes by ~1–3°, which is exactly what your data shows (1.56° vs 2.52°). The feature is geometrically blind to kyphosis.
- `neck_angle` is computed from nose → shoulder midpoint. This catches *some* forward head displacement but is corrupted by where the user is looking. Tilting your head down to read your phone produces a large `neck_angle` change with *no* posture change.
- `shoulder_tilt` is the only feature that actually separates `lean` from everything else, and it does so well. That class is probably your best-performing one — verify this in the confusion matrix.

**Diagnosis:** Two of your three features have low signal for two of your four classes. RF cannot recover information that was never extracted from the landmarks.

### 1.2 You are throwing away ~85% of MediaPipe's information

MediaPipe Pose emits 33 landmarks, each with `(x, y, z, visibility)`. You are using **5 landmarks × 2 coordinates = 10 numbers** out of `33 × 4 = 132`. Specifically you are discarding:

- **Ears (7, 8):** The clinical reference point for forward head posture. The craniovertebral angle (tragus → C7 vs horizontal) is the standard measurement in physiotherapy.
- **z-coordinates everywhere:** MediaPipe gives *relative depth*. The z-distance between shoulders and hips is *the* signal for forward shoulder roll, which is the slouching signature.
- **Elbows (13, 14):** When shoulders roll forward, elbows track forward. A useful secondary signal.
- **Visibility scores:** You aren't filtering low-confidence landmarks. If the camera can't see an ear due to head tilt, you should know that and drop those features rather than feed garbage to the model.

### 1.3 Class definitions overlap on your features but not visually

Your class definitions are clinically distinct, but on the three current features:

| | neck_angle | spine_angle | shoulder_tilt |
|---|---|---|---|
| correct | low | low | low |
| slouching | low–mid | low–mid (per your data) | low |
| neck_forward | mid–high | low | low |
| lean | varies | low | high |

`correct_posture` and `slouching` are almost coincident in feature space. `neck_forward` and `slouching` overlap because slouching usually drags the head forward as a secondary effect, so `neck_angle` rises in both. The classifier is being asked to draw a decision boundary through noise.

### 1.4 The model choice is *not* the primary problem, but RF is suboptimal here

Random Forest is a reasonable default but with only 3 weakly-discriminative features on a few-hundred-sample dataset, you've hit the ceiling that the features allow. Switching to XGBoost or LightGBM on the same features will yield maybe +1–3% — not enough. **Solve features first, model second.** That said, when you do switch features, LightGBM will give you another 2–4% on top.

### 1.5 The data collection process is creating significant label noise

Three sources of noise are visible from your description:

- **Contributors didn't exaggerate.** "Slouching" sessions look like mild slouching, so the *labels* say slouching but the *landmarks* describe near-correct posture. This is direct label noise.
- **Webcam angle varies between contributors.** A frontal camera vs a 30° offset camera produces totally different x-coordinates for the same posture. You probably have no normalization for camera placement.
- **No inter-rater reliability check.** No one verified that what contributor A labelled as "slouching" matches what contributor B labelled as "slouching."

**Net diagnosis:** ~15–25% of your slouching samples are probably better described as correct or neck_forward.

---

## Part 2 — Recommended MediaPipe Landmark Strategy

Use these landmarks (all from MediaPipe Pose, 33-landmark model):

| Index | Landmark | Purpose | Critical for |
|---|---|---|---|
| 0 | nose | head reference, midline | lean, neck_forward |
| 7 | left_ear | forward head measurement | **neck_forward** |
| 8 | right_ear | forward head measurement | **neck_forward** |
| 11 | left_shoulder | torso top, roll detection | slouching, lean |
| 12 | right_shoulder | torso top, roll detection | slouching, lean |
| 13 | left_elbow | secondary slouching signal | slouching |
| 14 | right_elbow | secondary slouching signal | slouching |
| 23 | left_hip | torso bottom anchor | all |
| 24 | right_hip | torso bottom anchor | all |

**Discarded as redundant or unreliable:** eyes (2, 5), mouth corners (9, 10) — they add noise without adding signal beyond what nose+ears already give. Knees and below are out of frame in most seated webcam setups.

**Use all four coordinates per landmark:** `x`, `y`, `z`, `visibility`. The `z` axis (depth) is your biggest under-used asset.

**Landmark filtering rule:** If `visibility < 0.5` for any of {ears, shoulders, hips}, mark the frame as unreliable and skip classification (just hold the previous label).

### 2.1 Separation power by landmark — ranked

For each class, which landmarks carry the most discriminative information:

- **neck_forward → EARS, then nose.** Ear-shoulder horizontal offset is the gold-standard clinical sign. Without ears, you cannot reliably detect forward head posture from a webcam.
- **slouching → SHOULDERS (especially z), then elbows, then hips.** The signature is forward shoulder roll, which lives almost entirely in the z-axis.
- **lean → SHOULDERS + HIPS (the line between their midpoints).** Both lines tilt in the same direction during a lean.
- **correct_posture → all of the above with low magnitude.** It is the "everything is small" class.

---

## Part 3 — Recommended Feature Engineering Strategy

### 3.1 The 14 features I recommend

Computed after the normalization step (see §3.2). All distances are normalized by shoulder width, so they become **camera-distance invariant**. All angles are already scale-invariant.

| # | Feature | Formula (informal) | Primary class |
|---|---|---|---|
| 1 | `ear_shoulder_offset_x` | `mean(ear_x) - mean(shoulder_x)` / shoulder_width | **neck_forward** |
| 2 | `craniovertebral_angle` | angle( mean(ear) → mean(shoulder) ) vs vertical | **neck_forward** |
| 3 | `head_forward_offset_z` | `mean(ear_z) - mean(shoulder_z)` | **neck_forward** |
| 4 | `nose_shoulder_offset_x` | `nose_x - mean(shoulder_x)` / shoulder_width | neck_forward, lean |
| 5 | `shoulder_roll_z` | `mean(shoulder_z) - mean(hip_z)` | **slouching** |
| 6 | `torso_compression_ratio` | `\|mean(shoulder_y) - mean(hip_y)\|` / shoulder_width | **slouching** |
| 7 | `elbow_forward_offset_z` | `mean(elbow_z) - mean(shoulder_z)` | slouching (secondary) |
| 8 | `spine_angle_3d` | angle( shoulder_mid → hip_mid ) in xz-plane vs vertical | slouching, lean |
| 9 | `shoulder_tilt_angle` | angle( left_shoulder → right_shoulder ) vs horizontal | **lean** |
| 10 | `hip_tilt_angle` | angle( left_hip → right_hip ) vs horizontal | **lean** |
| 11 | `midline_deviation_angle` | angle( shoulder_mid → hip_mid ) in xy-plane vs vertical | **lean** |
| 12 | `nose_centerline_offset_x` | `nose_x - shoulder_mid_x` / shoulder_width | lean |
| 13 | `lateral_asymmetry_index` | `\|left_ear_to_left_shoulder\| - \|right_ear_to_right_shoulder\|` | lean |
| 14 | `landmark_confidence_mean` | mean visibility of all 9 used landmarks | quality gate |

### 3.2 Normalization step (mandatory, do this before features)

```python
def normalize_landmarks(landmarks):
    # 1. Center origin at hip midpoint
    hip_mid = 0.5 * (landmarks[23] + landmarks[24])
    centered = landmarks - hip_mid

    # 2. Scale by shoulder width (in xy plane, ignore z for scale)
    shoulder_width = np.linalg.norm(centered[11][:2] - centered[12][:2])
    if shoulder_width < 1e-6:
        return None  # invalid frame
    scaled = centered / shoulder_width

    # 3. Optionally rotate so shoulder line is horizontal -- DO NOT DO THIS
    #    for `lean` detection, since you'd destroy the signal. Keep raw orientation.
    return scaled
```

**Why no rotation alignment:** rotating so the shoulder line is horizontal destroys the `shoulder_tilt_angle` signal, which is precisely the thing you need for `lean`. Keep the world frame.

### 3.3 How each feature separates the classes

| Feature | correct | slouching | neck_forward | lean |
|---|---|---|---|---|
| `ear_shoulder_offset_x` | ~0 | small + | **large +** | small |
| `craniovertebral_angle` | ~90° | ~75–85° | **<70°** | ~85° |
| `shoulder_roll_z` | ~0 | **strongly +** | mild + | ~0 |
| `torso_compression_ratio` | ~1.6 | **<1.4** | ~1.5 | ~1.6 |
| `shoulder_tilt_angle` | ~0° | ~0° | ~0° | **>5°** |
| `midline_deviation_angle` | ~0° | <3° | <3° | **>5°** |
| `nose_centerline_offset_x` | ~0 | ~0 | ~0 | **non-zero** |

Each class now has at least two features where it is clearly distinguishable from the others. This is what you didn't have before.

### 3.4 Feature ranking (most useful first)

1. `ear_shoulder_offset_x` — single highest-leverage feature in the whole set
2. `shoulder_roll_z` — the slouching feature you've been missing
3. `craniovertebral_angle` — robust version of #1
4. `torso_compression_ratio` — secondary slouching signal
5. `shoulder_tilt_angle` — lean
6. `midline_deviation_angle` — lean
7. `head_forward_offset_z` — neck_forward in depth axis
8. `hip_tilt_angle` — lean confirmation
9. `nose_centerline_offset_x` — lean / head asymmetry
10. `elbow_forward_offset_z` — slouching secondary
11. `spine_angle_3d` — your old spine_angle but with z, useful again
12. `lateral_asymmetry_index` — composite lean
13. `nose_shoulder_offset_x` — redundant with ear offset, keep as backup
14. `landmark_confidence_mean` — quality gate, not for classification but for rejection

---

## Part 4 — Recommended Dataset Improvement Strategy

**You do not need a full recollection.** You need to fix labels and augment, in this order:

### 4.1 Heuristic relabeling (1–2 hours)

Recompute the new 14 features on every landmark sample you've already collected. Then apply this rule engine to generate clean labels:

```python
def heuristic_label(features):
    # Strong lean signals
    if (abs(features['shoulder_tilt_angle']) > 6 or
        abs(features['midline_deviation_angle']) > 6):
        return 'lean'

    # Strong forward head signals
    if (features['ear_shoulder_offset_x'] > 0.35 or
        features['craniovertebral_angle'] < 72):
        # Distinguish neck_forward (head forward, torso ok)
        # from slouching (head forward AS PART OF whole upper body collapse)
        if (features['shoulder_roll_z'] > 0.15 or
            features['torso_compression_ratio'] < 1.45):
            return 'slouching'
        else:
            return 'neck_forward'

    # Strong slouching without forward head (rarer but exists)
    if (features['shoulder_roll_z'] > 0.20 or
        features['torso_compression_ratio'] < 1.40):
        return 'slouching'

    # Default
    return 'correct_posture'
```

Then compare to original human labels. Where they disagree, **trust the heuristic** if it's a clean signal, **drop the sample** if it's ambiguous (multiple thresholds borderline). Expect to drop 10–20% of samples — that's fine, you're trading quantity for quality.

Tune thresholds on a small held-out set of 30–50 manually verified samples per class.

### 4.2 Geometric perturbation augmentation (2–3 hours)

For each clean sample, generate 5–10 augmented variants by applying small landmark perturbations that *preserve the class*:

```python
def augment_landmarks(landmarks, label, n=8):
    augmented = []
    for _ in range(n):
        aug = landmarks.copy()
        # Small global noise
        aug += np.random.normal(0, 0.005, aug.shape)
        # Random small rotation of body (simulates camera angle change)
        theta = np.random.uniform(-0.05, 0.05)  # ~3°
        R = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])
        aug[:, :3] = aug[:, :3] @ R.T
        augmented.append(aug)
    return augmented
```

### 4.3 Synthetic class-exaggeration (3–4 hours)

This is where you get the **exaggerated slouching samples you currently lack**. Take correct-posture samples and *mathematically deform* them into exaggerated class examples:

```python
def synthesize_slouching(correct_landmarks, severity=0.5):
    out = correct_landmarks.copy()
    # Shoulders forward in z
    for idx in [11, 12]:
        out[idx, 2] += severity * 0.20  # forward in z
        out[idx, 1] += severity * 0.05  # slightly down in y
    # Head follows (slouching drags head forward as well)
    for idx in [0, 7, 8]:
        out[idx, 2] += severity * 0.15
    # Elbows track forward
    for idx in [13, 14]:
        out[idx, 2] += severity * 0.12
    return out

def synthesize_neck_forward(correct_landmarks, severity=0.5):
    out = correct_landmarks.copy()
    # Head forward, shoulders stay
    for idx in [0, 7, 8]:
        out[idx, 2] += severity * 0.25
        out[idx, 0] += np.random.uniform(-0.02, 0.02)  # small lateral noise
    return out

def synthesize_lean(correct_landmarks, severity=0.5, direction='left'):
    out = correct_landmarks.copy()
    sign = -1 if direction == 'left' else 1
    # Tilt the whole upper body
    angle = sign * severity * 0.15  # ~8°
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])
    for idx in [0, 7, 8, 11, 12, 13, 14]:
        out[idx, :3] = out[idx, :3] @ R.T
    return out
```

Generate ~200 synthetic samples per class from your correct-posture pool with severity drawn from `Uniform(0.3, 1.0)`. This single step replaces the "everyone needs to record exaggerated slouching" requirement.

### 4.4 Semi-supervised bootstrap (optional, after the above)

Train a model on the heuristic-labeled + synthetic dataset. Then run it on any *unlabeled* webcam frames you have lying around. Keep predictions with confidence > 0.85 as pseudo-labels. Retrain. Modest gain (1–2%), only worth doing if you have unlabeled data.

### 4.5 What about public datasets?

Honest answer: **mostly not worth it.** MPII, COCO-Pose, and Human3.6M have pose landmarks but no posture-quality labels. Re-labeling them is roughly the same effort as augmenting your own data. **Skip public datasets.** The synthetic augmentation is higher leverage.

---

## Part 5 — Recommended Model Strategy

### 5.1 Ranking for this use case

| Model | Accuracy expectation | Pi 5 latency | Memory | Verdict |
|---|---|---|---|---|
| Random Forest (current) | 85–88% on new features | ~3 ms | ~10 MB | Fine baseline, keep for comparison |
| **LightGBM** | **88–92%** | **~1 ms** | **~3 MB** | **Recommended primary** |
| XGBoost | 88–91% | ~2 ms | ~5 MB | Slightly worse than LGBM on ARM |
| SVM (RBF) | 85–89% | ~5 ms | ~20 MB | OK but scales badly with data |
| Logistic Regression | 78–84% | <1 ms | <1 MB | Use as sanity check only |
| MLP (small) | 85–90% | ~2 ms | ~5 MB | Overkill for tabular; no advantage |
| **Hybrid Rule + ML** | **89–93%** | **~2 ms** | **~5 MB** | **Recommended with LGBM** |
| Temporal models (LSTM/TCN) | 90–94% | ~10 ms | ~20 MB | Overkill — do temporal smoothing post-hoc instead |

### 5.2 Recommendation: LightGBM + thin rule overlay

```python
class PostureClassifier:
    def __init__(self, lgbm_model, rule_weights=0.3):
        self.lgbm = lgbm_model
        self.rule_weight = rule_weights

    def predict_proba(self, features_dict):
        # ML probabilities
        ml_probs = self.lgbm.predict_proba([list(features_dict.values())])[0]
        # Rule probabilities (soft version of §4.1 thresholds)
        rule_probs = self._rule_probs(features_dict)
        # Fuse
        return (1 - self.rule_weight) * ml_probs + self.rule_weight * rule_probs

    def _rule_probs(self, f):
        # Soft scores in [0, 1] per class
        lean_score = sigmoid((abs(f['shoulder_tilt_angle']) - 4) / 2) * 0.5 + \
                     sigmoid((abs(f['midline_deviation_angle']) - 4) / 2) * 0.5
        neck_score = sigmoid((f['ear_shoulder_offset_x'] - 0.30) / 0.10)
        slouch_score = sigmoid((f['shoulder_roll_z'] - 0.12) / 0.06) * 0.6 + \
                       sigmoid((1.45 - f['torso_compression_ratio']) / 0.10) * 0.4
        correct_score = 1.0 - max(lean_score, neck_score, slouch_score)
        probs = np.array([correct_score, slouch_score, neck_score, lean_score])
        return probs / probs.sum()
```

The rule overlay catches edge cases LGBM gets wrong (especially exaggerated single-class instances at distribution edges) and gives the system interpretability — when it says "slouching," you can show the user *which threshold tripped*.

### 5.3 LightGBM config for this dataset

```python
params = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'num_leaves': 15,        # small dataset, prevent overfit
    'max_depth': 5,
    'learning_rate': 0.05,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 3,
    'min_data_in_leaf': 10,
    'lambda_l2': 0.1,
    'verbose': -1,
    'n_jobs': 2,             # Pi 5 has 4 cores; leave 2 for MediaPipe
}
```

Train with early stopping on 5-fold stratified CV. Should converge in 50–150 rounds.

---

## Part 6 — Recommended Temporal Smoothing Strategy

Frame-by-frame classification at 10–15 fps on Pi 5 will flicker on borderline cases. The fix is a two-stage smoother: EMA on probabilities + hysteresis on the final label.

```python
class TemporalSmoother:
    def __init__(self, alpha=0.3, hysteresis_frames=8, classes=None):
        self.alpha = alpha
        self.smoothed = None
        self.hysteresis = hysteresis_frames
        self.current_label = 'correct_posture'
        self.candidate_label = None
        self.candidate_streak = 0
        self.classes = classes or ['correct_posture', 'slouching',
                                   'neck_forward', 'lean']

    def update(self, raw_probs):
        # 1. EMA on probabilities
        if self.smoothed is None:
            self.smoothed = raw_probs.copy()
        else:
            self.smoothed = self.alpha * raw_probs + (1 - self.alpha) * self.smoothed

        # 2. Argmax with hysteresis
        argmax_label = self.classes[np.argmax(self.smoothed)]
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

        return self.current_label, self.smoothed
```

**Parameters explained:**
- `alpha = 0.3` — at 15 fps, the effective time-constant is ~3 frames ≈ 0.2 s. Fast enough to feel responsive, slow enough to kill single-frame noise.
- `hysteresis_frames = 8` — at 15 fps that's ~0.5 s of sustained different-class evidence before the label switches. Eliminates flicker on the boundary between classes.

**Do not use:**
- Majority voting over a sliding window: discrete, loses confidence information.
- Median filtering: only relevant for continuous-valued outputs, not class labels.
- Full LSTM/TCN: heavy and unnecessary when EMA + hysteresis gets you 95% of the way there.

---

## Part 7 — Recommended Full CV Architecture

```
Webcam frame (15 fps)
    │
    ▼
┌───────────────────────┐
│ MediaPipe Pose Lite   │   ~70 ms on Pi 5
│ (33 landmarks)        │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│ Landmark filter       │   Drop frame if visibility on
│ (visibility >= 0.5    │   ears/shoulders/hips < 0.5
│  on key landmarks)    │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│ Normalization         │   Origin = hip midpoint
│ (origin shift,        │   Scale = shoulder width
│  shoulder-width scale)│   No rotation alignment
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│ Feature extraction    │   14 features (Part 3)
│                       │   <1 ms
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│ Rule engine           │   4 soft scores from clinical
│ (soft probabilities)  │   thresholds
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│ LightGBM classifier   │   ML probabilities, ~1 ms
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│ Confidence fusion     │   0.7·ML + 0.3·rules
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│ Temporal smoother     │   EMA + hysteresis (Part 6)
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│ PostureState builder  │   Assemble structured object
│                       │   (Part 10)
└───────┬───────────────┘
        │
        ▼
   PostureState ──→ RAG / Chat / UI
```

End-to-end latency budget on Pi 5: ~75–80 ms per frame, giving comfortable 12–15 fps headroom.

---

## Part 8 — Recommended CV → RAG Integration Architecture

The RAG must be **posture-conditioned at three points**:

1. **Retrieval filter:** Document chunks are tagged with metadata `applicable_postures: [slouching, neck_forward, ...]`. The retriever filters by `current_posture` before semantic search.
2. **Query rewriting:** The user query is augmented with posture-aware context before embedding. E.g. user asks "Am I sitting right?" → expanded query embedded as "Am I sitting right? Currently classified slouching with forward shoulder roll."
3. **Prompt context:** The LLM prompt includes the structured PostureState so the response references actual observed values.

### 8.1 Document tagging (one-time work)

Re-chunk your physiotherapy textbook and tag every chunk with applicable posture classes. Roughly:

```json
{
  "chunk_id": "ergo_042",
  "text": "Forward head posture, in which the head protrudes anteriorly...",
  "applicable_postures": ["neck_forward", "slouching"],
  "topic": "forward_head_posture",
  "anatomical_region": "cervical_spine",
  "intervention_type": "education"
}
```

Tag in batches with an LLM in advance (one-time job, takes ~30 min on Groq), then human-review the top-50 most-retrieved chunks for correctness.

### 8.2 Retrieval pipeline

```
User query
    │
    ▼
┌─────────────────────────┐
│ Read current PostureState│
└───────┬─────────────────┘
        ▼
┌─────────────────────────┐
│ Query expansion         │   "{user_query} | posture={class}
│                         │    indicators={key_features}"
└───────┬─────────────────┘
        ▼
┌─────────────────────────┐
│ Embed expanded query    │   BGE-small (local) or
│                         │   bge-m3 via API
└───────┬─────────────────┘
        ▼
┌─────────────────────────┐
│ ChromaDB filter+search  │   where: applicable_postures
│                         │          contains current_class
│                         │   top-k = 20
└───────┬─────────────────┘
        ▼
┌─────────────────────────┐
│ BM25 keyword search     │   over same posture-filtered set
│                         │   top-k = 20
└───────┬─────────────────┘
        ▼
┌─────────────────────────┐
│ RRF fusion              │   k=60
└───────┬─────────────────┘
        ▼
┌─────────────────────────┐
│ Reranker                │   bge-reranker-v2-m3 via API
│ (top-5 from top-20)     │   (Pi 5 can't run locally fast)
└───────┬─────────────────┘
        ▼
┌─────────────────────────┐
│ Build LLM prompt        │   System + PostureState +
│                         │   retrieved chunks + user query
└───────┬─────────────────┘
        ▼
┌─────────────────────────┐
│ Groq Llama 3.1-8B       │   ~500 ms
└───────┬─────────────────┘
        ▼
   Grounded response
```

### 8.3 Prompt template

```
You are a posture correction coach. You give specific, evidence-based
guidance grounded ONLY in the retrieved physiotherapy references.

=== CURRENT POSTURE OBSERVATION ===
The user's camera currently shows:
- Classified posture: {posture_class}
- Confidence: {confidence:.2f}
- Time in this posture: {posture_duration_sec:.0f} seconds
- Session distribution: {posture_distribution}

Key measured indicators:
- Forward head offset: {ear_shoulder_offset_x:.2f} (normal: <0.20)
- Craniovertebral angle: {craniovertebral_angle:.0f}° (normal: ~85°)
- Shoulder forward roll: {shoulder_roll_z:.2f} (normal: ~0)
- Torso compression: {torso_compression_ratio:.2f} (normal: ~1.6)
- Shoulder tilt: {shoulder_tilt_angle:.1f}° (normal: <3°)
- Midline deviation: {midline_deviation_angle:.1f}° (normal: <3°)

=== RETRIEVED REFERENCES ===
{retrieved_chunks}

=== USER QUESTION ===
{user_question}

=== INSTRUCTIONS ===
- Refer to the specific measured values when explaining the classification.
- Quote or paraphrase from the retrieved references for any
  physiological or corrective claim.
- If the references do not address the user's question, say so rather
  than improvising.
- Be concise (3–6 sentences) unless the user asks for detail.
```

---

## Part 9 — Recommended Live Posture Chatbot Architecture

The chatbot is a *consumer* of the PostureState that the CV pipeline produces. They run in separate threads/processes communicating over a shared in-memory state.

### 9.1 Process model

```
┌──────────────────────┐         ┌──────────────────────┐
│  CV Worker (thread)  │         │  Chat Worker (thread)│
│  ──────────────────  │         │  ────────────────────│
│  Captures webcam     │         │  Listens to UI       │
│  Runs pipeline (Pt 7)│         │  On user message:    │
│  Updates state @15Hz │ ◀─────▶ │   1. Snapshot state  │
│                      │  shared │   2. Build prompt    │
│                      │  state  │   3. Retrieve+rerank │
└──────────────────────┘         │   4. Call Groq       │
                                 │   5. Stream response │
                                 └──────────────────────┘
         ▲
         │
         ▼
┌──────────────────────┐
│  Session Logger      │
│  ──────────────────  │
│  Writes PostureState │
│  to a ring buffer    │
│  (last 5 min) and to │
│  SQLite (full session)│
└──────────────────────┘
```

### 9.2 Shared state implementation

Use `threading.Lock` with a single mutable PostureState. The chat worker reads under the lock, the CV worker writes under the lock. Latency is negligible (sub-ms).

```python
class SharedPostureState:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = None
        self._history = collections.deque(maxlen=4500)  # 5 min @ 15fps

    def update(self, state):
        with self._lock:
            self._state = state
            self._history.append(state)

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self._state), list(self._history)
```

### 9.3 Sample question routing

| User question pattern | Handling |
|---|---|
| "Am I sitting correctly?" | Read state, answer based on `posture_class` |
| "Why am I slouching?" | Read state, surface the 2 features that exceeded thresholds, retrieve chunks on slouching causes |
| "What should I adjust first?" | Rank features by deviation magnitude, retrieve chunks on the largest one |
| "Is my neck still too far forward?" | Compare current `ear_shoulder_offset_x` to 30-sec-ago value, report delta |
| Generic ergonomics question | Standard RAG without heavy posture conditioning |

Implement as soft routing: include the question type as a flag in the prompt rather than as hard branches in code. The LLM handles the rest.

---

## Part 10 — Recommended PostureState Object

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal

PostureClass = Literal['correct_posture', 'slouching', 'neck_forward', 'lean']

@dataclass
class PostureState:
    # --- Classification ---
    posture_class: PostureClass
    confidence: float                          # smoothed top-class probability
    class_probabilities: Dict[str, float]      # all 4 classes

    # --- Core measured features ---
    ear_shoulder_offset_x: float
    craniovertebral_angle: float               # degrees
    head_forward_offset_z: float
    shoulder_roll_z: float
    torso_compression_ratio: float
    shoulder_tilt_angle: float                 # degrees
    hip_tilt_angle: float                      # degrees
    midline_deviation_angle: float             # degrees
    nose_centerline_offset_x: float
    spine_angle_3d: float                      # degrees
    elbow_forward_offset_z: float
    lateral_asymmetry_index: float

    # --- Quality ---
    landmark_confidence_mean: float
    is_reliable: bool                          # all key landmarks visible

    # --- Timing ---
    timestamp: datetime
    posture_duration_sec: float                # time in current class
    session_duration_sec: float

    # --- Session aggregates ---
    posture_distribution: Dict[str, float]     # % time per class this session
    correction_events: int                     # count of transitions to correct
    longest_bad_posture_streak_sec: float

    # --- Convenience computed properties ---
    @property
    def feature_deviations(self) -> Dict[str, float]:
        """How far each feature is from the 'correct' threshold."""
        return {
            'forward_head': max(0, self.ear_shoulder_offset_x - 0.20),
            'shoulder_roll': max(0, self.shoulder_roll_z - 0.08),
            'torso_compression': max(0, 1.55 - self.torso_compression_ratio),
            'shoulder_tilt': max(0, abs(self.shoulder_tilt_angle) - 3),
            'midline_deviation': max(0, abs(self.midline_deviation_angle) - 3),
        }

    @property
    def primary_issue(self) -> str:
        """Feature with the largest normalized deviation."""
        devs = self.feature_deviations
        return max(devs, key=devs.get) if any(devs.values()) else 'none'
```

### 10.1 What goes to the RAG vs what stays in memory

| Field | Sent to RAG prompt | Used for retrieval filter | Stored in session log |
|---|:-:|:-:|:-:|
| posture_class | ✓ | ✓ | ✓ |
| confidence | ✓ | | ✓ |
| class_probabilities | | | ✓ |
| ear_shoulder_offset_x | ✓ | | ✓ |
| craniovertebral_angle | ✓ | | ✓ |
| shoulder_roll_z | ✓ | | ✓ |
| torso_compression_ratio | ✓ | | ✓ |
| shoulder_tilt_angle | ✓ | | ✓ |
| midline_deviation_angle | ✓ | | ✓ |
| other features | only if `primary_issue` | | ✓ |
| posture_duration_sec | ✓ | | ✓ |
| posture_distribution | ✓ | | ✓ |
| timestamp | | | ✓ |
| is_reliable | (refuse to answer if false) | | ✓ |

### 10.2 Retrieval filters

ChromaDB `where` clause uses two filters:

```python
where = {
    "$and": [
        {"applicable_postures": {"$contains": state.posture_class}},
        {"anatomical_region": {"$in": _anatomy_for(state.primary_issue)}}
    ]
}
```

`_anatomy_for('forward_head')` returns `['cervical_spine', 'thoracic_spine']`, etc.

### 10.3 Prompt structure principle

The PostureState appears **before** the retrieved chunks in the prompt. This is deliberate: it primes the model to interpret the references in context of the user's actual state, not generically. If the state is `is_reliable=False`, the prompt instructs the model to ask the user to reposition rather than guess.


---

## Part 11 — Recommended RAG Improvements

You already have a solid base (ChromaDB + Groq Llama). Here's what to add, in priority order:

### 11.1 Metadata filtering (highest leverage, ~4 hours of work)

Tag every chunk with:
- `applicable_postures`: list from {`correct_posture`, `slouching`, `neck_forward`, `lean`}
- `anatomical_region`: one of `cervical_spine`, `thoracic_spine`, `lumbar_spine`, `shoulder`, `general`
- `content_type`: one of `definition`, `cause`, `consequence`, `correction`, `exercise`
- `evidence_level`: rough quality tag (`primary_text`, `commentary`, `example`)

Filter on `applicable_postures` at retrieval time. This single change eliminates the majority of irrelevant chunks for any specific-posture question.

### 11.2 Hybrid retrieval (BM25 + dense + RRF)

ChromaDB returns dense semantic matches. Add a parallel BM25 index over the same chunks. Fuse via Reciprocal Rank Fusion. This catches the cases where the user uses textbook keywords ("kyphosis," "C7," "scapular protraction") that don't embed well in a small embedding model.

```python
from rank_bm25 import BM25Okapi

def rrf_fuse(dense_results, sparse_results, k=60, top_k=20):
    scores = {}
    for rank, doc_id in enumerate(dense_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    for rank, doc_id in enumerate(sparse_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)[:top_k]
```

### 11.3 Reranker

`bge-reranker-v2-m3` over top-20 candidates. Don't run it locally on Pi 5 — it's ~600M parameters. Use a hosted endpoint (Cohere Rerank, Jina Rerank, or a Replicate endpoint). This is the second-highest-leverage improvement after metadata filtering.

If hosted reranking is not in your budget, fall back to: keep top-10 from RRF and let the LLM see them all. Modern Llama 3.1-8B can handle the noise reasonably.

### 11.4 Query expansion with PostureState

Already covered in §8.2 — augment the embedded query with current posture context. Implement this *before* you implement reranking; it's cheaper and gets you most of the gain.

### 11.5 Anti-hallucination guardrails

The prompt template in §8.3 already does most of this work ("if the references do not address the user's question, say so"). Add an explicit grounding check:

```python
def grounding_check(response, retrieved_chunks):
    # Quick sanity: does the response reference at least one chunk's
    # key term? If not, flag for fallback.
    chunk_terms = set()
    for c in retrieved_chunks:
        chunk_terms.update(extract_key_nouns(c['text']))
    response_terms = set(extract_key_nouns(response))
    overlap = len(chunk_terms & response_terms)
    return overlap >= 2
```

If the check fails, fall back to a templated answer that uses only the PostureState (no references) and tells the user "I don't have detailed reference material on this specific aspect of your posture."

### 11.6 What NOT to do

- **Don't fine-tune the LLM.** Not worth it for an academic demo. Better embeddings, better retrieval, better prompts get you 95% of the way.
- **Don't switch to a larger model.** Llama 3.1-8B-Instant via Groq is the right pick for latency. Larger models cost you the responsiveness that makes the coach feel live.
- **Don't add a second LLM as a "judge."** You don't need RAGAS-style evaluation in production. Use it offline to tune retrieval; in production it adds latency for no user-visible benefit.

---

## Part 12 — Recommended Raspberry Pi 5 Deployment Architecture

### 12.1 What runs where

| Component | Where | Why |
|---|---|---|
| MediaPipe Pose Lite | **Local (Pi 5)** | Latency-critical, ~70 ms per frame at full speed |
| Feature extraction | **Local** | Microseconds, no reason to be remote |
| LightGBM classifier | **Local** | Microseconds, tiny model |
| Rule engine | **Local** | Pure Python, microseconds |
| Temporal smoother | **Local** | Stateful, frame-rate |
| Embeddings (BGE-small-en-v1.5) | **Local** | ~120 ms on Pi 5 CPU, avoids API round-trip on every query |
| ChromaDB | **Local** | Small KB (~1k chunks), fits easily |
| BM25 index | **Local** | Trivial memory cost |
| Reranker (bge-reranker-v2-m3) | **Remote** (hosted API) | 600M params — too slow locally |
| Groq Llama 3.1-8B | **Remote** (Groq API) | Already remote, ~500 ms is fine |
| Streamlit UI | **Local** | Server runs on Pi, browser anywhere on LAN |
| Session log (SQLite) | **Local** | Privacy + offline tolerance |

### 12.2 Threading / process model

Three threads in one Python process:

```
┌─────────────────────────────────────────────┐
│ Main process (Streamlit server)             │
│                                             │
│  ┌──────────────────┐  ┌──────────────────┐ │
│  │ CV thread        │  │ Chat thread      │ │
│  │ (daemon=True)    │  │ (daemon=True)    │ │
│  │                  │  │                  │ │
│  │ while True:      │  │ on user_msg:     │ │
│  │   frame = cap.read()│  read state    │ │
│  │   state = pipeline()│  retrieve       │ │
│  │   shared.update(state)│  call Groq    │ │
│  │   sleep(~67 ms)  │  │   stream reply   │ │
│  └──────────────────┘  └──────────────────┘ │
│                                             │
│  Streamlit UI thread (main)                 │
│  - displays current PostureState            │
│  - shows chat                               │
│  - polls shared state at 5 Hz for UI update │
└─────────────────────────────────────────────┘
```

Use `cv2.VideoCapture` with a small buffer to avoid latency buildup:

```python
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)
```

### 12.3 Latency budget

| Step | Budget | Notes |
|---|---|---|
| Webcam read | 5 ms | with buffer=1 |
| MediaPipe Pose Lite | 65 ms | use Lite (not Heavy) on Pi 5 |
| Feature extraction | 1 ms | numpy on 14 features |
| Classifier (LGBM + rules) | 2 ms | tiny model |
| Temporal smoother | <1 ms | |
| **Total per-frame** | **~73 ms** | gives ~13–14 fps comfortably |

For chat:

| Step | Budget |
|---|---|
| Embed query (BGE-small local) | 120 ms |
| ChromaDB filtered search | 30 ms |
| BM25 search | 20 ms |
| RRF fuse | <1 ms |
| Reranker (hosted) | 250 ms |
| Groq Llama 3.1-8B | 500 ms |
| **Total first-token** | **~900 ms** |

That's fast enough that users will perceive it as responsive. Stream the response token-by-token from Groq.

### 12.4 Memory budget on Pi 5 (4 GB variant)

| Component | RAM |
|---|---|
| Python interpreter | ~50 MB |
| MediaPipe + TFLite | ~180 MB |
| OpenCV | ~80 MB |
| LightGBM model | ~5 MB |
| BGE-small ONNX | ~130 MB |
| ChromaDB (in-memory) | ~80 MB |
| BM25 index | ~10 MB |
| Streamlit | ~150 MB |
| Session ring buffer | ~20 MB |
| Browser (if local) | ~250 MB |
| OS overhead | ~400 MB |
| **Total** | **~1.4 GB** |

Plenty of headroom on a 4 GB Pi 5 — the 8 GB variant is overkill for this project.

### 12.5 Caching strategy

- **Embed once at startup:** all KB chunks. Persist to disk, load on next start.
- **Cache top-k retrievals per posture class:** when posture stays the same for >30s, the retrieval results for follow-up questions overlap heavily. Use an LRU cache keyed by `(posture_class, query_hash)`.
- **Don't cache LLM responses:** users will ask the same question and expect the answer to reflect their *current* state, which changes.

### 12.6 Optimization checklist before demo

1. Use MediaPipe Pose **Lite** model, not Full/Heavy. Lite is ~3× faster, accuracy loss on coarse posture features is negligible.
2. Run MediaPipe with `static_image_mode=False` so it uses tracking between frames.
3. Resize webcam to 640×480 — MediaPipe internally scales anyway, no point feeding it 1080p.
4. Process every frame; if you fall behind, MediaPipe self-throttles fine.
5. Set OpenCV `BUFFERSIZE=1` (above) to prevent latency drift.
6. Pre-warm: run a few dummy inferences at startup so JIT/cache is hot when demo starts.

---

## Part 13 — Recommended Fix for MediaPipe / Protobuf Issues

### 13.1 Diagnosis

The Windows MediaPipe/protobuf conflict almost always traces to **protobuf version pinning incompatibilities**:

- `mediapipe >= 0.10.x` is built against `protobuf 4.x` (newer versions).
- Older MediaPipe (0.9.x) needs `protobuf <= 3.20.x`.
- Many other libraries (`google-*`, `tensorflow`, sometimes `chromadb` indirectly) drag in *different* protobuf pins.
- On Windows, pip's resolver often picks compatible-on-paper versions that crash at runtime with errors like:
  - `TypeError: Descriptors cannot not be created directly.`
  - `ImportError: cannot import name 'builder' from 'google.protobuf.internal'`
  - `AttributeError: module 'google.protobuf.descriptor' has no attribute '_internal_create_key'`

Secondary cause: NumPy 2.0 broke binary compatibility with anything built against NumPy 1.x, including MediaPipe wheels. If you `pip install` after May 2024 without pinning, you may get NumPy 2.x and silent breakage.

### 13.2 Cleanest fix — exact pin set

Create a fresh venv (do not reuse anaconda base):

```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Then install with these exact pins:

```
# requirements.txt
python_requires = ">=3.11,<3.12"

mediapipe==0.10.14
protobuf==4.25.3
numpy==1.26.4
opencv-python==4.10.0.84

lightgbm==4.3.0
scikit-learn==1.5.0
pandas==2.2.2

chromadb==0.5.5
rank-bm25==0.2.2
sentence-transformers==3.0.1

streamlit==1.36.0
groq==0.9.0
python-dotenv==1.0.1
```

Pip install with no resolver shortcuts:

```bash
pip install --no-cache-dir -r requirements.txt
```

### 13.3 Best Python version

**Python 3.11.9** is the sweet spot:
- MediaPipe 0.10.x has stable wheels.
- ChromaDB 0.5.x supports it cleanly.
- Most ML libraries have ARM64 wheels for 3.11 (important for Pi 5).
- 3.12 is risky — MediaPipe wheels lag and several libs have edge-case bugs.
- 3.10 works but lacks some perf improvements in 3.11.

Avoid 3.13 entirely until 2027.

### 13.4 Should development continue on Windows or move to Pi 5?

**Develop on Windows for fast iteration, test on Pi 5 weekly. Do not wait until the end to test on Pi 5.**

Reasons:
- Windows dev gives you VS Code, fast model retraining (your laptop is faster than the Pi for LGBM training), and easier debugging.
- The Pi 5 has different MediaPipe build internals (delegates to XNNPACK, different threading). Pure Windows development will have *nasty* surprises at the end (frame rate, threading bugs, file path differences, OpenCV camera backend differences).
- Set up a weekly "Pi parity check": run the full pipeline end-to-end on the Pi 5 with the latest code, log FPS and accuracy.

Concrete schedule recommendation:
1. **Week 1–2:** Windows-only. Build features, classifier, augmentation. Hit ≥85% accuracy on heuristic-relabeled data.
2. **Week 3:** First Pi 5 deployment. Install dependencies, run pipeline, measure FPS. Profile bottlenecks.
3. **Week 4:** Integrate RAG. Test full end-to-end on Pi 5.
4. **Week 5:** Polish, demo prep.

If you don't do step 2 by Week 3 you will pay for it in the last week.

---

## Part 14 — Exact Codebase Changes Required

Below is the minimum set of file-level changes to bring the system to the redesigned architecture. File paths are suggested; adapt to your existing layout.

```
g2b_posture_coach/
├── requirements.txt                    [REPLACE — see §13.2]
├── README.md                           [UPDATE — rename Medusa Systems → G2B]
├── data/
│   ├── raw_landmarks/                  [existing]
│   ├── heuristic_relabeled/            [NEW — output of relabeling]
│   └── augmented/                      [NEW — synthetic + perturbed]
├── src/
│   ├── cv/
│   │   ├── pose_extractor.py           [UPDATE — use 9 landmarks, keep z]
│   │   ├── normalizer.py               [NEW — §3.2 normalization]
│   │   ├── features.py                 [REWRITE — 14 features from §3.1]
│   │   ├── rule_engine.py              [NEW — §5.2 soft scores]
│   │   ├── classifier.py               [REWRITE — LightGBM + fusion]
│   │   ├── smoother.py                 [NEW — §6 temporal smoother]
│   │   └── pipeline.py                 [NEW — orchestrates Part 7]
│   ├── data/
│   │   ├── relabel.py                  [NEW — §4.1 heuristic relabel]
│   │   ├── augment.py                  [NEW — §4.2 perturbation]
│   │   ├── synthesize.py               [NEW — §4.3 class exaggeration]
│   │   └── train.py                    [UPDATE — train LGBM with new data]
│   ├── state/
│   │   └── posture_state.py            [NEW — dataclass from Part 10]
│   ├── rag/
│   │   ├── chunker.py                  [UPDATE — emit metadata tags]
│   │   ├── tagger.py                   [NEW — LLM-tag chunks by class]
│   │   ├── embedder.py                 [UPDATE — BGE-small local]
│   │   ├── retriever.py                [REWRITE — hybrid + filter + rerank]
│   │   ├── prompt_builder.py           [NEW — §8.3 template]
│   │   └── llm_client.py               [UPDATE — Groq streaming]
│   ├── app/
│   │   ├── streamlit_app.py            [UPDATE — show PostureState live]
│   │   └── shared_state.py             [NEW — thread-safe state]
│   └── workers/
│       ├── cv_worker.py                [NEW — runs pipeline in thread]
│       └── chat_worker.py              [NEW — handles user messages]
├── models/
│   ├── lgbm_posture_v2.txt             [NEW — trained model]
│   └── bge-small-en-v1.5/              [NEW — local embeddings]
├── notebooks/
│   ├── 01_relabel_existing_data.ipynb  [NEW]
│   ├── 02_augment_and_synthesize.ipynb [NEW]
│   ├── 03_train_lgbm.ipynb             [NEW]
│   └── 04_evaluate.ipynb               [NEW]
└── tests/
    ├── test_features.py                [NEW]
    ├── test_classifier.py              [NEW]
    └── test_pipeline.py                [NEW]
```

### Key new files — minimum viable skeleton

**`src/cv/features.py`** — implements all 14 features from §3.1. Pure function `extract_features(normalized_landmarks: np.ndarray) -> Dict[str, float]`.

**`src/cv/classifier.py`** —

```python
import lightgbm as lgb
import numpy as np
from .rule_engine import rule_probs

class PostureClassifier:
    CLASSES = ['correct_posture', 'slouching', 'neck_forward', 'lean']

    def __init__(self, model_path: str, rule_weight: float = 0.3):
        self.model = lgb.Booster(model_file=model_path)
        self.rule_weight = rule_weight

    def predict(self, features: dict):
        x = np.array([list(features.values())])
        ml = self.model.predict(x)[0]
        rl = rule_probs(features)
        fused = (1 - self.rule_weight) * ml + self.rule_weight * rl
        idx = int(np.argmax(fused))
        return self.CLASSES[idx], float(fused[idx]), dict(zip(self.CLASSES, fused.tolist()))
```

**`src/cv/pipeline.py`** —

```python
class PosturePipeline:
    def __init__(self, classifier, smoother):
        self.classifier = classifier
        self.smoother = smoother
        # MediaPipe setup, etc.

    def step(self, frame) -> 'PostureState':
        landmarks = self._extract_landmarks(frame)
        if not self._is_reliable(landmarks):
            return self._hold_last_state()
        normalized = normalize_landmarks(landmarks)
        features = extract_features(normalized)
        label, conf, probs = self.classifier.predict(features)
        smoothed_label, smoothed_probs = self.smoother.update(np.array(list(probs.values())))
        return build_posture_state(smoothed_label, smoothed_probs, features)
```

**`src/state/posture_state.py`** — exactly the dataclass from Part 10.

**`src/rag/retriever.py`** — the hybrid pipeline from §8.2.

### Files to update (not rewrite)

- `src/cv/pose_extractor.py`: extend the landmark indices list, keep all 4 coords per landmark.
- `src/rag/chunker.py`: emit metadata tags alongside text.
- `src/app/streamlit_app.py`: display PostureState live, show primary issue, show feature deviations as gauges.

### Files to remove or archive

- The old 3-feature `features.py` — archive, don't delete (reference for comparison).
- Any "Medusa Systems" branding strings throughout — search-replace to "G2B".

---

## Part 15 — Priority-Ranked Implementation Roadmap

Ordered so each step unblocks the next and delivers visible improvement.

### Sprint 1 — Foundation (Days 1–3)

**Goal: reproduce current system + clean environment.**

1. ☐ Create fresh `.venv` with pinned `requirements.txt` from §13.2.
2. ☐ Verify MediaPipe + OpenCV + Streamlit run end-to-end with current code.
3. ☐ Rename all "Medusa Systems" references to "G2B" (grep + sed).
4. ☐ Set up dev branch in git with the new directory structure.
5. ☐ Confirm current 72% accuracy on existing test set as baseline.

**Done when:** baseline reproduces, no protobuf errors, branch is clean.

### Sprint 2 — New features (Days 4–6)

**Goal: implement the 14 features and prove they separate the classes.**

1. ☐ Implement `src/cv/normalizer.py` — origin shift + shoulder-width scale.
2. ☐ Implement `src/cv/features.py` — all 14 features.
3. ☐ Run on existing landmark dataset, compute per-class feature means/stds.
4. ☐ Visualize feature distributions per class (pair plots). **Verify visually that classes separate** on the key features. If `ear_shoulder_offset_x` doesn't show clean separation between `neck_forward` and others, debug landmark extraction before going further.

**Done when:** plots show the expected per-class separation predicted in §3.3.

### Sprint 3 — Relabel + augment (Days 7–9)

**Goal: clean labels, expanded dataset.**

1. ☐ Implement `src/data/relabel.py` (heuristic rules from §4.1).
2. ☐ Run on all existing samples; drop ambiguous (~10–20%).
3. ☐ Implement `src/data/augment.py` (perturbation) and `synthesize.py` (class exaggeration).
4. ☐ Generate ~200 synthetic + 5× perturbation per real sample.
5. ☐ Final dataset: ~3000–5000 samples, well-balanced across classes.

**Done when:** dataset has ≥500 samples per class, label distribution within 10% of uniform.

### Sprint 4 — Model + smoother (Days 10–12)

**Goal: trained classifier + temporal smoothing.**

1. ☐ Implement `src/data/train.py` — LightGBM with config from §5.3.
2. ☐ 5-fold stratified CV. Target ≥88% mean accuracy.
3. ☐ Implement `src/cv/rule_engine.py` (soft rule probs).
4. ☐ Implement `src/cv/classifier.py` (LGBM + rule fusion).
5. ☐ Implement `src/cv/smoother.py` (EMA + hysteresis).
6. ☐ Implement `src/cv/pipeline.py` orchestrator.
7. ☐ Live test with webcam. Subjectively verify smoothing eliminates flicker.

**Done when:** live demo on Windows feels stable; logged confusion matrix shows clean diagonal.

### Sprint 5 — Pi 5 first deployment (Days 13–14)

**Goal: prove the system runs on Pi 5 before doing more software work.**

1. ☐ Flash Pi 5 with latest Raspberry Pi OS 64-bit (Bookworm).
2. ☐ Install dependencies (same pins, ARM64 wheels). MediaPipe needs `pip install mediapipe`; if it fails, build from source or use `mediapipe-rpi5` community wheel.
3. ☐ Run the full pipeline. Measure FPS. Target ≥10 fps on MediaPipe Lite.
4. ☐ Profile bottlenecks if FPS is bad.
5. ☐ Commit a `pi5_setup.md` document with exact install steps.

**Done when:** Pi 5 hits ≥10 fps and produces correct PostureState live.

### Sprint 6 — RAG redesign (Days 15–18)

**Goal: posture-conditioned RAG with metadata filtering and hybrid retrieval.**

1. ☐ Re-chunk physiotherapy KB; tag each chunk via LLM (§11.1).
2. ☐ Human-review top-50 most likely to be retrieved.
3. ☐ Rebuild ChromaDB with metadata.
4. ☐ Add BM25 index, implement RRF fusion.
5. ☐ Add (hosted) reranker.
6. ☐ Implement prompt builder (§8.3).
7. ☐ Implement grounding check (§11.5).

**Done when:** test 20 hand-crafted questions; each response correctly references the user's measured state.

### Sprint 7 — Live chatbot integration (Days 19–21)

**Goal: chat + CV running together, posture-aware responses live.**

1. ☐ Implement `SharedPostureState`.
2. ☐ Implement `CVWorker` and `ChatWorker` threads.
3. ☐ Update Streamlit app to show live state + chat.
4. ☐ Test concurrent: posture changes while chat is open, ask "what changed?"
5. ☐ End-to-end demo dry-run on Pi 5.

**Done when:** demo flow works: user sits badly → CV detects → ask "am I sitting right?" → posture-grounded answer.

### Sprint 8 — Polish (Days 22–25)

**Goal: production-quality demo.**

1. ☐ Session statistics in UI (time per class, correction events).
2. ☐ Visual overlay on webcam feed showing key landmarks + measured angles.
3. ☐ Mode for capturing demo videos.
4. ☐ Error handling: webcam disconnect, Groq API timeout, low landmark confidence.
5. ☐ Documentation: README, demo script, panel-defense Q&A prep.
6. ☐ Final accuracy report. Target ≥88% on held-out test set.

**Done when:** demo runs unattended for 10 minutes without errors; all panelist FAQs have prepared answers.

---

## Appendix A — Expected Accuracy Trajectory

| Stage | Expected accuracy | Notes |
|---|---|---|
| Current (3 features, RF) | 72.49% | baseline |
| New 14 features, RF | 82–86% | feature change alone |
| + heuristic relabeling | 85–89% | label noise removed |
| + augmentation/synthesis | 87–90% | better class balance |
| + LightGBM | 88–91% | small but real gain |
| + rule overlay fusion | 88–92% | catches edge cases |
| + temporal smoothing | (n/a — quality, not accuracy) | flicker gone |

If you do not reach ≥85% after Sprint 4, the most likely culprit is **landmark extraction failing on certain camera setups**. Re-check Sprint 2 feature distributions on the full dataset; you should see clean per-class clusters on `ear_shoulder_offset_x` and `shoulder_roll_z`. If you don't, fix MediaPipe configuration before changing anything else.

---

## Appendix B — Panel Defense Q&A (Likely Questions)

**Q: Why LightGBM over Random Forest?**
A: Lower latency (~1 ms vs ~3 ms), smaller model (~3 MB vs ~10 MB), marginally better accuracy on tabular features with shallow trees. On Pi 5 the latency margin matters.

**Q: How do you justify dropping samples during heuristic relabeling?**
A: Mislabeled training data caps the classifier's accuracy. We retain ambiguous samples in a separate "review" pool for future relabeling; we don't train on them. We trade quantity for label quality, then restore quantity via geometric augmentation.

**Q: Isn't synthesized data going to bias the model?**
A: We synthesize only the visible posture geometry, not the underlying physiology. The geometric transformations match how MediaPipe sees real slouching (forward z-shift of shoulders, etc.). Real held-out test data remains untouched as the truth. If synthetic-augmented training generalizes to that real test set, the augmentation is valid.

**Q: Why fuse rules with ML instead of one or the other?**
A: Pure rules are brittle to edge cases; pure ML on a small dataset is brittle to distribution shift. Fusion gives us interpretability (we can say "the model flagged slouching primarily because shoulder forward roll exceeded threshold") and robustness.

**Q: How is the RAG actually grounded in the camera?**
A: The PostureState object is injected into both the retrieval filter (so we only retrieve chunks tagged for the current posture class) and the LLM prompt (so the response references actual measured angles). The grounding check verifies the response uses at least 2 key terms from retrieved chunks.

**Q: What happens if MediaPipe fails to detect landmarks?**
A: The pipeline marks the frame as `is_reliable=False`, holds the previous PostureState, and the chatbot is instructed to ask the user to reposition. We never classify on bad landmarks.

**Q: Why not run a full LSTM/transformer for temporal modelling?**
A: EMA + hysteresis solves the flicker problem at ~zero compute cost. An LSTM would add 10× the latency and require sequence-level labels we don't have. The cost-benefit doesn't favour it.

