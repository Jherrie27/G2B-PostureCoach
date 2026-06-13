# G2B Posture Correction Coach — HANDOFF (LATEST)

**Last updated:** 2026-06-06
**GitHub:** https://github.com/Jherrie27/G2B-PostureCoach (branch `main`)
**Supersedes:** `HANDOFF.md` (this file adds the GitHub push, secret handling, the Python 3.13
install fix, the live-demo findings, and the presentation material).

This is the single source of truth. A new engineer or a fresh AI session should be able to pick up
from here without the chat history.

---

## 0. TL;DR

- Rebuilt the Posture Coach from a **3-feature Random Forest (~72%)** to a **14-feature LightGBM +
  rule fusion + temporal smoothing** CV pipeline, plus a **posture-conditioned RAG** chatbot
  (BGE-small + BM25 hybrid retrieval → Groq Llama 3.1-8B, grounded in a physiotherapy textbook).
- **Two facts shaped everything:** (1) **no raw landmark data exists** (old CSVs only stored 3
  scalars) → we trained on a **fully-synthetic dataset**; (2) the project runs in a **Python 3.12
  venv** (ChromaDB was dropped — no installable wheel — and replaced with a NumPy vector store).
- **Reported CV accuracy 0.991 is a synthetic-separability ceiling, NOT real-world.**
- **Shipped to GitHub** with a clean, secret-free history. The Groq key lives only in a local,
  gitignored `.env`.
- **Live demo finding:** if the **hips are not in the camera frame**, the system marks the frame
  `is_reliable=False`, freezes the label, and the chatbot refuses ("can't see your posture"). Fix:
  frame the user **from the waist up**. An optional code relaxation is described in §11.

---

## 1. Current status by phase

| Phase | What | Status |
|---|---|---|
| 1 | Env (3.12 venv, pinned deps, scaffold) | ✅ done |
| 2 | 14-feature CV pipeline + tests | ✅ done |
| 3 | Fully-synthetic dataset (4800 rows) | ✅ done |
| 4 | LightGBM + rule fusion (CV 0.991) | ✅ done |
| 5 | Smoother + PostureState + pipeline | ✅ done |
| 6 | Posture-conditioned RAG (NumPy store) | ✅ done |
| 7 | Workers + Streamlit app | ✅ done (headless-verified; live run works) |
| 8 | Pi 5 artifacts (`pi5_setup.md`, systemd) | ✅ docs done; **not run on Pi hardware** |
| — | GitHub publish + secret hygiene | ✅ done |
| 9/10 | Live accuracy tuning, Pi deploy, demo polish | ⏳ needs hardware / real data |

---

## 2. How to run

From the project root. Venv Python = `.venv\Scripts\python.exe` (Windows). Webcam auto-detects.

**Full app (video + live label + chat):**
```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```
**Headless terminal (posture only, no key needed):**
```powershell
.venv\Scripts\python.exe -u main.py
```
**Automated tests (no camera):**
```powershell
.venv\Scripts\python.exe -m pytest tests/test_features.py tests/test_smoother.py tests/test_state.py -q
```

---

## 3. Fresh-clone setup (for teammates)

```powershell
git clone https://github.com/Jherrie27/G2B-PostureCoach.git
cd G2B-PostureCoach
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env        # then paste a Groq key into .env
.venv\Scripts\python.exe -m streamlit run app.py
```

### If `requirements.txt` fails with `mediapipe==0.10.18 not found`
The user's **Python is too new (3.13)** — the pins only have wheels up to 3.12. Two options:
- **Fast:** install the runtime deps unpinned (let pip pick 3.13-compatible versions):
  ```powershell
  .venv\Scripts\python.exe -m pip install mediapipe opencv-python lightgbm sentence-transformers rank-bm25 groq streamlit python-dotenv
  ```
  (Drop the httpx pin here — newest `groq` works with newest `httpx`, so no conflict.)
- **Bulletproof:** use Python 3.12: `py -3.12 -m venv .venv`, then reinstall.

---

## 4. Secrets / API key (IMPORTANT)

- The Groq key is read from a **gitignored `.env`** (`GROQ_API_KEY=...`) via `python-dotenv` in
  `rag_query.py`. It raises a clear error if missing. **No key is in the repo or its history.**
- A live key was shared in chat for the demo. **Rotate/revoke it after the demo** at
  <https://console.groq.com> and put the new one in `.env`.
- Create `.env` quickly without an editor:
  ```powershell
  Set-Content -Path .env -Value 'GROQ_API_KEY=YOUR_KEY' -Encoding ascii
  ```
- `main.py` does NOT need the key (no chatbot). `app.py` DOES (it loads the chatbot at startup).

---

## 5. Environment

- **Python 3.12** venv (`.venv`). cp1252 console → all `print()`s are ASCII.
- Install in `requirements.txt` (self-sufficient: ChromaDB removed, `httpx==0.27.2` pinned,
  `sentence-transformers` + `python-dotenv` included). `requirements-core.txt` is the CV/ML subset.
- Heavy deps: `mediapipe`, `torch` (via sentence-transformers).

---

## 6. Architecture (data flow)

```
Webcam ─► PoseExtractor (MediaPipe, 9 landmarks ×4 coords)
       ─► reliability gate (ears+shoulders+hips visibility ≥ 0.5)   ← see §11 (hip framing issue)
       ─► normalize (origin = hip midpoint, scale = shoulder width, no rotation)
       ─► extract_features (14 features)
       ─► PostureClassifier  = 0.7 * LightGBM + 0.3 * rule_engine
       ─► TemporalSmoother (EMA α=0.3 + hysteresis 8 frames)
       ─► SessionTracker (distribution, correction_events, streaks)
       ─► PostureState ─► SharedPostureState (thread-safe)
                              │
        Streamlit app ◄───────┤
        ChatWorker ──► rag_query.answer(question, state):
            retriever (BGE-small dense + BM25 → RRF → posture filter)
            ─► prompt_builder (PostureState + refs) ─► Groq Llama 3.1-8B
            ─► grounding_check ─► answer
```

---

## 7. The 14 features (presentation reference)

"Clinically grounded" = each maps to a real physiotherapy measurement. Grouped by class:

**Forward-head / neck (4):** `ear_shoulder_offset_x`, `craniovertebral_angle` (clinical gold
standard for forward head), `head_forward_offset_z`, `nose_shoulder_offset_x`.
**Slouching (4):** `shoulder_roll_z` (forward shoulder roll, in depth), `torso_compression_ratio`
(shoulder-to-hip vertical distance shrinks), `elbow_forward_offset_z`, `spine_angle_3d`.
**Lean / lateral (5):** `shoulder_tilt_angle`, `hip_tilt_angle`, `midline_deviation_angle`,
`nose_centerline_offset_x`, `lateral_asymmetry_index`.
**Quality gate (1):** `landmark_confidence_mean` (used for reliability, not classification).

**Conventions (these fixed real bugs in the guide):**
- Tilt features fold to (−90, 90] → **level = 0°** (was 180°), so `lean` separates from `correct`.
- `craniovertebral`/`spine`/`midline` measure **deviation from vertical** (upright = 0°, was ~178°),
  so the relabel thresholds work (heuristic agreement 25% → 67%).

**Per-class separation (synthetic means):**
| | correct | slouching | neck_forward | lean |
|---|---|---|---|---|
| ear_shoulder_offset_x | ~0.15 | ~−0.20 | **~1.14** | ~0.15 |
| shoulder_roll_z | ~0 | **~0.78** | ~0 | ~0 |
| abs(shoulder_tilt) | ~2° | ~3° | ~2° | **~7°** |
| craniovertebral_angle | ~12° | ~14° | **~56°** | ~12° |
| torso_compression_ratio | ~1.6 | **~1.36** | ~1.5 | ~1.5 |

---

## 8. Landmarks used

**9 MediaPipe landmarks, each with x, y, z (depth), visibility = 36 numbers/frame**
(vs the original 5 landmarks × x,y = 10).

| Landmark (index) | Role | New vs v1? |
|---|---|---|
| Nose (0) | head reference/midline | kept |
| **Left/Right ear (7, 8)** | forward-head measurement | **NEW** |
| Left/Right shoulder (11, 12) | torso top, shoulder roll | kept |
| **Left/Right elbow (13, 14)** | secondary slouch signal | **NEW** |
| Left/Right hip (23, 24) | torso bottom / normalization anchor | kept |

Recovered information the original threw away: **ears** (forward head), the **z/depth axis**
(slouching + forward head live here), **elbows**, and **visibility scores**.

---

## 9. Why LightGBM over Random Forest (presentation reference)

**The model was not the main fix — the features were.** LightGBM was chosen for **Pi deployment**:
- Inference ~1 ms vs ~3 ms; model ~3 MB vs ~10 MB (matters on the Pi's CPU at 12–15 fps).
- Gradient boosting (sequential error-correcting trees) usually edges out RF bagging on tabular data;
  histogram-based splitting is faster/leaner; built-in L2 regularization helps a small dataset.
- We add a **rule overlay**: `0.7×LightGBM + 0.3×clinical rules` → interpretable (we can say *which
  threshold tripped*).

**The slouching↔correct fix (the headline story):** the old `spine_angle` measured whole-torso lean,
which barely changes when you slouch (hips stay put, upper back rounds). So slouching (~1.5°) and
correct (~2.5°) were nearly identical. We added **`shoulder_roll_z`** (forward shoulder roll in
depth) and **`torso_compression_ratio`** — the things that actually change when slouching — which the
x,y-only original was blind to. Now they separate cleanly.

---

## 10. RAG vs training data (a common question)

These are **separate systems**:
- The **CV classifier** (LightGBM) was trained on the **synthetic posture dataset**
  (`data/augmented/training_set.csv`). It only labels posture.
- The **chatbot/RAG** retrieves from the **physiotherapy textbook knowledge base** — *"Posture:
  Types, Exercises and Health Effects"*, **2842 chunks** in `rag_db_v3/`. It quotes the textbook
  (cites `REF 1`, `REF 3`, …), NOT the training data.

---

## 11. Live-demo findings & known issues

| Issue | Cause | Fix |
|---|---|---|
| **Label stuck / wrong (e.g. "Lean conf=0.39"), chatbot says "can't see your posture"** | **Hips not in the camera frame** → reliability gate fails (`is_reliable=False`) → pipeline holds a stale label and the chatbot refuses by design | **Frame the user from the waist up** + good front lighting. (Optional code fix below.) |
| Webcam black / wrong | multiple cameras; index 0 is a virtual/IR cam that opens but yields no frames | auto-detected; override `\$env:G2B_CAMERA_INDEX=1` (try 0,1,2). Fixed in `src/utils/camera.py`. |
| `mediapipe==0.10.18 not found` on install | user on Python 3.13 (pins only cover ≤3.12) | unpinned runtime install or use Python 3.12 (see §3) |
| `RuntimeError: GROQ_API_KEY is not set` | no `.env` / wrong folder | create `.env` (§4); run from project root |
| Wrong class even when framed well | **synthetic-trained model** (no real landmark data) | known limitation — collect real data (§13) |
| MediaPipe `DLL load failed` (Windows) | missing MSVC runtime / flaky native import | install VC++ Redistributable; constants already decoupled into `landmarks.py` |
| `groq` `proxies` TypeError | groq 0.9.0 vs httpx ≥0.28 | `httpx==0.27.2` pinned (or use newest groq unpinned) |

**Optional code fix for the hip-framing problem (recommended for desk webcams):** relax the
reliability gate in `src/cv/landmarks.py` (`KEY_FOR_RELIABILITY`) so hips are not *required*, and/or
lower `VISIBILITY_MIN` (0.5 → 0.4). MediaPipe still estimates off-screen hips, so the system degrades
gracefully instead of refusing. **Not yet applied** — decide based on whether the demo can frame the
waist.

---

## 12. Files & repo

**What's committed to GitHub (runnable out of the box, ~15 MB, 79 files):**
- Source: `src/` (`cv/`, `data/`, `state/`, `rag/`, `workers/`, `utils/`), `app.py`, `main.py`,
  `rag_query.py`, `train_lgbm.py`, `tests/`.
- Artifacts: `models/posture_lgbm_v3.txt` + `feature_order.json`; `rag_db_v3/embeddings.npy` +
  `chunks.jsonl`; `data/augmented/training_set.csv`, `data/kb_chunks_{raw,tagged}.jsonl`.
- Docs: `README.md`, `INSTRUCTIONS.md`, `HANDOFF.md`, this `Handoff-latest.md`, `CHANGES.md`,
  `pi5_setup.md`, `deploy/g2b-coach.service`, `G2B_POSTURE_REDESIGN.md`, `G2B_EXECUTION_GUIDE.md`.
- Fallbacks: `CV/*.csv`, `*_v2_backup.py`, `retrain.py`, `session.py`, `Test.py`.

**Gitignored (local only, NOT pushed):** `.env`, `.venv/`, `__pycache__/`, `images/`,
`mpii_human_pose_v1_u12_2/`, `*.zip`, `rag_db_v2/`, `*.pkl`, `.claude/`, `G2B-AI4-PostureCoach/`
(a stray nested repo — safe to delete), `bfg.jar`.

**Config files:** `.gitignore` (excludes the above), `.gitattributes` (normalizes line endings),
`.env.example` (key template).

---

## 13. What remains

1. **Live accuracy check** — sit framed from the waist up, run `tests/test_live_classifier.py`,
   confirm ≥3/4 postures. Tune `src/cv/rule_engine.py` thresholds or `synthesize.py` magnitudes if a
   class is consistently wrong.
2. **Hip-framing code relaxation** (§11) — optional, makes desk-webcam use robust.
3. **Raspberry Pi 5 deploy** — follow `pi5_setup.md`. **Use a USB webcam** (the CSI camera *module*
   is not supported — v3 uses only `cv2.VideoCapture`; the old `main.py` had a `picamera2` fallback
   that the rewrite dropped). Re-adding `picamera2` is a known, contained TODO.
4. **torch on ARM** — the RAG embedder needs PyTorch; install can be heavy/fragile on the Pi.
5. **Highest-value:** collect **real** raw-landmark data (a collector dumping all 33 landmarks ×4
   coords to `data/raw_landmarks/CV/*.csv`) to replace/augment the synthetic set and get a real
   accuracy number. `build_dataset.py` already auto-mixes real samples in when present.

---

## 14. Panel-defense quick answers

- **"Why was the old model confused between slouching and correct?"** Its features measured whole-body
  tilt; slouching is forward shoulder-roll + upper-back rounding, which lives in the depth axis the
  old x,y-only system ignored. We added `shoulder_roll_z` + `torso_compression_ratio`.
- **"What's your accuracy?"** ~99% on our **synthetic** evaluation set — we present this as evidence
  the features are now discriminative, not as a real-world number (the original collection didn't
  save raw landmarks, so we trained synthetically). Next step is real-data evaluation.
- **"Why LightGBM?"** ~3× faster, ~3× smaller for the Pi, slightly more accurate; the rule overlay
  keeps it interpretable. The features were the real fix, not the model.
- **"How is the chatbot grounded?"** Retrieval is filtered by the live posture class, the prompt
  injects the measured values, and a grounding check falls back to a state-only answer if the
  response doesn't use the references. It quotes a physiotherapy textbook, not the training data.
- **"What if it can't see the user?"** It computes landmark visibility; if key points (incl. hips)
  are < 0.5 visible it marks the frame unreliable, holds the last state, and the coach asks the user
  to reposition rather than guess.

---

## 15. Quick orientation for a new session

- CV math: `src/cv/features.py`. Synthetic data: `src/data/synthesize.py`. Thresholds:
  `src/cv/rule_engine.py` and `src/state/posture_state.py:NORMAL_THRESHOLDS` (keep them consistent).
- End-to-end change loop: edit features/thresholds → `src.data.build_dataset` → `train_lgbm.py` →
  run tests. RAG change: `src/rag/retriever.py` / `prompt_builder.py`.
- Everything is on `main` at https://github.com/Jherrie27/G2B-PostureCoach. The old v1/v2 system is
  preserved as fallback files.
