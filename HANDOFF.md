# G2B Posture Correction Coach — HANDOFF

**Date:** 2026-06-05
**Branch:** `feature/redesign-v3`
**Status:** Phases 1–8 implemented & committed. Software complete and verified for everything
that does not require a webcam/Raspberry Pi. Live-on-camera and Pi steps remain (need hardware).

This document is self-contained: a new engineer (or a fresh AI session) should be able to pick
up the project from here without re-reading the chat history.

---

## 0. TL;DR

- We rebuilt the Posture Coach from a **3-feature Random Forest** to a **14-feature LightGBM +
  rule fusion + temporal smoothing** CV pipeline, plus a **posture-conditioned RAG** chatbot,
  following `G2B_EXECUTION_GUIDE.md` (companion: `G2B_POSTURE_REDESIGN.md`).
- **Two facts changed the plan** (decided with the user):
  1. **No raw landmark data exists** — the `CV/*.csv` only have 3 scalar features, not the
     33-landmark dumps the guide assumed. → We use a **fully-synthetic dataset**.
  2. The environment is a **fresh Python 3.12 venv** with adjusted pins (not the guide's 3.11).
- The model reports **CV accuracy 0.991**, but that is a **synthetic-separability ceiling**, NOT
  a real-world number. Real-webcam accuracy will be lower until real landmark data is collected.
- A real bug was found & fixed during testing: **the webcam is camera index 1, not 0** (index 0
  opens but yields no frames on this machine). Auto-detection added.
- Full deviation log is in **`CHANGES.md`**. Pi deploy steps in **`pi5_setup.md`**.

---

## 1. How to run (do this first)

All commands run from the project root. The venv Python is `.venv\Scripts\python.exe`
(no "activate" needed). The webcam auto-detects; override with `$env:G2B_CAMERA_INDEX=1` if needed.

**Full app (video + live label + posture-aware chat):**
```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```
Opens `http://localhost:8501`. Allow camera. Slouch/lean, then ask the coach a question.

**Headless terminal (prints posture changes):**
```powershell
.venv\Scripts\python.exe -u main.py
```
`-u` = unbuffered so output shows live. `Ctrl+C` to stop.

**Automated tests (no camera, ~2s):**
```powershell
.venv\Scripts\python.exe -m pytest tests/test_features.py tests/test_smoother.py tests/test_state.py -q
.venv\Scripts\python.exe tests/test_env.py        # -> ALL IMPORTS OK
.venv\Scripts\python.exe tests/test_env_rag.py    # -> RAG IMPORTS OK
```

**Manual live tests (need you in the camera frame):**
```powershell
.venv\Scripts\python.exe tests/test_live_classifier.py   # 30s, prints label+features per frame
.venv\Scripts\python.exe tests/test_pipeline.py          # 60s, smoothed labels + primary_issue
```

---

## 2. Environment

- **Python:** 3.12.8 (conda base), but the project runs in a dedicated **`.venv`** (3.12).
- **Console encoding is cp1252** — all `print()`s use ASCII (no `→`/`✓`) to avoid `UnicodeEncodeError`.
- **Install (already done in `.venv`):**
  - `requirements-core.txt` → CV/ML/app stack (mediapipe 0.10.18, opencv, numpy 1.26.4,
    lightgbm, scikit-learn, pandas, streamlit, groq, pytest, matplotlib, seaborn, jupyter, rank-bm25).
  - Then separately: `sentence-transformers==3.0.1` (pulls torch 2.12) and `httpx==0.27.2`.
  - **ChromaDB is NOT installed/used** (see §6). `requirements.txt` is the full target (Pi) with
    ChromaDB removed and httpx pinned.
- **To recreate the venv from scratch:**
  ```powershell
  python -m venv .venv
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -r requirements-core.txt
  .venv\Scripts\python.exe -m pip install "sentence-transformers==3.0.1" "httpx==0.27.2"
  ```
- **GROQ_API_KEY:** `rag_query.py` reads it from env, falling back to the key already in the repo
  (move it to a `.env` for production).

---

## 3. Architecture (data flow)

```
Webcam ─► PoseExtractor (MediaPipe, 9 landmarks ×4 coords)
       ─► normalize (origin=hip mid, scale=shoulder width, no rotation)
       ─► extract_features (14 features)
       ─► PostureClassifier  = 0.7 * LightGBM + 0.3 * rule_engine
       ─► TemporalSmoother (EMA α=0.3 + hysteresis 8 frames)
       ─► SessionTracker (distribution, correction_events, streaks)
       ─► PostureState (dataclass) ─► SharedPostureState (thread-safe)
                                          │
                Streamlit app ◄───────────┤
                ChatWorker ──► rag_query.answer(question, state):
                     retriever (BGE-small dense + BM25 → RRF → posture filter)
                     ─► prompt_builder (PostureState + refs) ─► Groq Llama 3.1-8B
                     ─► grounding_check ─► answer
```

---

## 4. The 14 features (`src/cv/features.py`)

Order (also in `models/feature_order.json`):
`ear_shoulder_offset_x, craniovertebral_angle, head_forward_offset_z, nose_shoulder_offset_x,
shoulder_roll_z, torso_compression_ratio, elbow_forward_offset_z, spine_angle_3d,
shoulder_tilt_angle, hip_tilt_angle, midline_deviation_angle, nose_centerline_offset_x,
lateral_asymmetry_index, landmark_confidence_mean`.

**Conventions (these were bugs in the guide, fixed here):**
- `shoulder_tilt_angle` / `hip_tilt_angle`: line tilt **folded to (−90, 90]** → level = 0°, grows
  with lean. (Guide gave 180° for a level line.)
- `craniovertebral_angle` / `spine_angle_3d` / `midline_deviation_angle`: **deviation from the
  vertical axis** via `min(a, 180−a)` → upright = 0°. (Guide gave ~178° for upright, which broke
  the relabel thresholds.)

**Per-class signatures (means on the synthetic set):**
| class | ear_offset_x | shoulder_roll_z | abs(tilt) | cranio | compression |
|---|---|---|---|---|---|
| correct_posture | ~0.15 | ~0 | ~2° | ~12° | ~1.6 |
| slouching | ~−0.20 | **~0.78** | ~3° | ~14° | **~1.36** |
| neck_forward | **~1.14** | ~0 | ~2° | **~56°** | ~1.5 |
| lean | ~0.15 | ~0 | **~7°** | ~12° | ~1.5 |

**Important:** landmark constants live in `src/cv/landmarks.py` (NO mediapipe import) so
data/training/feature code doesn't load the native MediaPipe DLL (which intermittently fails on
Windows). `pose_extractor.py` and `normalizer.py` re-export them.

---

## 5. Data & model

- **Synthetic dataset:** `src/data/build_dataset.py` → `data/augmented/training_set.csv`
  (4800 rows, 1200/class, all `source=synthetic`). It synthesizes a "correct" landmark pool
  (`src/data/synthesize.py:make_correct_pool`) then deforms it into slouch/neck/lean.
  - Rebuild: `.venv\Scripts\python.exe -m src.data.build_dataset`
  - It auto-mixes in **real** samples if `data/raw_landmarks/CV/*.csv` ever contains real
    landmark dumps (columns `landmark_<i>_<x|y|z|v>`); edit `csv_to_features._row_to_landmark_array`
    if column names differ.
- **Model:** `train_lgbm.py` → `models/posture_lgbm_v3.txt` + `models/feature_order.json`.
  - Retrain: `.venv\Scripts\python.exe train_lgbm.py`
  - Last result: **CV acc 0.991 ±0.002**, held-out 0.994, per-class F1 ≥ 0.98.
  - **Caveat:** synthetic ceiling, not real-world. Top features by gain: torso_compression,
    craniovertebral, spine_angle_3d, shoulder_roll_z, ear_shoulder_offset_x (no leakage).
- **Classifier fusion:** `src/cv/classifier.py` = `0.7*LGBM + 0.3*rule_engine`. If live behavior
  is bad, lower `rule_weight` or tune thresholds in `src/cv/rule_engine.py`.

---

## 6. RAG (posture-conditioned)

- **ChromaDB was dropped.** `chroma-hnswlib` has no installable cp312 wheel (0.7.5 has a wheel but
  every buildable chromadb pins 0.7.3/0.7.6 → needs MSVC). Replaced with a **NumPy cosine store**.
- **Store:** `rag_db_v3/embeddings.npy` (2842×384, BGE-small) + `rag_db_v3/chunks.jsonl`.
- **Chunks came from the existing `rag_db_v2/rag_db/chroma.sqlite3`** (2842 chunks, already tagged
  with `posture_class`). `src/rag/export_chunks.py` reuses those tags (mapping
  `correct→correct_posture`, `leaning→lean`; `general/exercise/breaks`→all) + local heuristics for
  `anatomical_region`/`content_type`/`key_terms` → `data/kb_chunks_tagged.jsonl`.
  **No Groq tagging needed** (`src/rag/tagger.py` exists for richer tags but is unused).
- **Rebuild index:** `.venv\Scripts\python.exe -m src.rag.export_chunks` then
  `.venv\Scripts\python.exe -m src.rag.build_index` (downloads BGE-small once).
- **Retriever:** `src/rag/retriever.py` — dense cosine + BM25 → RRF (k=60) → Python-side posture
  filter (falls back to unfiltered top-k if filter empties).
- **Query path:** `rag_query.answer(question, state)` → expand query with posture → retrieve →
  `prompt_builder.build_prompt` → Groq `llama-3.1-8b-instant` → `grounding_check.is_grounded`
  (falls back to a state-only template if ungrounded).
- **Verified E2E:** simulated slouching state → `grounded=True`, answer cited measured values.
- **httpx pinned to 0.27.2** (groq 0.9.0 passes `proxies=`, removed in httpx ≥0.28).

---

## 7. File map (new / changed)

**CV pipeline** (`src/cv/`): `landmarks.py` (constants, no mediapipe), `pose_extractor.py`,
`normalizer.py`, `features.py`, `rule_engine.py`, `classifier.py`, `smoother.py`, `pipeline.py`.

**Data** (`src/data/`): `csv_to_features.py`, `relabel.py`, `augment.py`, `synthesize.py`,
`build_dataset.py`.

**State** (`src/state/`): `posture_state.py` (dataclass + `feature_deviations`/`primary_issue`),
`session_tracker.py`.

**RAG** (`src/rag/`): `export_chunks.py`, `build_index.py`, `retriever.py`, `prompt_builder.py`,
`grounding_check.py`, `reranker.py` (Noop default), `tagger.py` (unused).

**Workers** (`src/workers/`): `shared_state.py`, `cv_worker.py`, `chat_worker.py`.

**Utils** (`src/utils/`): `config.py` (Pi-aware complexity/fps), `camera.py` (auto-detect index).

**Top level:** `train_lgbm.py`, `app.py` (rewritten), `main.py` (rewritten), `rag_query.py`
(rewritten). Backups: `app_v2_backup.py`, `main_v2_backup.py`, `rag_query_v2_backup.py`.

**Artifacts:** `models/posture_lgbm_v3.txt`, `models/feature_order.json`,
`data/augmented/training_set.csv`, `data/kb_chunks_{raw,tagged}.jsonl`, `rag_db_v3/`.

**Tests:** `tests/test_env.py`, `test_env_rag.py`, `test_features.py`, `test_smoother.py`,
`test_state.py` (all automated/green); `test_live_classifier.py`, `test_pipeline.py` (manual).

**Docs:** `CHANGES.md` (deviation log), `pi5_setup.md`, `deploy/g2b-coach.service`, this file.

**Kept as fallback (do NOT delete):** `posture_classifier.pkl`, `posture_classifier_v2.pkl`,
`rag_db_v2/`, `CV/`, the three `*_v2_backup.py` files.

---

## 8. Git history (branch `feature/redesign-v3`)

```
Fix: auto-detect working webcam index (your cam is index 1, not 0)
Phase 8 artifacts + CHANGES log
Phase 7: workers + shared state + Streamlit app integrated
Phase 6: posture-conditioned RAG (NumPy store + hybrid retrieval + grounding)
Phase 5: temporal smoother + PostureState + session tracker + pipeline
Phase 4: LightGBM v3 trained (CV acc 0.991), rule-fusion classifier
Phase 3: fully-synthetic balanced training set (4800 rows, 1200/class)
Phase 2: 14-feature CV pipeline (pose_extractor, normalizer, features) + tests
Phase 1: env pinned (3.12 adjusted), dir scaffolded, core deps installed
```
(Pre-existing `Publish` / `Initial commit` commits sit below; they already tracked the large
`images/`, `mpii_*`, and data files.)

---

## 9. Known issues, fixes, gotchas

| Issue | Cause | Status |
|---|---|---|
| `main.py` exits instantly, no output | Webcam is **index 1**, not 0 (index 0 opens but MSMF can't grab frames) | **Fixed** — `src/utils/camera.py` auto-detects; override `G2B_CAMERA_INDEX` |
| No posture lines printed | Captured stdout is block-buffered | Run with `python -u`, or in a real terminal (TTY) |
| `UnicodeEncodeError` on `→`/`✓` | venv console is cp1252 | All prints are ASCII |
| `TypeError: ... 'proxies'` from groq | groq 0.9.0 vs httpx ≥0.28 | **Fixed** — `httpx==0.27.2` |
| `chromadb` won't install | `chroma-hnswlib` no cp312 wheel | **Designed out** — NumPy store |
| MediaPipe `DLL load failed` intermittently | heavy native import in data code | **Fixed** — constants in `landmarks.py` |
| GROQ key hardcoded in `rag_query.py` | inherited from old code | Works; move to `.env` for prod |

Live-debug references: `G2B_EXECUTION_GUIDE.md` Phase 9 (symptom→fix tables for MediaPipe,
webcam, classifier, smoother, RAG, Streamlit, Pi).

---

## 10. What remains (needs hardware / a person)

1. **Live accuracy check** (Phases 4.4 / 5.5): sit in front of the camera, run
   `tests/test_live_classifier.py`, confirm ≥3/4 postures classify correctly. If a class is
   consistently wrong, capture the printed `ear_x / sh_z / tilt` values and tune
   `src/cv/rule_engine.py` thresholds or the `synthesize.py` deformation magnitudes.
2. **Streamlit UI live** (Phase 7.4): `streamlit run app.py`.
3. **Raspberry Pi 5 deploy + FPS** (Phase 8): follow `pi5_setup.md` + `deploy/g2b-coach.service`.
   Target ≥10 fps with `model_complexity=0` (auto-set on aarch64).
4. **Demo polish** (Phase 10): landmark overlay on the webcam feed, demo script. Q&A prep already
   in the guide's Phase 10.3.
5. **(Highest-value future work)** Collect **real** raw-landmark data to replace/augment the
   synthetic set and get a real accuracy number. Write a collector that dumps all 33 landmarks
   ×4 coords per frame to `data/raw_landmarks/CV/*.csv`; the pipeline auto-mixes them in.

---

## 11. Quick orientation for a new session

- Read `CHANGES.md` (why things differ from the guide) and this file.
- The CV math lives in `src/cv/features.py`; the synthetic data generator in
  `src/data/synthesize.py`; thresholds in `src/cv/rule_engine.py` and
  `src/state/posture_state.py:NORMAL_THRESHOLDS` (keep these two consistent).
- To change behavior end-to-end: edit features/thresholds → `build_dataset` → `train_lgbm` →
  re-run tests. To change retrieval: edit `src/rag/retriever.py` / `prompt_builder.py`.
- Everything is committed on `feature/redesign-v3`. Old system is intact as fallback.
```
