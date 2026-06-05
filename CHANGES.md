# CHANGES.md — Deviations from the Execution Guide

This log records every place the implementation departs from
`G2B_EXECUTION_GUIDE.md`, and why. (Guide "Rule of thumb #6": document deviations
as you go so you can defend them.)

## Big-picture deviations

### 1. Training data is 100% synthetic (no raw landmark dumps existed)
- **Guide assumed:** `CV/*.csv` contain raw 33-landmark `x/y/z/visibility` dumps,
  so the 14 features could be recomputed and the data relabeled/augmented.
- **Reality:** `CV/*.csv` contain only the 3 legacy scalar features
  (`neck_angle, spine_angle, shoulder_tilt`) + `label`. The 14 features need ear
  positions, z-depth, elbows and per-landmark visibility — **none of which can be
  reconstructed from 3 angles.**
- **Decision (user):** fully-synthetic dataset. `src/data/synthesize.py`
  `make_correct_pool()` generates anatomically-plausible "correct" seated landmark
  arrays from a canonical template (with scale/translation/sway/jitter), then
  `synth_slouching/neck_forward/lean` deform them. 1200 samples/class, 4800 total.
- **Defense:** the deformations match how MediaPipe physically sees each posture
  (forward z-shift for slouch, ear z-offset for forward head, xy tilt for lean).
  The reported accuracy is therefore a **synthetic-separability ceiling (~0.99 CV)**,
  NOT a real-world number. To get real numbers, collect raw landmark dumps with a
  new collector that writes `landmark_<i>_<x|y|z|v>` columns; `csv_to_features.py`
  and `build_dataset.py` already auto-mix real samples in when present.

### 2. Environment: fresh Python 3.12 venv with adjusted pins (not 3.11/0.10.14)
- **Guide:** nuke env, build a pinned **3.11.9** venv with `mediapipe==0.10.14`.
- **Decision (user):** fresh **3.12** `.venv` with `mediapipe==0.10.18` (the version
  with cp312 wheels). All other pins kept where 3.12-compatible.
- Run scripts via `./.venv/Scripts/python.exe`. The venv console is **cp1252** — all
  `print()`s use ASCII (no `→`/`✓`) to avoid `UnicodeEncodeError`.

### 3. RAG vector store: NumPy cosine matrix, not ChromaDB
- **Guide:** store vectors in ChromaDB (`rag_db_v3/`), filter with `where`.
- **Reality:** `chroma-hnswlib` has **no installable wheel** for this platform — the
  only cp312 wheel is `0.7.5`, but every buildable `chromadb` pins `0.7.3` or `0.7.6`,
  both of which try to compile and need MSVC Build Tools.
- **Decision:** drop ChromaDB. At ~2.8k chunks a brute-force NumPy cosine matrix is
  faster than HNSW and dependency-free. `rag_db_v3/` = `embeddings.npy` +
  `chunks.jsonl`. Retrieval = dense cosine + BM25 → RRF → Python-side posture filter.
  Same behavior as the guide's design, fewer moving parts.

### 4. Chunk tagging: reused existing `posture_class`, skipped Groq tagging
- **Guide:** tag all chunks with an LLM (`tagger.py`, ~2842 Groq calls).
- **Reality:** the existing `rag_db_v2/rag_db/chroma.sqlite3` already carries a
  `posture_class` tag per chunk. `src/rag/export_chunks.py` reuses those (mapping
  `correct→correct_posture`, `leaning→lean`, and treating `general/exercise/breaks`
  as applicable to all) and derives `anatomical_region`/`content_type`/`key_terms`
  with local heuristics. `tagger.py` is kept for the richer-tagging path but unused.
- **Defense:** zero API cost, deterministic, and posture-filtered retrieval verified
  to return on-class chunks.

## Code-level fixes to guide bugs

### 5. `features.py` — tilt angle convention (shoulder/hip tilt)
- Guide computed `arctan2(dy,dx)` vs the x-axis, so a **level** shoulder line read
  **180°**, not 0°, and a lean read ~163° — the feature couldn't separate
  `correct` from `lean`. Fixed: `_line_tilt_from_horizontal_deg` folds to (-90, 90]
  so level = 0° and magnitude grows with tilt.

### 6. `features.py` — vertical-deviation convention (craniovertebral/spine/midline)
- Guide measured angle from the **+y** axis, but torso/head vectors point **−y** (up)
  in image coords, so an *upright* posture read ~178°. The relabel thresholds
  (`> 18`, `> 6`, meaning "deviated") then fired on everything → heuristic agreement
  was 25%. Fixed: `_deviation_from_vertical_deg` folds with `min(a, 180−a)` so
  upright = 0° and deviation grows. Heuristic agreement rose to ~67%.

### 7. Decoupled landmark constants from MediaPipe (`src/cv/landmarks.py`)
- Guide put `LANDMARK_INDICES`/`ORDERED_NAMES` in `pose_extractor.py`, which imports
  `mediapipe`. That made data/training/feature code transitively load the native
  MediaPipe DLLs (heavy + intermittently `DLL load failed` on Windows). Moved the
  constants to a dependency-free `landmarks.py`; `pose_extractor.py` re-exports them.

### 8. `synthetic 'correct' template` tuned so correct = no-deviation
- Raised the shoulder→hip vertical gap so `torso_compression_ratio ≈ 1.6` (above the
  1.55 "normal" threshold) and tightened head-depth jitter, so a correct posture
  reports `primary_issue == "none"` instead of a spurious compression/forward-head
  deviation.

### 9. `httpx==0.27.2` pinned
- `groq==0.9.0` passes `proxies=` to httpx, which `httpx>=0.28` removed
  (`TypeError: ... unexpected keyword 'proxies'`). Pinned httpx back.

## Staged installs (Windows wheel issues)
- `requirements-core.txt` (CV/ML/app) installed first; RAG deps
  (`sentence-transformers`, torch) installed separately. `requirements.txt` is the
  full target (ChromaDB removed, httpx pinned) used for the Pi.

## What is NOT yet verified (needs hardware / a person)
- Live webcam classification & smoothing (Phase 4.4 / 5.5 — `tests/test_*` are manual).
- Live Streamlit UI (Phase 7.4).
- Raspberry Pi 5 deployment & FPS (Phase 8).
- Demo polish / landmark overlay (Phase 10).

## Files kept as fallback (not deleted, per guide rule #3)
`posture_classifier.pkl`, `posture_classifier_v2.pkl`, `rag_db_v2/`, `CV/`,
`rag_query_v2_backup.py`, `app_v2_backup.py`, `main_v2_backup.py`.
