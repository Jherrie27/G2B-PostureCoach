# G2B Posture Correction Coach

Real-time posture analysis + AI coaching for desk workers, built for Raspberry Pi 5.

A webcam feed is analyzed with **MediaPipe Pose**, reduced to **14 posture features**, and
classified into `correct_posture | slouching | neck_forward | lean` by a **LightGBM + rule-fusion**
model with **temporal smoothing**. A **posture-conditioned RAG** chatbot (BGE-small retrieval +
BM25 hybrid + Groq Llama 3.1-8B) answers questions grounded in your *live* measured posture.

> **Status:** v3 redesign. The classifier is currently trained on a **synthetic** dataset (the
> original collection only stored 3 scalar features, not raw landmarks), so reported accuracy is a
> synthetic-separability ceiling — see `CHANGES.md` and `HANDOFF.md`. Collect real landmark data for
> real-world numbers.

## Quick start

```bash
# 1. Python 3.12 venv
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip          # Windows
.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.venv\Scripts\python.exe -m pip install "sentence-transformers==3.0.1" "httpx==0.27.2"

# 2. API key
copy .env.example .env        # then edit .env and set GROQ_API_KEY

# 3. Run
.venv\Scripts\python.exe -m streamlit run app.py     # full UI: video + chat
.venv\Scripts\python.exe -u main.py                  # headless terminal
```

The webcam index is auto-detected; override with `G2B_CAMERA_INDEX=1` if needed.

## What's included (and runnable out of the box)

- `src/` — CV pipeline (`cv/`), data synthesis (`data/`), state (`state/`), RAG (`rag/`),
  workers (`workers/`), utils (`utils/`).
- `app.py` (Streamlit), `main.py` (headless), `rag_query.py`, `train_lgbm.py`, `tests/`.
- **Trained artifacts:** `models/posture_lgbm_v3.txt`, `models/feature_order.json`.
- **RAG index:** `rag_db_v3/` (`embeddings.npy` + `chunks.jsonl`).
- **Data to rebuild:** `data/augmented/training_set.csv`, `data/kb_chunks_tagged.jsonl`.

## Rebuild from scratch (optional)

```bash
.venv\Scripts\python.exe -m src.data.build_dataset    # -> data/augmented/training_set.csv
.venv\Scripts\python.exe train_lgbm.py                # -> models/posture_lgbm_v3.txt
.venv\Scripts\python.exe -m src.rag.build_index       # -> rag_db_v3/ (downloads BGE-small)
```

## Tests

```bash
Run
.venv\Scripts\python.exe -m streamlit run app.py

.venv\Scripts\python.exe -m pytest tests/test_features.py tests/test_smoother.py tests/test_state.py -q
.venv\Scripts\python.exe tests/test_live_classifier.py   # manual: needs you on camera
```

## Docs

- **`HANDOFF.md`** — full project context, architecture, file map, what remains.
- **`CHANGES.md`** — every deviation from the design guide and why.
- **`pi5_setup.md`** + **`deploy/g2b-coach.service`** — Raspberry Pi 5 deployment.
- **`G2B_POSTURE_REDESIGN.md`** / **`G2B_EXECUTION_GUIDE.md`** — the original design + step plan.

## License / attribution

Academic project (Group G2B). Physiotherapy KB: *Posture: Types, Exercises and Health Effects*.
