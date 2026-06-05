# Raspberry Pi 5 Setup — G2B Posture Coach v3

Adapted from Guide Phase 8 to match this project's actual stack (no ChromaDB,
NumPy vector store, httpx pin, pre-trained synthetic model).

## 1. OS prep (Raspberry Pi OS 64-bit / Bookworm)
```bash
uname -m            # must print aarch64
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip python3-dev git \
    build-essential libgl1 libglib2.0-0 libgtk-3-0 v4l-utils
```

## 2. Get the project + branch
```bash
cd ~
git clone <your-repo-url> g2b_posture_coach
cd g2b_posture_coach
git checkout feature/redesign-v3
```

## 3. venv + deps
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Core CV/ML/app + RAG. ChromaDB is NOT used (NumPy store), so no hnswlib build.
pip install -r requirements.txt
# If mediapipe has no wheel for your Python:
#   pip install --index-url https://www.piwheels.org/simple mediapipe
```
Gotchas:
- `numpy.core.multiarray failed to import` → `pip install --force-reinstall numpy==1.26.4`
- `libGL.so.1: cannot open shared object file` → `sudo apt install libgl1`
- groq `proxies` TypeError → ensure `httpx==0.27.2` (it's pinned in requirements.txt)

## 4. Copy the trained artifacts (tiny) from the dev machine
```bash
# from dev machine:
scp models/posture_lgbm_v3.txt models/feature_order.json pi@<pi-ip>:~/g2b_posture_coach/models/
scp -r rag_db_v3/ pi@<pi-ip>:~/g2b_posture_coach/        # embeddings.npy + chunks.jsonl
# (or rebuild rag_db_v3 on the Pi: python -m src.rag.build_index ; needs data/kb_chunks_tagged.jsonl)
```

## 5. API key
```bash
echo 'GROQ_API_KEY=your-key-here' > ~/g2b_posture_coach/.env
```
`rag_query.py` reads `GROQ_API_KEY` from the environment (falls back to the repo key).

## 6. Performance config (Pi auto-detected)
`src/utils/config.py` returns `model_complexity=0` (MediaPipe Lite) and `target_fps=12`
on aarch64 automatically. Override with env vars if needed:
```bash
export G2B_MP_COMPLEXITY=0
export G2B_TARGET_FPS=12
```

## 7. First run + FPS check
```bash
python main.py            # headless; or:
python tests/test_pipeline.py   # prints fps at the end (target >= 10)
```

## 8. Streamlit on the LAN
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
# open http://<pi-ip>:8501
```

## 9. Auto-start on boot
Install `deploy/g2b-coach.service` (edit User/paths if not `pi`):
```bash
sudo cp deploy/g2b-coach.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now g2b-coach.service
journalctl -u g2b-coach.service -f
```
