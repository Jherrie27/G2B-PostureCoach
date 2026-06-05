# How to Run the G2B Posture Coach (Team Setup Guide)

Step-by-step instructions to get the project running on your own computer.
Aimed at teammates — no prior knowledge of the codebase needed. Total time: ~15 min
(most of it is the one-time install download).

> If anything fails, jump to **Troubleshooting** at the bottom — it covers the common issues.

---

## 0. What you need first (prerequisites)

| Requirement | How to check | Notes |
|---|---|---|
| **Python 3.11 or 3.12** | `python --version` | **Not 3.13** (some libraries have no wheels yet). Get it from python.org if missing. |
| **Git** | `git --version` | To clone the repo. |
| **A webcam** | — | Built-in or USB. |
| **A Groq API key** | — | Free at <https://console.groq.com> → API Keys. Needed for the chatbot. |
| **Internet (first run)** | — | The first run downloads a small embedding model (~130 MB). |

---

## 1. Get the code

```bash
git clone https://github.com/Jherrie27/G2B-PostureCoach.git
cd G2B-PostureCoach
```

---

## 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
```
**macOS / Linux:**
```bash
python3 -m venv .venv
```

You do **not** need to "activate" it — just call the venv's Python directly in the steps below:
- Windows: `.venv\Scripts\python.exe`
- macOS/Linux: `.venv/bin/python`

(If you prefer activation: Windows `.venv\Scripts\Activate.ps1`, mac/Linux `source .venv/bin/activate`, then you can use plain `python`.)

---

## 3. Install the dependencies (one time, ~5–10 min)

**Windows:**
```powershell
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
**macOS / Linux:**
```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

This installs MediaPipe, OpenCV, LightGBM, the embedding model library (PyTorch), Streamlit, etc.
The PyTorch download is the big one — be patient on the first install.

---

## 4. Add your Groq API key

The chatbot needs a Groq key. Copy the template and paste your key into it:

**Windows:**
```powershell
copy .env.example .env
notepad .env
```
**macOS / Linux:**
```bash
cp .env.example .env
nano .env
```

Edit the line so it reads (no quotes, no spaces):
```
GROQ_API_KEY=gsk_your_actual_key_here
```
Save and close. The `.env` file is **private** — it is never uploaded to GitHub.

---

## 5. Run the project

### Option A — Full app (webcam video + live posture + chatbot)  ← recommended
**Windows:**
```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```
**macOS / Linux:**
```bash
.venv/bin/python -m streamlit run app.py
```
A browser tab opens at `http://localhost:8501`. **Allow camera access.** You'll see your video,
a live posture label, metric tiles, and a chat box. Try slouching, then ask the coach
*"Am I sitting correctly?"*

### Option B — Simple terminal mode (prints your posture)
**Windows:**
```powershell
.venv\Scripts\python.exe -u main.py
```
**macOS / Linux:**
```bash
.venv/bin/python -u main.py
```
It prints a line whenever your posture changes. Press **Ctrl+C** to stop.

---

## 6. Try the 4 postures

While the app or `main.py` is running, do each of these and watch the label:

| Sit like this | Expected label |
|---|---|
| Upright, head over shoulders | `correct_posture` |
| Round your upper back, shoulders forward | `slouching` |
| Torso straight, head jutting forward (turtle) | `neck_forward` |
| Tilt your body to one side | `lean` |

> Heads-up: the model is currently trained on **synthetic data**, so live accuracy is not final.
> If a posture is consistently mislabeled, that's a known limitation, not a setup mistake — note it
> and report it to the team.

---

## 7. (Optional) Run the automated tests
**Windows:**
```powershell
.venv\Scripts\python.exe -m pytest tests/test_features.py tests/test_smoother.py tests/test_state.py -q
.venv\Scripts\python.exe tests/test_env.py
```
Expect `14 passed` and `ALL IMPORTS OK`. This confirms the install is healthy.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| **`RuntimeError: GROQ_API_KEY is not set`** | You skipped Step 4, or `.env` has a typo. It must be `GROQ_API_KEY=gsk_...` on its own line. |
| **App opens but webcam is black / wrong camera** | You have multiple cameras. Set the index before running: PowerShell `\$env:G2B_CAMERA_INDEX=1` (try 0, 1, 2). mac/Linux `export G2B_CAMERA_INDEX=1`. |
| **"No working camera found"** | Camera is in use by another app (Zoom/Teams) — close it. Or it needs permission (Windows: Settings → Privacy → Camera). |
| **`python` not found / wrong version** | Install Python 3.11 or 3.12 from python.org and re-create the venv (Step 2). Don't use 3.13. |
| **MediaPipe import error on Windows** (`DLL load failed`) | Install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe), then retry. |
| **First chat message is slow** | The embedding model downloads/loads on first use (~130 MB). It's fast afterward. |
| **`pip` install is very slow** | That's PyTorch downloading. Let it finish; it's a one-time cost. |
| **Streamlit didn't open a browser** | Open `http://localhost:8501` manually. |

---

## Notes for the Raspberry Pi 5 deployment

Pi setup is **separate** — see **`pi5_setup.md`** and `deploy/g2b-coach.service`. Two things to know:
- Use a **USB webcam** on the Pi for now (the CSI ribbon-cable *camera module* isn't supported by
  the current code yet).
- The code automatically uses the lighter, faster MediaPipe model on the Pi.

---

## Where to learn more
- **`README.md`** — project overview and quick start.
- **`HANDOFF.md`** — full architecture, file map, and what's left to do.
- **`CHANGES.md`** — design decisions and why things are the way they are.
