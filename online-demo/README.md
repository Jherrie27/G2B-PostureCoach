# Online Browser Demo

This is a separate deployment path for a public URL demo. It does not replace
the Streamlit/Raspberry Pi app in `app.py`, `deploy/`, or
`submission/Web_Deployment/`.

## What it does

- Runs entirely in the professor's browser over HTTPS.
- Uses the browser webcam through `getUserMedia`.
- Runs MediaPipe Pose Landmarker in JavaScript.
- Reuses the existing 14-feature posture math.
- Loads `../models/posture_lgbm_v3.txt` and evaluates the LightGBM trees in
  JavaScript.
- Provides a local rule-based coach so no Groq key is exposed in the browser.

## GitHub Pages URL

The included workflow publishes this demo to GitHub Pages:

```text
https://jherrie27.github.io/G2B-PostureCoach/
```

The direct demo path is:

```text
https://jherrie27.github.io/G2B-PostureCoach/online-demo/
```

GitHub Pages must be configured to use GitHub Actions. The workflow can also be
started manually from the Actions tab.

## Local smoke test

From the project root:

```powershell
.venv\Scripts\python.exe -m http.server 8000
```

Open:

```text
http://localhost:8000/online-demo/
```

Camera access works on `localhost` and HTTPS origins. It will not work from a
plain `file://` URL.
