# Skin Cancer Classifier

Binary benign/malignant classifier for dermoscopic skin lesion images.

- **Frontend** — React 19 + Vite + MUI (repo root, deploys to Vercel)
- **Backend** — Flask + TensorFlow (`Skin_cancer_backend-main/`, deploys to Render)

> Research/educational project. Not a medical device and not a substitute for
> diagnosis by a qualified clinician.

## Architecture

```
/                             Vite frontend  -> Vercel
  render.yaml                 backend service definition (rootDir points below)
  Skin_cancer_backend-main/   Flask API      -> Render
    app.py                    GET / health, POST /predict
    best_model.keras          76MB, saved with Keras 2.15.0
```

The backend **cannot run on Vercel**: TensorFlow plus the 76MB model is roughly
3x Vercel's 250MB serverless function limit. That is why the API lives on Render.

## API

| Method | Path       | Body                     | Response |
|--------|------------|--------------------------|----------|
| GET    | `/`        | —                        | `{"status":"ok"}` |
| POST   | `/predict` | multipart, field `image` | `{"probability":0.87,"result":"malignant"}` |

Classified `malignant` when probability >= 0.231.

## Local development

Backend — **requires Python 3.10 or 3.11**. TensorFlow 2.15 does not support 3.12+:

```bash
cd Skin_cancer_backend-main
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt
python app.py                                    # serves on :8080
```

Frontend:

```bash
npm install
npm run dev                                      # proxies to :8080 by default
```

## Deployment

### Backend (Render)

1. dashboard.render.com → **New** → **Blueprint**
2. Connect this repo. Render reads `render.yaml` at the root, which already sets
   `rootDir: Skin_cancer_backend-main`, binds `$PORT`, and pins Python 3.10.13.
3. **Apply**. First build takes 10–15 min (TensorFlow is a large wheel).
4. Confirm `https://<service>.onrender.com/` returns `{"status":"ok"}`.

### Frontend (Vercel)

1. Project **Settings → Environment Variables**
2. Add `VITE_API_URL` = `https://<service>.onrender.com` (no trailing slash)
3. **Redeploy.** Vite inlines env vars at build time — setting the variable
   without rebuilding has no effect.
4. **Settings → Deployment Protection** — turn off if you want the site publicly
   reachable without a Vercel login.

## Known constraints

- **Free-tier memory.** TF-CPU plus the model sits near Render free's 512MB
  ceiling. If the service OOMs, move to Render Starter (2GB) or Hugging Face
  Spaces (free, 16GB).
- **Cold starts.** Free services sleep after 15 min idle; the next request pays
  a ~60s penalty while TensorFlow reloads.
- **Version pins are load-bearing.** `best_model.keras` was saved with Keras
  2.15.0. TF 2.16+ ships Keras 3 and cannot deserialize this file's custom
  `KANLayer`. Do not unpin `tensorflow-cpu==2.15.0` or `numpy<2`.
- **No input normalization.** `preprocess_image()` feeds raw 0–255 floats to the
  model. If training used `/255.0` or a `preprocess_input`, this needs to match
  or every prediction is wrong. Verify against the training notebook.
