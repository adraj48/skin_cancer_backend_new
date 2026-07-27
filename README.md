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
    model.tflite              25MB, what actually gets served
    best_model.h5             77MB, source model the tflite was converted from
    best_model.keras          77MB, original -- unloadable on Linux, see below
```

The backend **cannot run on Vercel**: even the TFLite build exceeds Vercel's
250MB serverless function limit once numpy and OpenCV are counted, and the full
TensorFlow build was roughly 3x over. The API lives on Render.

Serving runs on **TFLite, not TensorFlow**. TF's runtime is ~400MB resident,
which left too little of Render's 512MB free tier for an inference pass -- the
container was OOM-killed on every request while the health check still passed.
tflite-runtime is ~5MB. `model.tflite` was converted from `best_model.h5`
float32 with no quantisation and verified to agree to 3.3e-07, so the 0.231
decision threshold remains calibrated.

## API

| Method | Path       | Body                     | Response |
|--------|------------|--------------------------|----------|
| GET    | `/`        | —                        | `{"status":"ok"}` |
| POST   | `/predict` | multipart, field `image` | `{"probability":0.87,"result":"malignant"}` |

Classified `malignant` when probability >= 0.231.

## Local development

Backend — **requires Python 3.10 or 3.11** (`tflite-runtime` publishes no 3.12+
wheels):

```bash
cd Skin_cancer_backend-main
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt
python app.py                                    # serves on :8080
```

`tflite-runtime` ships Linux wheels only. On Windows/macOS `pip install` of it
will fail; install `tensorflow-cpu==2.15.0` instead and `app.py` falls back to
`tf.lite.Interpreter` automatically — same graph, same numbers.

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

- **`best_model.keras` does not load on Linux.** It was saved on Windows by
  Keras 2.x, which builds the archive's internal HDF5 paths with
  `os.path.join()`. HDF5 treats `\` as an ordinary name character rather than a
  group separator, so the file holds 279 flat groups called `layers\stem_conv`
  instead of a nested `layers/stem_conv` tree. On Linux the loader looks up
  `layers/stem_conv/vars/0`, finds nothing, and raises *"Layer stem_conv
  expected 1 variables, but received 0 variables during loading"*. Kept only for
  reference. Re-saving must be done **on Windows**, where it still loads.
- **Raw 0–255 input is correct.** The graph begins with `Rescaling(1/255)` and
  `Normalization(mean=[0.485, 0.456, 0.406])`, so preprocessing is inside the
  model. Adding a `/255.0` in `preprocess_image()` would double-scale and
  corrupt every prediction.
- **Version pins are load-bearing.** `tflite-runtime` 2.14 is built against the
  numpy 1.x ABI, and `opencv-python-headless` 5.x requires numpy>=2 — the two
  pins together are what keep the install resolvable.
- **Cold starts.** Free services sleep after 15 min idle; the next request pays
  a startup penalty while the interpreter loads.
- **`KANLayer` weights need explicit `name=`.** HDF5 keys weights by name within
  a layer, so two unnamed `add_weight()` calls collide on save. The `.keras`
  format stored them positionally, which hid this.
