# LightMamba-ASL Frontend

React (Create React App) frontend for the LightMamba-ASL video-based ASL recognition system.

---

## Requirements

- Node.js >= 18.x
- npm >= 9.x
- Flask backend running on `http://127.0.0.1:5000`

---

## Installation

```bash
cd lightmamba-asl-frontend
npm install
```

---

## Environment Configuration

The `.env` file is already included:

```
REACT_APP_API_BASE_URL=http://127.0.0.1:5000
```

This uses the CRA `REACT_APP_` prefix. Do NOT use `VITE_` or `import.meta.env`.

---

## Running

```bash
npm start
```

Opens at `http://localhost:3000`.

---

## Pages

| Route        | Page                  |
|--------------|-----------------------|
| `/`          | Dashboard             |
| `/recognize` | Video Recognition     |
| `/live`      | Live Recognition      |
| `/skeleton`  | Skeleton Tracking     |
| `/model`     | Model Architecture    |
| `/results`   | Research Results      |
| `/about`     | About Project         |

---

## MediaPipe Setup

MediaPipe Tasks Vision is loaded from CDN at runtime:

```
https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm
```

Models are fetched from Google Storage on first use. An internet connection is required for the first load. After that, the browser caches the WASM and model files.

---

## Video Upload

- Only `.mp4` files are accepted
- The **original MP4** is sent to the Flask backend for prediction
- The skeleton overlay is a separate browser-side visualization — it does NOT affect the backend prediction

---

## Webcam Permissions

Chrome/Edge: Allow camera when prompted.  
If denied: go to `chrome://settings/content/camera` and allow `localhost`.

---

## Live Recognition — WebM Note

Most browsers record in `video/webm` format. The current backend accepts only `.mp4`.  
If you record a gesture and see a format error, use the **Video Upload** page with an MP4 file instead.  
WebM support can be added to the backend later.

---

## CORS Troubleshooting

If you see CORS errors in the browser console:

1. Ensure Flask is running: `python app.py`
2. Ensure `flask-cors` is installed: `pip install flask-cors`
3. The backend uses `CORS(app)` which allows all origins by default

---

## Backend Offline

The frontend degrades gracefully when the backend is offline:
- Sidebar shows **Backend Offline**
- Classes fall back to the hardcoded list
- Model info shows unavailable
- Prediction buttons remain visible but will show an error on submit

---

## Results Page

Training curves, confusion matrix, and metrics are fetched from:

```
GET http://127.0.0.1:5000/outputs/metrics/training_history.json
GET http://127.0.0.1:5000/outputs/metrics/test_metrics.json
GET http://127.0.0.1:5000/outputs/confusion_matrix/confusion_matrix.png
```

These are only available after running:

```bash
python -m backend.training.train
python -m backend.evaluation.evaluate
```

---

## Build for Production

```bash
npm run build
```

Outputs to `build/`. Can be served with any static file server.
