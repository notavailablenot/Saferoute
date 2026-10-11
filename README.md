# SafeRoute: Real-Time Traffic Sign Detection and Driver Warning System

CPE178P Foundations of AI, FOPI01, Group 01 (Avengoza, Hembrador, Ledda), Mapúa University.

Offline three-tier application:

| Tier | Component | Technology |
|---|---|---|
| Presentation | Desktop client: live Drive tab (webcam / video file) and Image upload tab | PyQt6, OpenCV |
| Logic | REST API, two-stage cascade (YOLO11n detector + CNN classifier), ByteTrack-style tracking, alert manager | FastAPI, ONNX Runtime, NumPy |
| Data Access | Sessions, alerts, per-second telemetry, model registry, evaluation runs | SQLite, SQLAlchemy |

```
GUI (PyQt6) --JPEG frame--> POST /detect/image?session_id=N --> DetectorEngine (YOLO11n ONNX)
     ^                                                      --> crop --> ClassifierEngine (CNN ONNX)
     |                                                      --> ByteTracker --> AlertManager --> SQLite
     +----------- JSON: boxes, labels, tiers, new alerts, timings <-------------------------------+
```

## 1. Setup (WSL2 Ubuntu, Python 3.12)
```bash
uv venv --python 3.12 --seed .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```
## 1.1 Troubleshoot & Known Issues
# Codespace: Missing System Libraries (`ImportError: libGL.so.1` or `libEGL.so.1`)
When running tests (`pytest`) in a fresh GitHub Codespace, you may encounter import errors related to OpenCV (`cv2`) or PyQt6 lacking graphics and display drivers in the headless environment.
**Fix:** Install the missing system libraries by running the following in the Codespace terminal:
```bash
sudo apt-get update
sudo apt-get install -y libgl1 libxkbcommon-x11-0 libxkbcommon0 libegl1 libfontconfig1 libglib2.0-0
```
## 2. Models
| File | What | How it was made |
|---|---|---|
| `models/baseline_cnn.onnx` | 10-class sign classifier (64 x 64 crops) | `scripts/train_baseline.py`, `scripts/export_onnx.py` |
| `models/detector_yolo11n.onnx` / `.pt` | 1-class traffic-sign detector (640 x 640) | Ultralytics YOLO11n, `scripts/export_detector.py` (with parity check) |

## 3. Run the backend
Docker (offline after the first build):
```bash
docker compose up -d --build
docker compose ps            # STATUS shows (healthy)
curl http://127.0.0.1:8000/health
```
Or natively: `uvicorn saferoute.api.main:app --host 127.0.0.1 --port 8000`.
API docs (Swagger UI): http://127.0.0.1:8000/docs

## 4. Run the GUI
```bash
pip install -r requirements-gui.txt      # if not installed through pyproject
python -m saferoute.gui
```
- **Drive tab:** Start webcam or Open video file. Signs are tracked, each physical sign raises one alert (tier colour, audio cue, speed-limit badge), and FPS and latency are shown in the status bar and logged per second.
- **Image upload tab:** upload a JPEG or PNG and see boxes, classes, confidences and timings.
- The webcam is not visible inside WSL2. For the webcam, run the GUI with Windows Python (`pip install -r requirements-gui.txt`, then `python -m saferoute.gui`); video files work in both.

## 5. Evaluate and benchmark
```bash
python scripts/evaluate.py classifier         # accuracy, P, R, F1, confusion matrix (GTSRB test)
python scripts/evaluate.py detector           # P, R, F1, mAP@0.5, mAP@0.5:0.95 (ONNX, test split)
python scripts/label_crops.py                 # label real test-photo signs once
python scripts/evaluate.py cascade            # classifier accuracy on real road photos
python scripts/benchmark.py models            # 50 warm-up + 500 timed runs, p50/p95/p99
python scripts/benchmark.py api --tag docker  # API response time against the running container
bash scripts/measure_docker.sh                # image size and container startup time
```
Results are written to `reports/`.

## 6. Configuration (environment variables)
| Variable | Default |
|---|---|
| `SAFEROUTE_CLS_MODEL` | `models/baseline_cnn.onnx` |
| `SAFEROUTE_DET_MODEL` | `models/detector_yolo11n.onnx` |
| `SAFEROUTE_DB_URL` | `sqlite:///saferoute.db` |
| `SAFEROUTE_MIN_CLS_CONF` | `0.8` (below this the sign is reported as a generic "Sign ahead") |
| `SAFEROUTE_DET_CONF` | `0.35` |
| `SAFEROUTE_API` (GUI) | `http://127.0.0.1:8000` |

## 7. Repository layout
```
configs/classes.yaml        10 classes and alert tiers
saferoute/vision/           detector_engine, classifier_engine, cascade, tracker, baseline_cnn, datasets
saferoute/api/main.py       FastAPI application
saferoute/data_access/      SQLAlchemy models and repositories
saferoute/evaluation/       metrics (accuracy, P, R, F1, AP, latency)
saferoute/gui/              PyQt6 client (app, api_client, overlay)
scripts/                    data import, audit, training, export, evaluation, benchmark, Docker metrics
notebooks/                  01 dataset audit, 02 baseline CNN, 03 YOLO detector
tests/                      pytest suite (engines, tracker, API, GUI smoke test, metrics, data tools)
Dockerfile, docker-compose.yml, requirements*.txt
```
Datasets and training runs are not committed; see `docs/dataset_card.md`.
