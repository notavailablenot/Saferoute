"""SafeRoute Logic Tier: FastAPI service.

Run (dev):    uvicorn saferoute.api.main:app --host 127.0.0.1 --port 8000
Run (Docker): docker compose up -d --build
Docs:         http://127.0.0.1:8000/docs  (auto-generated Swagger UI)

Endpoints
  GET   /health                      readiness, model hashes, execution providers
  GET   /models                      loaded model files
  POST  /predict/image               classifier only (one sign crop)            [Sprint 1]
  POST  /detect/image                full cascade on a frame; with ?session_id=
                                     also tracks signs and issues driver alerts  [Sprint 2]
  POST  /sessions                    start a driving session                    [Sprint 1, Angelo]
  PATCH /sessions/{id}/close         stop a session
  GET   /sessions/{id}/alerts        alerts logged for a session
  POST  /sessions/{id}/telemetry     per-second FPS / latency from the GUI
  GET   /sessions/{id}/telemetry     telemetry rows for a session
"""
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from saferoute.data_access.models import Alert, PerfMetric, Session
from saferoute.data_access.repositories import AlertRepo, SessionRepo, init_db
from saferoute.vision.cascade import GENERIC, SignCascade
from saferoute.vision.classifier_engine import ClassifierEngine
from saferoute.vision.detector_engine import DetectorEngine
from saferoute.vision.tracker import SessionPipeline

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png"}

log = logging.getLogger("saferoute.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

state: dict = {"engine": None, "detector": None, "cascade": None, "load_error": None,
               "det_error": None, "db": None, "pipelines": {}, "tiers": {}, "started": None}
_lock = threading.Lock()


def cfg() -> dict:
    """Read configuration from the environment at startup (not at import time)."""
    return {
        "cls_model": Path(os.getenv("SAFEROUTE_CLS_MODEL", "models/baseline_cnn.onnx")),
        "det_model": Path(os.getenv("SAFEROUTE_DET_MODEL", "models/detector_yolo11n.onnx")),
        "classes": Path(os.getenv("SAFEROUTE_CLASSES", "configs/classes.yaml")),
        "db_url": os.getenv("SAFEROUTE_DB_URL", "sqlite:///saferoute.db"),
        "min_cls_conf": float(os.getenv("SAFEROUTE_MIN_CLS_CONF", "0.8")),
        "det_conf": float(os.getenv("SAFEROUTE_DET_CONF", "0.35")),
        "cooldown_s": float(os.getenv("SAFEROUTE_COOLDOWN_S", "5")),
    }


def load_classes(path: Path) -> tuple[list[str], dict[str, int]]:
    data = yaml.safe_load(path.read_text())["classes"]
    keys = sorted(data, key=int)
    names = [data[k]["name"] for k in keys]
    tiers = {data[k]["name"]: int(data[k].get("tier", 3)) for k in keys}
    tiers[GENERIC] = 3
    return names, tiers  # names include the NOT_SIGN rejection class (tier 0, never alerted)


@asynccontextmanager
async def lifespan(app: FastAPI):
    c = cfg()
    state.update(cfg=c, pipelines={}, started=time.time(), load_error=None, det_error=None,
                 engine=None, detector=None, cascade=None)
    state["db"] = init_db(c["db_url"])
    try:  # models are preloaded once at startup (pitfall: slow inference)
        names, state["tiers"] = load_classes(c["classes"])
        state["engine"] = ClassifierEngine(c["cls_model"], names)
    except Exception as exc:  # service still starts; /health reports why it is not ready
        state["load_error"] = f"{type(exc).__name__}: {exc}"
        log.error("classifier not loaded: %s", state["load_error"])
    try:
        state["detector"] = DetectorEngine(c["det_model"], conf=c["det_conf"])
        if state["engine"] is not None:
            state["cascade"] = SignCascade(state["detector"], state["engine"], state["tiers"],
                                           min_cls_conf=c["min_cls_conf"])
    except Exception as exc:
        state["det_error"] = f"{type(exc).__name__}: {exc}"
        log.error("detector not loaded: %s", state["det_error"])
    log.info("startup finished in %.2f s", time.time() - state["started"])
    yield
    state.update(engine=None, detector=None, cascade=None, pipelines={})


app = FastAPI(title="SafeRoute API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost", "http://127.0.0.1"],
                   allow_methods=["*"], allow_headers=["*"])


def get_db():
    """Dependency: one database session per request."""
    with state["db"]() as db:
        yield db


class SessionCreate(BaseModel):
    model_version: str = Field(..., min_length=1, examples=["cascade-yolo11n+cnn-v1"])


class TelemetryIn(BaseModel):
    fps: float = Field(..., ge=0)
    inference_ms: float = Field(..., ge=0)
    end_to_end_ms: float = Field(..., ge=0)


async def read_image(file: UploadFile) -> tuple[bytes, np.ndarray]:
    """Validate an upload and decode it to a BGR array (415 / 413 / 422 on bad input)."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="only JPEG or PNG images are accepted")
    data = await file.read()
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="empty file or larger than 5 MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="could not decode image")
    return data, img


def open_session(db, session_id: int) -> Session:
    s = db.get(Session, session_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    if s.end_time is not None:
        raise HTTPException(status_code=409, detail=f"session {session_id} is already closed")
    return s


@app.get("/health")
def health():
    eng, det = state["engine"], state["detector"]
    body = {"status": "ready" if eng is not None else "not_ready",
            "detector_ready": det is not None,
            "uptime_s": round(time.time() - state["started"], 1) if state["started"] else 0}
    if eng is None:
        body["reason"] = state["load_error"]
        return body
    body.update(model_path=str(state["cfg"]["cls_model"]), model_sha256=eng.model_hash,
                execution_provider=eng.provider, num_classes=len(eng.class_names))
    if det is not None:
        body.update(detector_path=str(state["cfg"]["det_model"]), detector_sha256=det.model_hash,
                    detector_provider=det.provider)
    else:
        body["detector_reason"] = state["det_error"]
    return body


@app.get("/models")
def models():
    c = state.get("cfg") or cfg()
    return {"classifier": str(c["cls_model"]) if state["engine"] else None,
            "detector": str(c["det_model"]) if state["detector"] else None,
            "min_classifier_confidence": c["min_cls_conf"]}


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Classify one already-cropped sign image (Sprint 1 classifier endpoint)."""
    eng = state["engine"]
    if eng is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {state['load_error']}")
    data, _ = await read_image(file)
    t0 = time.perf_counter()
    try:
        result = eng.predict(data)
    except Exception as exc:  # fail gracefully, never crash
        raise HTTPException(status_code=422, detail=f"could not process image: {exc}") from exc
    result["api_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    return result


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...),
                       session_id: int | None = Query(None, description="track and alert within this session"),
                       db=Depends(get_db)):
    """Run the full cascade on one frame. Without session_id: stateless detection (image upload).
    With session_id: signs are tracked across frames and new driver alerts are returned and logged."""
    t0 = time.perf_counter()
    cas = state["cascade"]
    if cas is None:
        reason = state["det_error"] or state["load_error"]
        raise HTTPException(status_code=503, detail=f"detector not loaded: {reason}")
    _, img = await read_image(file)
    if session_id is not None:
        open_session(db, session_id)
    result = cas.run(img)
    alerts = []
    if session_id is not None:
        with _lock:
            pipe = state["pipelines"].setdefault(
                session_id, SessionPipeline(state["tiers"], state["cfg"]["cooldown_s"]))
            result["detections"], alerts = pipe.step(result["detections"], time.monotonic())
        repo = AlertRepo(db)
        for a in alerts:
            repo.save(Alert(session_id=session_id, track_id=a["track_id"], sign_class=a["label"],
                            tier=a["tier"], confidence=a["confidence"]))
    api_ms = round((time.perf_counter() - t0) * 1000, 3)
    log.info("detect session=%s signs=%d alerts=%d api_ms=%.1f", session_id,
             len(result["detections"]), len(alerts), api_ms)
    return {"image": {"width": int(img.shape[1]), "height": int(img.shape[0])},
            "detections": result["detections"], "rejected": result.get("rejected", 0), "alerts": alerts,
            "timings": {"detect_ms": result["detect_ms"], "classify_ms": result["classify_ms"],
                        "pipeline_ms": result["pipeline_ms"], "api_ms": api_ms}}


@app.post("/sessions")
def create_session(data: SessionCreate, db=Depends(get_db)):
    """Start a driving session in the database."""
    s = SessionRepo(db).create(model_version=data.model_version)
    return {"Status": "Session started", "session_id": s.id, "model_version": s.model_version}


@app.patch("/sessions/{session_id}/close")
def close_session(session_id: int, db=Depends(get_db)):
    """Stop a session: sets end_time and releases its tracker state."""
    s = open_session(db, session_id)
    SessionRepo(db).close(s.id)
    with _lock:
        state["pipelines"].pop(session_id, None)
    return {"Status": "Session closed", "session_id": s.id,
            "end_time": s.end_time.isoformat() if s.end_time else None,
            "alerts": len(AlertRepo(db).list(s.id))}


@app.get("/sessions/{session_id}/alerts")
def session_alerts(session_id: int, db=Depends(get_db)):
    if db.get(Session, session_id) is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    return [{"id": a.id, "track_id": a.track_id, "label": a.sign_class, "tier": a.tier,
             "confidence": a.confidence, "timestamp": a.timestamp.isoformat()}
            for a in AlertRepo(db).list(session_id)]


@app.post("/sessions/{session_id}/telemetry")
def post_telemetry(session_id: int, t: TelemetryIn, db=Depends(get_db)):
    open_session(db, session_id)
    SessionRepo(db).save_telemetry([PerfMetric(session_id=session_id, fps=t.fps,
                                               inference_ms=t.inference_ms,
                                               end_to_end_ms=t.end_to_end_ms,
                                               timestamp=datetime.now(timezone.utc))])
    return {"saved": True}


@app.get("/sessions/{session_id}/telemetry")
def get_telemetry(session_id: int, db=Depends(get_db)):
    s = db.get(Session, session_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    return [{"timestamp": p.timestamp.isoformat(), "fps": p.fps, "inference_ms": p.inference_ms,
             "end_to_end_ms": p.end_to_end_ms} for p in s.telemetry]
