"""SafeRoute Logic Tier: FastAPI service (Sprint 1 endpoints).

Run:  uvicorn saferoute.api.main:app --host 127.0.0.1 --port 8000
Docs: http://127.0.0.1:8000/docs  (auto-generated Swagger UI)
"""
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from saferoute.vision.classifier_engine import ClassifierEngine
from saferoute.data_access.repositories import SessionRepo, init_db

MODEL_PATH = Path(os.getenv("SAFEROUTE_CLS_MODEL", "models/baseline_cnn.onnx"))
CLASSES_YAML = Path(os.getenv("SAFEROUTE_CLASSES", "configs/classes.yaml"))
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png"}

state: dict = {"engine": None, "load_error": None}


def load_class_names(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text())["classes"]
    return [data[k]["name"] for k in sorted(data, key=int)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the model once at startup (Sprint 2 pitfall: "slow inference").
    try:
        state["engine"] = ClassifierEngine(MODEL_PATH, load_class_names(CLASSES_YAML))
    except Exception as exc:  # service still starts; /health reports why it is not ready
        state["load_error"] = f"{type(exc).__name__}: {exc}"
    yield
    state["engine"] = None


app = FastAPI(title="SafeRoute Inference API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost", "http://127.0.0.1"],
                   allow_methods=["*"], allow_headers=["*"])

#sessions update
SessionFactory = init_db

def get_db():
    """Dependencies to yield a database session per request"""
    with SessionFactory() as db:
        yield db

class SessionCreate(BaseModel): 
    model_version: str

@app.get("/health")
def health():
    eng = state["engine"]
    if eng is None:
        return {"status": "not_ready", "reason": state["load_error"]}
    return {"status": "ready", "model_path": str(MODEL_PATH), "model_sha256": eng.model_hash,
            "execution_provider": eng.provider, "num_classes": len(eng.class_names)}


@app.get("/models")
def models():
    return {"classifier": str(MODEL_PATH) if state["engine"] else None,
            "detector": None}  # YOLO ONNX detector is wired in Sprint 2


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    eng = state["engine"]
    if eng is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {state['load_error']}")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="only JPEG or PNG images are accepted")
    data = await file.read()
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="empty file or larger than 5 MB")
    t0 = time.perf_counter()
    try:
        result = eng.predict(data)
    except Exception as exc:  # undecodable image etc.: fail gracefully, never crash
        raise HTTPException(status_code=422, detail=f"could not process image: {exc}") from exc
    result["api_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    return result

@app.post("/sessions")
def create_session(data: SessionCreate, db = Depends(get_db)):
    """Stub endpoint: Starts a driving session in the database"""
    repo = SessionRepo(db)
    new_session = repo.create(model_version=data.model_version)

    return {
        "Status": "Session started",
        "session_id": new_session.id,
        "model_version": new_session.model_version
    }
