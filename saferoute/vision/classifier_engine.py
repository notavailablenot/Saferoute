"""ONNX Runtime engine for the Sprint 1 baseline sign classifier (crop -> class).

Loaded ONCE at API startup (never per request). Preprocessing here must match
the training transforms in saferoute/vision/datasets.py exactly.
"""
import hashlib
import io
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ClassifierEngine:
    def __init__(self, model_path: Path, class_names: list[str], img_size: int = 64):
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = class_names
        self.img_size = img_size
        self.model_hash = sha256_of(model_path)
        self.provider = self.session.get_providers()[0]
        n_out = self.session.get_outputs()[0].shape[-1]
        if isinstance(n_out, int) and n_out != len(class_names):
            raise ValueError(f"model outputs {n_out} classes but {len(class_names)} names were given")

    def _to_tensor(self, im: Image.Image) -> np.ndarray:
        im = im.convert("RGB").resize((self.img_size, self.img_size))
        x = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
        return x.transpose(2, 0, 1)[None]  # NCHW, batch of 1

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        return self._to_tensor(Image.open(io.BytesIO(image_bytes)))

    def predict_array(self, rgb: np.ndarray, top_k: int = 3) -> dict:
        """Classify an RGB uint8 crop (H x W x 3), e.g. a detector box cut from a frame."""
        return self._run(self._to_tensor(Image.fromarray(rgb)), top_k)

    def predict(self, image_bytes: bytes, top_k: int = 3) -> dict:
        return self._run(self.preprocess(image_bytes), top_k)

    def _run(self, x: np.ndarray, top_k: int) -> dict:
        t0 = time.perf_counter()
        logits = self.session.run(None, {self.input_name: x})[0][0]
        infer_ms = (time.perf_counter() - t0) * 1000
        e = np.exp(logits - logits.max())
        probs = e / e.sum()  # softmax applied here, not inside the model
        idx = np.argsort(probs)[::-1][:top_k]
        return {
            "label": self.class_names[int(idx[0])],
            "confidence": float(probs[idx[0]]),
            "top_k": [{"label": self.class_names[int(i)], "confidence": float(probs[i])} for i in idx],
            "inference_ms": round(infer_ms, 3),
        }
