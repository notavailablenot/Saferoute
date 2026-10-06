import importlib
import io
from pathlib import Path

import numpy as np
import onnx
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _tiny_classifier(path: Path, n_classes: int = 16):
    """Stand-in ONNX model with the same I/O contract as the baseline CNN: [1,3,64,64] -> [1,16]."""
    w = numpy_helper.from_array(np.random.rand(3, n_classes).astype(np.float32), "W")
    g = helper.make_graph(
        [helper.make_node("GlobalAveragePool", ["input"], ["gap"]),
         helper.make_node("Flatten", ["gap"], ["flat"]),
         helper.make_node("MatMul", ["flat", "W"], ["logits"])],
        "tiny", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, n_classes])], [w])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)


def _client(monkeypatch, model_path):
    monkeypatch.setenv("SAFEROUTE_CLS_MODEL", str(model_path))
    monkeypatch.setenv("SAFEROUTE_CLASSES", str(ROOT / "configs" / "classes.yaml"))
    import saferoute.api.main as main
    importlib.reload(main)
    return TestClient(main.app)


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (120, 90), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_predict_with_model(tmp_path, monkeypatch):
    _tiny_classifier(tmp_path / "m.onnx")
    with _client(monkeypatch, tmp_path / "m.onnx") as c:
        h = c.get("/health").json()
        assert h["status"] == "ready" and h["num_classes"] == 16
        r = c.post("/predict/image", files={"file": ("s.png", _png(), "image/png")})
        assert r.status_code == 200
        body = r.json()
        assert len(body["top_k"]) == 3 and 0 <= body["confidence"] <= 1
        assert c.post("/predict/image", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 415
        assert c.post("/predict/image", files={"file": ("b.png", b"garbage", "image/png")}).status_code == 422


def test_missing_model_degrades_gracefully(tmp_path, monkeypatch):
    with _client(monkeypatch, tmp_path / "missing.onnx") as c:
        assert c.get("/health").json()["status"] == "not_ready"
        r = c.post("/predict/image", files={"file": ("s.png", _png(), "image/png")})
        assert r.status_code == 503
