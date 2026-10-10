import importlib
import io
from pathlib import Path

from fastapi.testclient import TestClient
from helpers import tiny_classifier, tiny_detector
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _client(monkeypatch, tmp_path, cls_model, det_model=None):
    monkeypatch.setenv("SAFEROUTE_CLS_MODEL", str(cls_model))
    monkeypatch.setenv("SAFEROUTE_DET_MODEL", str(det_model or tmp_path / "no_detector.onnx"))
    monkeypatch.setenv("SAFEROUTE_CLASSES", str(ROOT / "configs" / "classes.yaml"))
    monkeypatch.setenv("SAFEROUTE_DB_URL", f"sqlite:///{tmp_path / 'api.db'}")
    import saferoute.api.main as main
    importlib.reload(main)
    return TestClient(main.app)


def _img(fmt="PNG", size=(640, 640)):
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, format=fmt)
    return buf.getvalue()


def test_predict_with_model(tmp_path, monkeypatch):
    tiny_classifier(tmp_path / "m.onnx")
    with _client(monkeypatch, tmp_path, tmp_path / "m.onnx") as c:
        h = c.get("/health").json()
        assert h["status"] == "ready" and h["num_classes"] == 12 and h["detector_ready"] is False
        r = c.post("/predict/image", files={"file": ("s.png", _img(size=(120, 90)), "image/png")})
        assert r.status_code == 200
        body = r.json()
        assert len(body["top_k"]) == 3 and 0 <= body["confidence"] <= 1
        assert c.post("/predict/image", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 415
        assert c.post("/predict/image", files={"file": ("b.png", b"garbage", "image/png")}).status_code == 422
        # detector missing: /detect/image degrades to 503 instead of crashing
        assert c.post("/detect/image", files={"file": ("f.png", _img(), "image/png")}).status_code == 503


def test_missing_model_degrades_gracefully(tmp_path, monkeypatch):
    with _client(monkeypatch, tmp_path, tmp_path / "missing.onnx") as c:
        assert c.get("/health").json()["status"] == "not_ready"
        r = c.post("/predict/image", files={"file": ("s.png", _img(), "image/png")})
        assert r.status_code == 503


def test_detect_stateless(tmp_path, monkeypatch):
    tiny_classifier(tmp_path / "c.onnx", favour=4)  # always SPEED_LIMIT_30
    tiny_detector(tmp_path / "d.onnx", [(320, 320, 100, 100, 0.92)])
    with _client(monkeypatch, tmp_path, tmp_path / "c.onnx", tmp_path / "d.onnx") as c:
        assert c.get("/health").json()["detector_ready"] is True
        r = c.post("/detect/image", files={"file": ("f.jpg", _img("JPEG"), "image/jpeg")})
        assert r.status_code == 200
        body = r.json()
        d = body["detections"][0]
        assert d["label"] == "SPEED_LIMIT_30" and d["tier"] == 2
        assert body["alerts"] == [] and body["image"] == {"width": 640, "height": 640}
        assert {"detect_ms", "classify_ms", "pipeline_ms", "api_ms"} <= set(body["timings"])


def test_session_flow_alert_once_and_telemetry(tmp_path, monkeypatch):
    tiny_classifier(tmp_path / "c.onnx", favour=0)  # always STOP
    tiny_detector(tmp_path / "d.onnx", [(320, 320, 100, 100, 0.92)])
    with _client(monkeypatch, tmp_path, tmp_path / "c.onnx", tmp_path / "d.onnx") as c:
        sid = c.post("/sessions", json={"model_version": "test"}).json()["session_id"]
        alerts = []
        for i in range(8):  # an approaching sign (frames grow, so the box grows) -> exactly one alert
            r = c.post(f"/detect/image?session_id={sid}",
                       files={"file": ("f.png", _img(size=(640 + 40 * i, 640 + 40 * i)), "image/png")})
            assert r.status_code == 200
            alerts += r.json()["alerts"]
        assert [a["label"] for a in alerts] == ["STOP"]
        assert c.post(f"/sessions/{sid}/telemetry",
                      json={"fps": 14.2, "inference_ms": 9.5, "end_to_end_ms": 61.0}).status_code == 200
        assert len(c.get(f"/sessions/{sid}/telemetry").json()) == 1
        assert [a["label"] for a in c.get(f"/sessions/{sid}/alerts").json()] == ["STOP"]
        closed = c.patch(f"/sessions/{sid}/close").json()
        assert closed["alerts"] == 1 and closed["end_time"]
        assert c.patch(f"/sessions/{sid}/close").status_code == 409
        assert c.post(f"/detect/image?session_id={sid}",
                      files={"file": ("f.png", _img(), "image/png")}).status_code == 409
        assert c.get("/sessions/999/alerts").status_code == 404
