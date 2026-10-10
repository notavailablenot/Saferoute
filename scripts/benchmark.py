"""Latency and throughput benchmarks (UC-02 step 5; report Section 9).

    python scripts/benchmark.py models                 # ONNX engines in-process: 50 warm-up + 500 timed
    python scripts/benchmark.py api --tag native       # running API (uvicorn) on 127.0.0.1:8000
    python scripts/benchmark.py api --tag docker       # same, while the Docker container is up
    python scripts/benchmark.py telemetry --session 3  # FPS / end-to-end latency logged by the GUI

Results: reports/benchmark_<mode>[_<tag>].json
"""
import argparse
import glob
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from saferoute.evaluation.metrics import latency_summary  # noqa: E402


def images(pattern: str, n: int = 50):
    paths = sorted(glob.glob(pattern))[:n]
    if not paths:
        raise SystemExit(f"no images match {pattern}")
    return paths


def env_info() -> dict:
    import onnxruntime as ort
    info = {"python": platform.python_version(), "os": platform.platform(),
            "cpu": platform.processor() or platform.machine(), "onnxruntime": ort.__version__,
            "available_providers": ort.get_available_providers()}
    try:
        import subprocess
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        info["gpu"] = gpu or None
    except Exception:
        info["gpu"] = None
    return info


def save(name: str, rep: dict):
    out = ROOT / "reports" / f"benchmark_{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"written: {out.relative_to(ROOT)}")


def bench_models(a):
    import yaml

    from saferoute.vision.cascade import SignCascade
    from saferoute.vision.classifier_engine import ClassifierEngine
    from saferoute.vision.detector_engine import DetectorEngine
    data = yaml.safe_load((ROOT / "configs/classes.yaml").read_text())["classes"]
    names = [data[k]["name"] for k in sorted(data, key=int)]
    tiers = {data[k]["name"]: data[k]["tier"] for k in data}
    det = DetectorEngine(Path(a.det_model))
    cls = ClassifierEngine(Path(a.cls_model), names)
    cas = SignCascade(det, cls, tiers)
    frames = [cv2.imread(p) for p in images(a.images)]
    crops = [cv2.imread(p)[:, :, ::-1].copy() for p in images(a.crops)]
    rep = {"env": env_info(), "detector_provider": det.provider, "classifier_provider": cls.provider,
           "warmup": a.warmup, "timed": a.n}
    for name, fn, pool in (("detector", lambda x: det.detect(x)[1], frames),
                           ("classifier", lambda x: cls.predict_array(x)["inference_ms"], crops),
                           ("cascade_pipeline", lambda x: cas.run(x)["pipeline_ms"], frames)):
        for i in range(a.warmup):
            fn(pool[i % len(pool)])
        ms = [fn(pool[i % len(pool)]) for i in range(a.n)]
        rep[name] = latency_summary(ms)
        print(name, rep[name])
    save("models", rep)


def bench_api(a):
    import httpx
    c = httpx.Client(base_url=a.url, timeout=10)
    h = c.get("/health").json()
    if h.get("status") != "ready":
        raise SystemExit(f"API not ready: {h}")
    blobs = []
    for p in images(a.images):
        img = cv2.imread(p)
        if img.shape[1] > 960:
            img = cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1])))
        blobs.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
    rt, api, pipe = [], [], []
    for i in range(a.warmup + a.n):
        t0 = time.perf_counter()
        r = c.post("/detect/image", files={"file": ("f.jpg", blobs[i % len(blobs)], "image/jpeg")})
        dt = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        if i >= a.warmup:
            t = r.json()["timings"]
            rt.append(dt)
            api.append(t["api_ms"])
            pipe.append(t["pipeline_ms"])
    rep = {"env": env_info(), "url": a.url, "tag": a.tag, "backend_provider": h.get("detector_provider"),
           "round_trip_client": latency_summary(rt), "api_server_side": latency_summary(api),
           "model_pipeline": latency_summary(pipe),
           "throughput_fps_sequential": round(1000 / float(np.mean(rt)), 1)}
    save(f"api_{a.tag}", rep)


def bench_telemetry(a):
    import httpx
    rows = httpx.get(f"{a.url}/sessions/{a.session}/telemetry", timeout=5).json()
    if not rows:
        raise SystemExit("no telemetry rows for this session")
    fps = [r["fps"] for r in rows]
    e2e = [r["end_to_end_ms"] for r in rows]
    rep = {"session": a.session, "seconds": len(rows),
           "fps_mean": round(float(np.mean(fps)), 2), "fps_min": round(float(np.min(fps)), 2),
           "end_to_end": latency_summary(e2e),
           "first_minute_vs_last_minute_e2e_ms": [round(float(np.mean(e2e[:60])), 2),
                                                   round(float(np.mean(e2e[-60:])), 2)]}
    save(f"telemetry_session{a.session}", rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["models", "api", "telemetry"])
    ap.add_argument("--det-model", default="models/detector_yolo11n.onnx")
    ap.add_argument("--cls-model", default="models/baseline_cnn.onnx")
    ap.add_argument("--images", default="data/processed/signs_v0/images/test/*")
    ap.add_argument("--crops", default="data/processed/crops/test/*/*")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--tag", default="native")
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    {"models": bench_models, "api": bench_api, "telemetry": bench_telemetry}[a.mode](a)


if __name__ == "__main__":
    main()
