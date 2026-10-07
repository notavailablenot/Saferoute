"""Export trained models to ONNX.

Baseline CNN:  python scripts/export_onnx.py cnn --weights models/baseline_cnn.pt --out models/baseline_cnn.onnx
YOLO detector: python scripts/export_onnx.py yolo --weights runs/detect/train/weights/best.pt --imgsz 640 [--half]
"""
import argparse
from pathlib import Path

import numpy as np


def export_cnn(weights: str, out: str):
    import onnxruntime as ort
    import torch

    from saferoute.vision.baseline_cnn import BaselineSignCNN
    ckpt = torch.load(weights, map_location="cpu")
    model = BaselineSignCNN(num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    dummy = torch.randn(1, 3, 64, 64)
    torch.onnx.export(model, dummy, out, input_names=["input"], output_names=["logits"], opset_version=18)
    # parity check: ONNX output must match PyTorch
    ref = model(dummy).detach().numpy()
    got = ort.InferenceSession(out, providers=["CPUExecutionProvider"]).run(None, {"input": dummy.numpy()})[0]
    print(f"exported {out}; max |torch - onnx| = {np.abs(ref - got).max():.2e}")
    assert np.allclose(ref, got, atol=1e-4)
    # PyTorch 2.x's exporter writes the weights to a separate "<out>.data" file. Merge them back
    # so the model is ONE self-contained file (safe to commit, share, and COPY into Docker).
    import onnx
    merged = onnx.load(out)  # also reads <out>.data if it exists
    onnx.save(merged, out)
    Path(out + ".data").unlink(missing_ok=True)
    print(f"saved self-contained {out} ({Path(out).stat().st_size / 1e6:.1f} MB)")


def export_yolo(weights: str, imgsz: int, half: bool):
    from ultralytics import YOLO
    path = YOLO(weights).export(format="onnx", imgsz=imgsz, simplify=True, half=half)
    print(f"exported {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["cnn", "yolo"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="models/baseline_cnn.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true")
    a = ap.parse_args()
    export_cnn(a.weights, a.out) if a.kind == "cnn" else export_yolo(a.weights, a.imgsz, a.half)
