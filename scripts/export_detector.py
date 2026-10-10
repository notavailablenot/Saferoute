"""Export the Sprint 1 YOLO11n detector to ONNX and verify it against Ultralytics.

Usage (in WSL, venv active, from the repo root):
    python scripts/export_detector.py
    python scripts/export_detector.py --weights runs/detect/runs/detect/sprint1_signs/weights/best.pt

Steps
  1. Ultralytics exports best.pt to ONNX (static 1x3x640x640, opset 17, simplified).
  2. The file is copied to models/detector_yolo11n.onnx (single file, no external .data).
  3. Parity check A (pre/post-processing): Ultralytics running the ONNX file vs our
     DetectorEngine on the same file. Boxes must match within 1 px.
  4. Parity check B (export fidelity): Ultralytics running best.pt (PyTorch) vs our
     DetectorEngine. Ultralytics' PyTorch path pads to a multiple of 32 instead of a 640
     square, so boxes may differ slightly; we require IoU >= 0.9 per matched box.
  5. Results are written to reports/detector_onnx_parity.json.
"""
import argparse
import glob
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def compare(ref_boxes, ours):
    """Match each reference box to the best of ours; return (count_equal, ious, max_px_diff)."""
    ious, diffs = [], []
    for rb in ref_boxes:
        best = max(ours, key=lambda o: iou(rb, o["box"]), default=None)
        if best is None:
            ious.append(0.0)
            continue
        ious.append(iou(rb, best["box"]))
        diffs.append(float(np.max(np.abs(np.array(rb) - np.array(best["box"])))))
    return len(ref_boxes) == len(ours), ious, max(diffs, default=0.0)


def main():
    from ultralytics import YOLO

    from saferoute.vision.detector_engine import DetectorEngine

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="best.pt (default: newest under runs/)")
    ap.add_argument("--images", default="data/processed/signs_v0/images/test")
    ap.add_argument("--out", default="models/detector_yolo11n.onnx")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()

    weights = a.weights or max(glob.glob("runs/**/weights/best.pt", recursive=True),
                               key=lambda p: Path(p).stat().st_mtime)
    print(f"weights: {weights}")
    exported = Path(YOLO(weights).export(format="onnx", imgsz=640, opset=17, simplify=True,
                                         dynamic=False))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import onnx  # load + save merges any external .data weights into ONE self-contained file
    onnx.save(onnx.load(str(exported)), str(out))
    onnx.checker.check_model(str(out))
    print(f"ONNX saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    pt_copy = out.with_suffix(".pt")
    shutil.copy(weights, pt_copy)  # keep the PyTorch weights with the codebase as well
    print(f"PyTorch weights copied: {pt_copy}")

    imgs = sorted(glob.glob(f"{a.images}/*"))[: a.n]
    assert imgs, f"no images found in {a.images}"
    onnx_ul, pt_ul = YOLO(str(out), task="detect"), YOLO(weights)
    ours = DetectorEngine(out, conf=0.25, iou=0.7)  # Ultralytics predict defaults
    rep = {"weights": weights, "onnx": str(out), "images": len(imgs), "A_pre_post": {}, "B_pytorch": {}}
    for tag, model, px_tol, iou_tol in (("A_pre_post", onnx_ul, 1.0, 0.99),
                                        ("B_pytorch", pt_ul, None, 0.90)):
        same_count, all_ious, worst_px = 0, [], 0.0
        for p in imgs:
            img = cv2.imread(p)
            r = model.predict(img, conf=0.25, iou=0.7, imgsz=640, verbose=False)[0]
            ref = r.boxes.xyxy.cpu().numpy().tolist()
            dets, _ = ours.detect(img)
            eq, ious, px = compare(ref, dets)
            same_count += eq
            all_ious += ious
            worst_px = max(worst_px, px)
        ok = (min(all_ious, default=1.0) >= iou_tol) and (px_tol is None or worst_px <= px_tol)
        rep[tag] = {"same_box_count": f"{same_count}/{len(imgs)}",
                    "min_iou": round(min(all_ious, default=1.0), 4),
                    "mean_iou": round(float(np.mean(all_ious)) if all_ious else 1.0, 4),
                    "max_px_diff": round(worst_px, 3), "pass": bool(ok)}
        print(tag, rep[tag])
    Path("reports").mkdir(exist_ok=True)
    Path("reports/detector_onnx_parity.json").write_text(json.dumps(rep, indent=2))
    print("written: reports/detector_onnx_parity.json")


if __name__ == "__main__":
    main()
