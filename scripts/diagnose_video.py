"""Diagnose false alerts on a drive video: what does each stage see for every detection?

    python scripts/diagnose_video.py sample_drive.mp4            # every 5th frame
    python scripts/diagnose_video.py sample_drive.mp4 --every 3

Writes reports/diagnose/:
  detections.csv      one row per detection: frame, det_conf, box size/aspect/area share,
                      classifier top-1/top-2 and their margin
  crops_sheet.png     contact sheet of every detected crop with its numbers (open it in VS Code)
  frame_XXXXX.jpg     frames with boxes (first 30 frames that have detections)
Uses the ONNX engines directly (no API needed), with the same settings as the backend.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from saferoute.vision.cascade import SignCascade  # noqa: E402
from saferoute.vision.classifier_engine import ClassifierEngine  # noqa: E402
from saferoute.vision.detector_engine import DetectorEngine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--out", default="reports/diagnose")
    a = ap.parse_args()
    data = yaml.safe_load((ROOT / "configs/classes.yaml").read_text())["classes"]
    names = [data[k]["name"] for k in sorted(data, key=int)]
    det = DetectorEngine(ROOT / "models/detector_yolo11n.onnx", conf=a.det_conf)
    cls = ClassifierEngine(ROOT / "models/baseline_cnn.onnx", names)
    cas = SignCascade(det, cls, {})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(a.video)
    rows, thumbs, saved, idx = [], [], 0, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % a.every:
            continue
        H, W = frame.shape[:2]
        dets, _ = det.detect(frame)
        vis = frame.copy()
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            bw, bh = x2 - x1, y2 - y1
            crop = cas.crop(frame, d["box"])
            if crop is None:
                continue
            p = cls.predict_array(crop)
            t1, t2 = p["top_k"][0], p["top_k"][1]
            row = {"frame": idx, "det_conf": round(d["det_conf"], 3), "w": round(bw), "h": round(bh),
                   "aspect_w_over_h": round(bw / max(bh, 1), 2),
                   "area_share": round(bw * bh / (W * H), 4),
                   "top1": t1["label"], "top1_conf": round(t1["confidence"], 3),
                   "top2": t2["label"], "top2_conf": round(t2["confidence"], 3),
                   "margin": round(t1["confidence"] - t2["confidence"], 3)}
            rows.append(row)
            th = cv2.resize(np.ascontiguousarray(crop[:, :, ::-1]), (128, 128))
            panel = np.full((128 + 54, 128, 3), 255, np.uint8)
            panel[:128] = th
            for k, txt in enumerate([f"f{idx} det {row['det_conf']:.2f}",
                                     f"{t1['label'][:13]} {t1['confidence']:.2f}",
                                     f"a{row['aspect_w_over_h']} s{row['area_share']:.3f}"]):
                cv2.putText(panel, txt, (2, 142 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
            thumbs.append(panel)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.putText(vis, f"{t1['label']} {t1['confidence']:.2f} det {d['det_conf']:.2f}",
                        (int(x1), max(20, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if dets and saved < 30:
            cv2.imwrite(str(out / f"frame_{idx:05d}.jpg"), vis)
            saved += 1
    cap.release()

    with open(out / "detections.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if thumbs:
        cols = 10
        thumbs = thumbs[:200]
        while len(thumbs) % cols:
            thumbs.append(np.full_like(thumbs[0], 255))
        grid = np.vstack([np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)])
        cv2.imwrite(str(out / "crops_sheet.png"), grid)
    print(f"frames read: {idx + 1}, detections: {len(rows)}, frames saved: {saved}")
    if rows:
        for k in ("det_conf", "aspect_w_over_h", "area_share", "top1_conf", "margin"):
            v = np.array([r[k] for r in rows], float)
            print(f"{k:16s} min {v.min():.3f}  median {np.median(v):.3f}  max {v.max():.3f}")
        labels, counts = np.unique([r["top1"] for r in rows], return_counts=True)
        print("top-1 labels:", dict(zip(labels.tolist(), counts.tolist())))
    print(f"written: {out}/detections.csv, {out}/crops_sheet.png")


if __name__ == "__main__":
    main()
