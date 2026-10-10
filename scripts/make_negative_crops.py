"""Build the NOT_SIGN rejection class for the crop classifier (hard-negative mining).

Problem found in Sprint 2 testing: the classifier was trained only on traffic signs, so it must
pick one of the 10 sign classes for ANY crop. When the detector fired on a billboard, the crop
was confidently labelled YIELD and the driver got a false alert. Giving the classifier a
"not a sign" option lets the cascade reject such crops.

Negatives come from the PauloLab photos, split by split (no leakage into val/test):
  1. hard negatives: detector boxes (conf >= 0.10) that do not overlap any labelled sign
     (IoU < 0.1), i.e. the detector's own false positives on real road scenes;
  2. random negatives: background boxes with the size of real signs in that split, placed
     where they do not overlap any labelled sign (IoU < 0.05).
Crops use the same margin as the cascade (saferoute.vision.cascade.SignCascade.crop).

    python scripts/make_negative_crops.py
Output: data/processed/crops/{train,val,test}/NOT_SIGN/*.png and reports/negative_crops.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from saferoute.evaluation.metrics import box_iou  # noqa: E402
from saferoute.vision.cascade import SignCascade  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def gt_boxes(label: Path, w: int, h: int) -> np.ndarray:
    rows = []
    if label.exists():
        for ln in label.read_text().splitlines():
            p = ln.split()
            if len(p) >= 5:
                cx, cy, bw, bh = map(float, p[1:5])
                rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, float).reshape(-1, 4)


def random_boxes(rng, w, h, sizes, gts, n, max_iou=0.05, tries=50):
    out = []
    for _ in range(n * tries):
        if len(out) >= n:
            break
        bw, bh = sizes[rng.randrange(len(sizes))]
        s = rng.uniform(0.7, 1.6)
        bw, bh = max(16.0, bw * s), max(16.0, bh * s)
        if bw >= w or bh >= h:
            continue
        x1, y1 = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
        b = np.array([[x1, y1, x1 + bw, y1 + bh]])
        if len(gts) and box_iou(b, gts).max() >= max_iou:
            continue
        out.append(b[0].tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/signs_v0")
    ap.add_argument("--crops", default="data/processed/crops")
    ap.add_argument("--det-model", default="models/detector_yolo11n.onnx")
    ap.add_argument("--random-per-image", type=int, default=2)
    ap.add_argument("--hard-conf", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    from saferoute.vision.detector_engine import DetectorEngine
    det = DetectorEngine(Path(a.det_model), conf=a.hard_conf, iou=0.5)
    cutter = SignCascade(None, None, {})
    rng = random.Random(a.seed)
    stats = {}
    for split in ("train", "val", "test"):
        imgs = sorted(p for p in (Path(a.data) / "images" / split).glob("*") if p.suffix.lower() in IMG_EXT)
        out = Path(a.crops) / split / "NOT_SIGN"
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("*.png"):
            old.unlink()
        loaded, sizes = [], []
        for p in imgs:
            img = cv2.imread(str(p))
            if img is None:
                continue
            g = gt_boxes(Path(a.data) / "labels" / split / (p.stem + ".txt"), img.shape[1], img.shape[0])
            sizes += [(b[2] - b[0], b[3] - b[1]) for b in g]
            loaded.append((p, img, g))
        sizes = sizes or [(48.0, 48.0)]
        n_hard = n_rand = 0
        for p, img, g in loaded:
            dets, _ = det.detect(img)
            hard = [d["box"] for d in dets
                    if not len(g) or box_iou(np.array([d["box"]]), g).max() < 0.1]
            rand = random_boxes(rng, img.shape[1], img.shape[0], sizes, g, a.random_per_image)
            for kind, boxes in (("hard", hard), ("rand", rand)):
                for k, b in enumerate(boxes):
                    crop = cutter.crop(img, b)
                    if crop is None:
                        continue
                    cv2.imwrite(str(out / f"{p.stem}_{kind}{k}.png"), np.ascontiguousarray(crop[:, :, ::-1]))
                    n_hard += kind == "hard"
                    n_rand += kind == "rand"
        stats[split] = {"images": len(loaded), "hard_negatives": n_hard, "random_negatives": n_rand,
                        "total": n_hard + n_rand}
        print(split, stats[split])
    Path("reports").mkdir(exist_ok=True)
    Path("reports/negative_crops.json").write_text(json.dumps(stats, indent=2))
    print("written: reports/negative_crops.json")


if __name__ == "__main__":
    main()
