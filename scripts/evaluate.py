"""SafeRoute evaluation on the frozen test splits, using the exported ONNX models (UC-02).

    python scripts/evaluate.py classifier   # GTSRB test crops -> accuracy, P, R, F1, confusion matrix
    python scripts/evaluate.py detector     # PauloLab test photos -> P, R, F1, mAP@0.5, mAP@0.5:0.95
    python scripts/evaluate.py cascade      # real-photo crops labeled with scripts/label_crops.py
    python scripts/evaluate.py all

Results: reports/eval_<part>.json (+ confusion-matrix PNGs when matplotlib is installed).
Evaluating the ONNX files (not the .pt) also verifies the ONNX export end to end.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from saferoute.evaluation.metrics import classification_report, detection_report  # noqa: E402
from saferoute.vision.cascade import GENERIC, SignCascade  # noqa: E402
from saferoute.vision.classifier_engine import ClassifierEngine  # noqa: E402
from saferoute.vision.detector_engine import DetectorEngine  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
OTHER = "OTHER"


def class_names(path=ROOT / "configs/classes.yaml"):
    data = yaml.safe_load(Path(path).read_text())["classes"]
    return [data[k]["name"] for k in sorted(data, key=int)]


def save(name: str, rep: dict):
    out = ROOT / "reports" / f"eval_{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"written: {out.relative_to(ROOT)}")


def plot_cm(rep: dict, name: str, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cm = np.array(rep["confusion_matrix"], float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    labels = [s.replace("SPEED_LIMIT_", "SL").replace("PEDESTRIAN_CROSSING", "PED")
              .replace("SCHOOL_ZONE", "SCHOOL") for s in rep["labels"]]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / f"confusion_{name}.png", dpi=150)
    plt.close(fig)


def eval_classifier(a):
    names = class_names()
    eng = ClassifierEngine(Path(a.cls_model), names)
    y_true, y_pred, ms = [], [], []
    root = Path(a.crops)
    for name in names:
        for p in sorted((root / name).glob("*")):
            if p.suffix.lower() not in IMG_EXT:
                continue
            r = eng.predict(p.read_bytes())
            y_true.append(name)
            y_pred.append(r["label"])
            ms.append(r["inference_ms"])
    assert y_true, f"no crops found under {root}/<CLASS_NAME>/"
    rep = classification_report(y_true, y_pred, names)
    rep.update(model=a.cls_model, model_sha256=eng.model_hash, provider=eng.provider,
               split=str(root), mean_inference_ms=round(float(np.mean(ms)), 3))
    print(f"classifier: n={rep['n']} accuracy={rep['accuracy']} macro P/R/F1="
          f"{rep['macro']['precision']}/{rep['macro']['recall']}/{rep['macro']['f1']}")
    save("classifier", rep)
    plot_cm(rep, "classifier", "Baseline CNN (ONNX) on GTSRB test crops")


def yolo_gt(label_path: Path, w: int, h: int) -> np.ndarray:
    rows = []
    if label_path.exists():
        for line in label_path.read_text().split("\n"):
            parts = line.split()
            if len(parts) >= 5:
                _, cx, cy, bw, bh = map(float, parts[:5])
                rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, float).reshape(-1, 4)


def eval_detector(a):
    det = DetectorEngine(Path(a.det_model), conf=0.001, iou=0.7, max_det=300)
    imgs = sorted(p for p in (Path(a.data) / "images" / a.split).glob("*") if p.suffix.lower() in IMG_EXT)
    assert imgs, f"no images in {a.data}/images/{a.split}"
    preds, gts, ms = [], [], []
    for p in imgs:
        img = cv2.imread(str(p))
        dets, t = det.detect(img)
        ms.append(t)
        preds.append((np.array([d["box"] for d in dets], float).reshape(-1, 4),
                      np.array([d["det_conf"] for d in dets], float)))
        gts.append(yolo_gt(Path(a.data) / "labels" / a.split / (p.stem + ".txt"), img.shape[1], img.shape[0]))
    rep = detection_report(preds, gts, op_conf=a.op_conf)
    rep.update(model=a.det_model, model_sha256=det.model_hash, provider=det.provider,
               split=f"{a.data}/{a.split}", mean_inference_ms=round(float(np.mean(ms)), 3))
    print(f"detector: P={rep['precision']} R={rep['recall']} F1={rep['f1']} "
          f"mAP50={rep['map50']} mAP50-95={rep['map50_95']} (conf {a.op_conf})")
    save("detector", rep)


def eval_cascade(a):
    """Stage-2 accuracy on REAL road-photo crops (domain-gap check). Each ground-truth box of the
    test split was labeled by a team member with one of the 10 classes or OTHER (scripts/label_crops.py).
    The system is correct when it names the right class, or answers SIGN (generic) for OTHER."""
    names = class_names()
    cas = SignCascade(None, ClassifierEngine(Path(a.cls_model), names), {}, min_cls_conf=a.min_cls_conf)
    rows = list(csv.DictReader(open(a.labels)))
    assert rows, f"no rows in {a.labels}"
    y_true, y_pred, raw_correct, abstain = [], [], 0, 0
    for r in rows:
        if r["label"] == "SKIP":
            continue
        img = cv2.imread(r["image"])
        if img is None:
            raise SystemExit(f"cannot read {r['image']} (run from the repo root)")
        crop = cas.crop(img, [float(r[k]) for k in ("x1", "y1", "x2", "y2")])
        if crop is None:
            pred, raw = GENERIC, None
        else:
            p = cas.classifier.predict_array(crop)
            raw = p["label"]
            pred = raw if p["confidence"] >= a.min_cls_conf else GENERIC
        abstain += pred == GENERIC
        raw_correct += raw == r["label"]
        y_true.append(r["label"])
        y_pred.append(OTHER if pred == GENERIC else pred)
    rep = classification_report(y_true, y_pred, names + [OTHER])
    known = [i for i, t in enumerate(y_true) if t != OTHER]
    rep.update(min_classifier_conf=a.min_cls_conf, crops=len(y_true),
               in_taxonomy=len(known), generic_rate=round(abstain / max(len(y_true), 1), 4),
               accuracy_in_taxonomy=round(
                   sum(1 for i in known if y_pred[i] == y_true[i]) / max(len(known), 1), 4),
               raw_classifier_accuracy_without_threshold=round(raw_correct / max(len(known), 1), 4))
    print(f"cascade (real crops): n={len(y_true)} in-taxonomy={len(known)} accuracy={rep['accuracy']} "
          f"macro F1={rep['macro']['f1']} generic rate={rep['generic_rate']}")
    save("cascade", rep)
    plot_cm(rep, "cascade", f"Cascade stage 2 on real photos (threshold {a.min_cls_conf})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["classifier", "detector", "cascade", "all"])
    ap.add_argument("--cls-model", default="models/baseline_cnn.onnx")
    ap.add_argument("--det-model", default="models/detector_yolo11n.onnx")
    ap.add_argument("--crops", default="data/processed/crops/test")
    ap.add_argument("--data", default="data/processed/signs_v0")
    ap.add_argument("--split", default="test")
    ap.add_argument("--op-conf", type=float, default=0.35)
    ap.add_argument("--labels", default="reports/cascade_labels.csv")
    ap.add_argument("--min-cls-conf", type=float, default=0.8)
    a = ap.parse_args()
    t0 = time.time()
    if a.part in ("classifier", "all"):
        eval_classifier(a)
    if a.part in ("detector", "all"):
        eval_detector(a)
    if a.part in ("cascade", "all"):
        if Path(a.labels).exists():
            eval_cascade(a)
        else:
            print(f"skip cascade: {a.labels} not found (run scripts/label_crops.py first)")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
