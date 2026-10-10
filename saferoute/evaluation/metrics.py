"""Evaluation metrics for SafeRoute (UC-02), implemented in NumPy so they run on the ONNX models
without PyTorch: classification report (accuracy, precision, recall, F1, confusion matrix) and
detection metrics (precision, recall, F1 at an operating threshold; AP at IoU 0.50 and
mAP over IoU 0.50:0.95 with 101-point interpolation, as in COCO and Ultralytics)."""
import numpy as np


def classification_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    idx = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    tp = np.diag(cm).astype(float)
    support = cm.sum(1)
    pred_n = cm.sum(0)
    prec = np.divide(tp, pred_n, out=np.zeros_like(tp), where=pred_n > 0)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    present = support > 0
    total = int(support.sum())
    per_class = {c: {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                     "f1": round(float(f1[i]), 4), "support": int(support[i])}
                 for i, c in enumerate(labels)}
    w = support / max(total, 1)
    return {
        "n": total,
        "accuracy": round(float(tp.sum() / max(total, 1)), 4),
        "macro": {"precision": round(float(prec[present].mean()), 4) if present.any() else 0.0,
                  "recall": round(float(rec[present].mean()), 4) if present.any() else 0.0,
                  "f1": round(float(f1[present].mean()), 4) if present.any() else 0.0},
        "weighted": {"precision": round(float((prec * w).sum()), 4),
                     "recall": round(float((rec * w).sum()), 4), "f1": round(float((f1 * w).sum()), 4)},
        "per_class": per_class,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def match_image(pred_boxes, pred_scores, gt_boxes, iou_thr):
    """Greedy (highest score first) matching inside one image; returns a TP flag per prediction."""
    order = np.argsort(-pred_scores)
    ious = box_iou(pred_boxes[order], gt_boxes)
    used = np.zeros(len(gt_boxes), bool)
    tp = np.zeros(len(order), bool)
    for k in range(len(order)):
        if not len(gt_boxes):
            break
        cand = np.where(~used & (ious[k] >= iou_thr))[0]
        if len(cand):
            j = cand[np.argmax(ious[k, cand])]
            used[j] = True
            tp[k] = True
    out = np.zeros(len(order), bool)
    out[order] = tp
    return out


def average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = tp[order].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))  # precision envelope
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x)) if hasattr(np, "trapezoid") \
        else float(np.trapz(np.interp(x, mrec, mpre), x))


def detection_report(preds: list[tuple[np.ndarray, np.ndarray]], gts: list[np.ndarray],
                     op_conf: float = 0.35) -> dict:
    """preds[i] = (boxes Nx4, scores N) for image i (low conf threshold, e.g. 0.001);
    gts[i] = ground-truth boxes Mx4. Single-class detection (generic traffic sign)."""
    n_gt = int(sum(len(g) for g in gts))
    aps = []
    for thr in np.arange(0.5, 0.96, 0.05):
        tps, scs = [], []
        for (pb, ps), g in zip(preds, gts):
            tps.append(match_image(pb, ps, g, thr))
            scs.append(ps)
        tp = np.concatenate(tps) if tps else np.zeros(0, bool)
        sc = np.concatenate(scs) if scs else np.zeros(0)
        aps.append(average_precision(tp, sc, n_gt))
    tp_op = fp_op = 0
    for (pb, ps), g in zip(preds, gts):
        keep = ps >= op_conf
        m = match_image(pb[keep], ps[keep], g, 0.5)
        tp_op += int(m.sum())
        fp_op += int((~m).sum())
    fn_op = n_gt - tp_op
    p = tp_op / max(tp_op + fp_op, 1)
    r = tp_op / max(n_gt, 1)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"images": len(gts), "instances": n_gt, "operating_conf": op_conf,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp_op, "fp": fp_op, "fn": fn_op,
            "map50": round(aps[0], 4), "map50_95": round(float(np.mean(aps)), 4)}


def latency_summary(ms: list[float]) -> dict:
    a = np.asarray(ms, float)
    return {"n": int(a.size), "mean_ms": round(float(a.mean()), 3),
            "p50_ms": round(float(np.percentile(a, 50)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3),
            "p99_ms": round(float(np.percentile(a, 99)), 3),
            "max_ms": round(float(a.max()), 3), "fps_equiv": round(1000.0 / float(a.mean()), 1)}
