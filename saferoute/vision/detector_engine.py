"""ONNX Runtime engine for the YOLO11n traffic-sign detector (stage 1 of the cascade).

The model is exported by scripts/export_detector.py from the Sprint 1 Ultralytics run.
Its output is (1, 4 + num_classes, N): for each of N candidate boxes, cx, cy, w, h in
letterboxed input pixels followed by one score per class (already sigmoid-activated).

Pre- and post-processing follow Ultralytics exactly (letterbox with grey padding 114,
RGB, scaled to [0, 1]; confidence filter; NMS), so ONNX results match `model.predict`.
Loaded ONCE at API startup, never per request.
"""
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from saferoute.vision.classifier_engine import sha256_of


def letterbox(img: np.ndarray, size: int = 640, color: int = 114):
    """Resize keeping aspect ratio and pad to size x size (Ultralytics LetterBox, centered).

    Returns the padded image, the scale ratio, and the (left, top) padding in pixels.
    """
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(color, color, color))
    return img, r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Plain NumPy non-maximum suppression on xyxy boxes; returns kept indices."""
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


class DetectorEngine:
    def __init__(self, model_path: Path, conf: float = 0.35, iou: float = 0.5,
                 img_size: int = 640, max_det: int = 50):
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.provider = self.session.get_providers()[0]
        self.model_hash = sha256_of(Path(model_path))
        self.conf, self.iou, self.img_size, self.max_det = conf, iou, img_size, max_det

    def preprocess(self, bgr: np.ndarray):
        padded, r, pad = letterbox(bgr, self.img_size)
        x = padded[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        x = np.ascontiguousarray(x, dtype=np.float32)[None] / 255.0
        return x, r, pad

    def postprocess(self, out: np.ndarray, r: float, pad: tuple, shape: tuple) -> list[dict]:
        pred = out[0].T  # (N, 4 + nc)
        scores_all = pred[:, 4:]
        cls = scores_all.argmax(1)
        scores = scores_all[np.arange(len(pred)), cls]
        m = scores >= self.conf
        pred, scores, cls = pred[m], scores[m], cls[m]
        if not len(pred):
            return []
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        # class-wise NMS: offset boxes per class so different classes never suppress each other
        offs = boxes + cls[:, None] * 4096.0
        keep = nms(offs, scores, self.iou)[: self.max_det]
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes /= r
        H, W = shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
        return [{"box": [round(float(v), 1) for v in b], "det_conf": round(float(s), 4),
                 "det_class": int(c)} for b, s, c in zip(boxes, scores, cls)]

    def detect(self, bgr: np.ndarray) -> tuple[list[dict], float]:
        """Run the detector on a BGR frame; returns (detections, inference_ms)."""
        x, r, pad = self.preprocess(bgr)
        t0 = time.perf_counter()
        out = self.session.run(None, {self.input_name: x})[0]
        ms = (time.perf_counter() - t0) * 1000
        return self.postprocess(out, r, pad, bgr.shape), ms
