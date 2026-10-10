"""Two-stage SafeRoute cascade: YOLO11n localizes signs, the baseline CNN names them.

Every detector box is cropped (with a small margin) and classified. When the classifier
is not confident enough (domain gap: it was trained on GTSRB crops), the sign is reported
with the generic label SIGN instead of a possibly wrong class, so the driver still gets a
"Sign ahead" warning without a misleading value.

Rejection (Sprint 2): the classifier also has a NOT_SIGN class trained on hard negatives
(scripts/make_negative_crops.py). Detector boxes it labels NOT_SIGN (billboards, trees, ...)
are dropped, so they never reach the tracker or raise an alert. OTHER_SIGN (real road-photo
signs of unknown type) becomes the generic "Sign ahead" alert.
"""
import time

import numpy as np

from saferoute.vision.classifier_engine import ClassifierEngine
from saferoute.vision.detector_engine import DetectorEngine

GENERIC = "SIGN"
REJECT = "NOT_SIGN"
OTHER = "OTHER_SIGN"  # a real sign of unknown type -> generic "Sign ahead"


class SignCascade:
    def __init__(self, detector: DetectorEngine, classifier: ClassifierEngine,
                 tiers: dict[str, int], min_cls_conf: float = 0.8, margin: float = 0.08,
                 min_box_px: int = 12):
        self.detector, self.classifier = detector, classifier
        self.tiers = tiers
        self.min_cls_conf, self.margin, self.min_box_px = min_cls_conf, margin, min_box_px

    def crop(self, bgr: np.ndarray, box: list[float]) -> np.ndarray | None:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w < self.min_box_px or h < self.min_box_px:
            return None
        mx, my = w * self.margin, h * self.margin
        H, W = bgr.shape[:2]
        xa, ya = int(max(0, x1 - mx)), int(max(0, y1 - my))
        xb, yb = int(min(W, x2 + mx)), int(min(H, y2 + my))
        if xb - xa < 2 or yb - ya < 2:
            return None
        return np.ascontiguousarray(bgr[ya:yb, xa:xb, ::-1])  # to RGB for the classifier

    def run(self, bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()
        dets, det_ms = self.detector.detect(bgr)
        cls_ms = 0.0
        out, rejected = [], 0
        for d in dets:
            crop = self.crop(bgr, d["box"])
            if crop is None:  # too small to classify reliably: keep as a generic sign
                label, cconf, top = GENERIC, 0.0, []
            else:
                p = self.classifier.predict_array(crop)
                cls_ms += p["inference_ms"]
                label, cconf, top = p["label"], p["confidence"], p["top_k"]
                if label == REJECT:  # classifier says this is not a traffic sign
                    rejected += 1
                    continue
                if cconf < self.min_cls_conf or label == OTHER:
                    label = GENERIC
            out.append({**d, "label": label, "cls_conf": round(float(cconf), 4),
                        "raw_label": top[0]["label"] if top else None,
                        "tier": self.tiers.get(label, 3)})
        return {"detections": out, "rejected": rejected, "detect_ms": round(det_ms, 3), "classify_ms": round(cls_ms, 3),
                "pipeline_ms": round((time.perf_counter() - t0) * 1000, 3)}
