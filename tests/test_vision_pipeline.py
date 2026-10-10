import numpy as np
from helpers import tiny_classifier, tiny_detector

from saferoute.vision.cascade import GENERIC, SignCascade
from saferoute.vision.classifier_engine import ClassifierEngine
from saferoute.vision.detector_engine import DetectorEngine, letterbox, nms
from saferoute.vision.tracker import AlertManager, ByteTracker, SessionPipeline

NAMES = ["STOP", "YIELD", "NO_ENTRY", "SPEED_LIMIT_20", "SPEED_LIMIT_30", "SPEED_LIMIT_60",
         "SPEED_LIMIT_80", "SPEED_LIMIT_100", "PEDESTRIAN_CROSSING", "SCHOOL_ZONE"]
TIERS = {n: (1 if i < 3 else 2 if i < 8 else 3) for i, n in enumerate(NAMES)} | {GENERIC: 3}


def test_letterbox_geometry():
    img = np.zeros((480, 1280, 3), np.uint8)  # wide frame: width limits the scale
    out, r, (left, top) = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert r == 0.5 and left == 0 and top == 200  # 1280x480 -> 640x240, padded 200 top/bottom


def test_nms_removes_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert keep == [0, 2]


def test_detector_maps_boxes_back_to_frame(tmp_path):
    # frame 1280x480 -> r=0.5, top pad 200. A box at (100..200, 300..360) in the frame is
    # (50..100, 350..380) in letterbox space: cx=75, cy=365, w=50, h=30.
    tiny_detector(tmp_path / "d.onnx", [(75, 365, 50, 30, 0.9), (76, 366, 50, 30, 0.6),
                                        (300, 300, 20, 20, 0.1)])
    eng = DetectorEngine(tmp_path / "d.onnx", conf=0.35, iou=0.5)
    dets, ms = eng.detect(np.zeros((480, 1280, 3), np.uint8))
    assert len(dets) == 1 and ms >= 0  # duplicate removed by NMS, low score filtered
    assert np.allclose(dets[0]["box"], [100, 300, 200, 360], atol=0.5)
    assert dets[0]["det_conf"] == 0.9


def test_cascade_labels_and_generic_fallback(tmp_path):
    tiny_detector(tmp_path / "d.onnx", [(320, 320, 80, 80, 0.9)])
    tiny_classifier(tmp_path / "c_sure.onnx", n_classes=10, favour=0)
    tiny_classifier(tmp_path / "c_unsure.onnx", n_classes=10)
    det = DetectorEngine(tmp_path / "d.onnx")
    frame = np.full((640, 640, 3), 128, np.uint8)
    sure = SignCascade(det, ClassifierEngine(tmp_path / "c_sure.onnx", NAMES), TIERS).run(frame)
    d = sure["detections"][0]
    assert d["label"] == "STOP" and d["tier"] == 1 and d["cls_conf"] > 0.99
    tiny_classifier(tmp_path / "c_reject.onnx", favour=10)
    rej = SignCascade(det, ClassifierEngine(tmp_path / "c_reject.onnx", NAMES + ["NOT_SIGN"]), TIERS).run(frame)
    assert rej["detections"] == [] and rej["rejected"] == 1  # billboard-like crop: dropped, no alert
    unsure = SignCascade(det, ClassifierEngine(tmp_path / "c_unsure.onnx", NAMES), TIERS).run(frame)
    assert unsure["detections"][0]["label"] == GENERIC  # low confidence -> generic "Sign ahead"
    assert {"detect_ms", "classify_ms", "pipeline_ms"} <= set(sure)


def _det(box, conf=0.9, label="STOP", cconf=0.95):
    return {"box": list(box), "det_conf": conf, "label": label, "cls_conf": cconf}


def test_tracker_confirms_after_three_frames_and_recovers_low_scores():
    tr = ByteTracker()
    for i in range(2):
        tr.update([_det((100 + i, 100, 150 + i, 150))])
    assert not tr.tracks[0].confirmed(3)
    tr.update([_det((102, 100, 152, 150), conf=0.3)])  # low score still matched (ByteTrack)
    assert len(tr.tracks) == 1 and tr.tracks[0].confirmed(3)


def test_one_alert_per_sign_and_cooldown():
    pipe = SessionPipeline(TIERS, cooldown_s=5)
    alerts = []
    for i in range(10):  # the same sign seen in 10 consecutive frames
        _, a = pipe.step([_det((100 + 2 * i, 100, 150 + 2 * i, 150))], now=i * 0.1)
        alerts += a
    assert [a["label"] for a in alerts] == ["STOP"]
    # a new STOP track 1 s later (re-identification of the same sign) is suppressed
    for i in range(4):
        _, a = pipe.step([_det((400, 100, 450, 150))], now=1.0 + i * 0.1)
        alerts += a
    assert len(alerts) == 1


def test_alerts_sorted_by_tier_and_class_vote():
    am = AlertManager(TIERS, cooldown_s=0)
    tr = ByteTracker()
    for _ in range(3):
        tracks = tr.update([_det((0, 0, 50, 50), label="PEDESTRIAN_CROSSING"),
                            _det((200, 0, 250, 50), label="SPEED_LIMIT_30", cconf=0.9),
                            _det((400, 0, 450, 50), label="STOP")])
    events = am.process(tracks, now=0)
    assert [e["tier"] for e in events] == [1, 2, 3]
