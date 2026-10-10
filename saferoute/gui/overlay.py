"""Drawing helpers shared by the Drive and Image tabs (pure OpenCV/NumPy, no Qt)."""
import cv2
import numpy as np

# BGR colours per alert tier: 1 critical regulatory = red, 2 speed limit = blue, 3 warning = amber
TIER_COLORS = {1: (40, 40, 220), 2: (200, 120, 30), 3: (0, 170, 255)}

PRETTY = {
    "STOP": "STOP", "YIELD": "YIELD", "NO_ENTRY": "NO ENTRY",
    "SPEED_LIMIT_20": "SPEED LIMIT 20", "SPEED_LIMIT_30": "SPEED LIMIT 30",
    "SPEED_LIMIT_60": "SPEED LIMIT 60", "SPEED_LIMIT_80": "SPEED LIMIT 80",
    "SPEED_LIMIT_100": "SPEED LIMIT 100", "PEDESTRIAN_CROSSING": "PEDESTRIAN CROSSING",
    "SCHOOL_ZONE": "SCHOOL ZONE", "SIGN": "SIGN",
}


def pretty(label: str) -> str:
    return PRETTY.get(label, label.replace("_", " "))


def alert_text(label: str) -> str:
    return "Sign ahead" if label == "SIGN" else f"{pretty(label)} ahead"


def speed_value(label: str) -> int | None:
    if label.startswith("SPEED_LIMIT_"):
        try:
            return int(label.rsplit("_", 1)[1])
        except ValueError:
            return None
    return None


def draw_detections(frame: np.ndarray, detections: list[dict], show_track: bool = True) -> np.ndarray:
    """Draw boxes and labels on a copy of a BGR frame (boxes are in frame pixels)."""
    out = frame.copy()
    thick = max(2, round(min(out.shape[:2]) / 300))
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        label = d.get("track_label") or d.get("label", "SIGN")
        color = TIER_COLORS.get(d.get("tier", 3), TIER_COLORS[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        text = pretty(label)
        if d.get("cls_conf"):
            text += f" {d['cls_conf']:.2f}"
        if show_track and d.get("track_id") is not None:
            text = f"#{d['track_id']} " + text
        scale = 0.5 * thick / 2 + 0.25
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = max(y1, th + 6)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
        cv2.putText(out, text, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return out


def scale_detections(detections: list[dict], factor: float) -> list[dict]:
    """Scale boxes from the uploaded (resized) frame back to the original frame size."""
    if factor == 1.0:
        return detections
    return [{**d, "box": [v * factor for v in d["box"]]} for d in detections]
