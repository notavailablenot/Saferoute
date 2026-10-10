"""ByteTrack-style multi-object tracker and alert manager (UC-01 steps 5 and 6).

Tracker (after Zhang et al., ByteTrack, ECCV 2022), implemented in NumPy so the backend
container needs no extra dependency:
  1. high-confidence detections are matched to existing tracks by IoU (greedy, best first);
  2. remaining tracks get a second chance with low-confidence detections, which recovers
     signs that are briefly blurred or occluded;
  3. unmatched high-confidence detections above `new_track_thr` start new tracks;
  4. tracks not seen for `track_buffer` frames are removed.
A track is CONFIRMED once it was matched in at least `min_hits` of the last `window` frames,
and its class is the confidence-weighted vote over its history (stops 30/80 flicker).

AlertManager issues ONE alert per confirmed track, never re-alerts the same class within
`cooldown_s` seconds (re-identification of the same physical sign), and orders simultaneous
alerts by tier (1 = critical regulatory, 2 = speed limit, 3 = warning / generic sign).
"""
from collections import defaultdict, deque

import numpy as np

GENERIC = "SIGN"


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_match(iou: np.ndarray, thr: float):
    """Return (matches, unmatched_rows, unmatched_cols), pairing highest IoU first."""
    matches, used_r, used_c = [], set(), set()
    if iou.size:
        for flat in np.argsort(iou, axis=None)[::-1]:
            r, c = divmod(int(flat), iou.shape[1])
            if iou[r, c] < thr:
                break
            if r in used_r or c in used_c:
                continue
            matches.append((r, c))
            used_r.add(r)
            used_c.add(c)
    rows = [r for r in range(iou.shape[0]) if r not in used_r]
    cols = [c for c in range(iou.shape[1]) if c not in used_c]
    return matches, rows, cols


class Track:
    def __init__(self, tid: int, det: dict, window: int):
        self.id = tid
        self.box = np.array(det["box"], dtype=float)
        self.votes: dict[str, float] = defaultdict(float)
        self.history: deque[bool] = deque(maxlen=window)
        self.lost = 0
        self.last = det
        self.update(det)

    def update(self, det: dict):
        self.box = np.array(det["box"], dtype=float)
        weight = det.get("det_conf", 0.5) * max(det.get("cls_conf", 0.0), 0.05)
        self.votes[det.get("label", GENERIC)] += weight
        self.history.append(True)
        self.lost = 0
        self.last = det

    def miss(self):
        self.history.append(False)
        self.lost += 1

    @property
    def label(self) -> str:
        specific = {k: v for k, v in self.votes.items() if k != GENERIC}
        pool = specific or self.votes
        return max(pool, key=pool.get)

    def confirmed(self, min_hits: int) -> bool:
        return sum(self.history) >= min_hits


class ByteTracker:
    def __init__(self, high_thr=0.5, low_thr=0.1, new_track_thr=0.6, match_iou=0.3,
                 track_buffer=30, min_hits=3, window=5):
        self.high_thr, self.low_thr, self.new_thr = high_thr, low_thr, new_track_thr
        self.match_iou, self.buffer = match_iou, track_buffer
        self.min_hits, self.window = min_hits, window
        self.tracks: list[Track] = []
        self._next = 1

    def update(self, dets: list[dict]) -> list[Track]:
        high = [d for d in dets if d["det_conf"] >= self.high_thr]
        low = [d for d in dets if self.low_thr <= d["det_conf"] < self.high_thr]
        tboxes = np.array([t.box for t in self.tracks]).reshape(-1, 4)

        m1, rest_t, rest_h = greedy_match(
            iou_matrix(tboxes, np.array([d["box"] for d in high]).reshape(-1, 4)), self.match_iou)
        for ti, di in m1:
            self.tracks[ti].update(high[di])
        rest_boxes = tboxes[rest_t] if rest_t else np.zeros((0, 4))
        m2, still_t, _ = greedy_match(
            iou_matrix(rest_boxes, np.array([d["box"] for d in low]).reshape(-1, 4)), self.match_iou)
        for ri, di in m2:
            self.tracks[rest_t[ri]].update(low[di])
        for ri in still_t:
            self.tracks[rest_t[ri]].miss()
        for di in rest_h:
            if high[di]["det_conf"] >= self.new_thr:
                self.tracks.append(Track(self._next, high[di], self.window))
                self._next += 1
        self.tracks = [t for t in self.tracks if t.lost <= self.buffer]
        return [t for t in self.tracks if t.lost == 0]


class AlertManager:
    def __init__(self, tiers: dict[str, int], cooldown_s: float = 5.0, min_hits: int = 3):
        self.tiers, self.cooldown, self.min_hits = tiers, cooldown_s, min_hits
        self.alerted: set[int] = set()
        self.last_by_class: dict[str, float] = {}

    def process(self, tracks: list[Track], now: float) -> list[dict]:
        events = []
        for t in tracks:
            if t.id in self.alerted or not t.confirmed(self.min_hits):
                continue
            label = t.label
            self.alerted.add(t.id)  # one alert per track, even if suppressed by cooldown
            if now - self.last_by_class.get(label, -1e9) < self.cooldown:
                continue
            self.last_by_class[label] = now
            events.append({"track_id": t.id, "label": label,
                           "tier": self.tiers.get(label, 3),
                           "confidence": round(float(t.last.get("cls_conf", 0.0)), 4),
                           "box": [round(float(v), 1) for v in t.box]})
        return sorted(events, key=lambda e: e["tier"])


class SessionPipeline:
    """Per-session state: tracker + alert manager (one per driving session)."""

    def __init__(self, tiers: dict[str, int], cooldown_s: float = 5.0):
        self.tracker = ByteTracker()
        self.alerts = AlertManager(tiers, cooldown_s, self.tracker.min_hits)

    def step(self, dets: list[dict], now: float) -> tuple[list[dict], list[dict]]:
        active = self.tracker.update(dets)
        by_box = {tuple(t.last["box"]): t for t in active}
        for d in dets:
            t = by_box.get(tuple(d["box"]))
            d["track_id"] = t.id if t else None
            if t is not None and t.confirmed(self.tracker.min_hits):
                d["track_label"] = t.label
        return dets, self.alerts.process(active, now)
