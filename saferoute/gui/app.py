"""SafeRoute Presentation Tier: PyQt6 desktop client.

Run (backend must be running first, natively or in Docker):
    python -m saferoute.gui                       # API at http://127.0.0.1:8000
    SAFEROUTE_API=http://127.0.0.1:8000 python -m saferoute.gui

Tabs
  Drive  (UC-01): webcam or video file -> frames sent to POST /detect/image?session_id=...
                  -> boxes drawn on the exact frame, one alert per sign, speed-limit badge,
                  audio cue, per-second telemetry posted back to the API.
  Image  (upload): UploadTab (upload_tab.py): pick a JPEG/PNG -> POST /detect/image -> boxes,
                  labels and timings. Red Banner (banners.py) when the backend is offline.

Threads: capture + HTTP run in FrameWorker (QThread) and the health check in HealthWorker,
so the UI thread never blocks (Qt signals carry results back to the UI).
"""
import os
import sys
import tempfile
import time
import wave
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QThread, QTimer, QUrl, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QCheckBox, QFileDialog, QHBoxLayout,
                             QLabel, QListWidget, QMainWindow, QPushButton, QSpinBox,
                             QTabWidget, QVBoxLayout, QWidget)

from saferoute.gui.api_client import DEFAULT_URL, ApiClient
from saferoute.gui.banners import CAMERA_DISCONNECTED, Banner
from saferoute.gui.qt_env import fix_qt_env
from saferoute.gui.upload_tab import UploadTab
from saferoute.gui.overlay import (TIER_COLORS, alert_text, draw_detections, pretty,
                                   scale_detections, speed_value)

UPLOAD_MAX_W = 960  # frames are downscaled before upload (keeps transfer and decode fast)


def to_pixmap(bgr: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


def encode_frame(frame: np.ndarray) -> tuple[bytes, float]:
    """Resize to at most UPLOAD_MAX_W wide and JPEG-encode; returns (bytes, scale back factor)."""
    h, w = frame.shape[:2]
    factor = 1.0
    if w > UPLOAD_MAX_W:
        factor = w / UPLOAD_MAX_W
        frame = cv2.resize(frame, (UPLOAD_MAX_W, int(round(h / factor))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes(), factor


class FrameWorker(QThread):
    """Reads frames, sends each to the API, and emits (frame, result, end_to_end_ms)."""
    frame_ready = pyqtSignal(object, object, float)
    status = pyqtSignal(str, str)          # (kind, message): kind in {"camera", "backend", "ok"}
    stats = pyqtSignal(float, float, float)  # fps, inference_ms, end_to_end_ms (per second)
    ended = pyqtSignal(str)

    def __init__(self, api: ApiClient, source, session_id: int | None):
        super().__init__()
        self.api, self.source, self.session_id = api, source, session_id
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.ended.emit(f"Could not open source: {self.source}")
            return
        is_file = isinstance(self.source, str)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        misses, n, inf_sum, e2e_sum, t_window = 0, 0, 0.0, 0.0, time.perf_counter()
        while self._running:
            t_frame = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if is_file:
                    self.ended.emit("End of video")
                    break
                misses += 1  # UC-01 E1: camera lost -> retry 3 times over about 3 s
                self.status.emit("camera", "Camera disconnected: retrying")
                if misses > 3:
                    self.ended.emit("Camera disconnected")
                    break
                time.sleep(1.0)
                continue
            misses = 0
            result = None
            try:
                data, factor = encode_frame(frame)
                result = self.api.detect(data, self.session_id)
                result["detections"] = scale_detections(result["detections"], factor)
                self.status.emit("ok", "")
            except Exception as exc:  # backend down or bad reply: keep showing video
                self.status.emit("backend", f"Backend unavailable: {exc}"[:160])
                time.sleep(0.5)
            e2e = (time.perf_counter() - t_frame) * 1000
            self.frame_ready.emit(frame, result, e2e)
            if result is not None:
                n += 1
                inf_sum += result["timings"]["detect_ms"] + result["timings"]["classify_ms"]
                e2e_sum += e2e
            now = time.perf_counter()
            if now - t_window >= 1.0:
                fps = n / (now - t_window)
                inf, e2e_avg = (inf_sum / n, e2e_sum / n) if n else (0.0, 0.0)
                self.stats.emit(fps, inf, e2e_avg)
                if self.session_id is not None and n:
                    try:
                        self.api.post_telemetry(self.session_id, fps, inf, e2e_avg)
                    except Exception:
                        pass
                n, inf_sum, e2e_sum, t_window = 0, 0.0, 0.0, now
            if is_file:  # never play a file faster than its native frame rate
                wait = 1.0 / native_fps - (time.perf_counter() - t_frame)
                if wait > 0:
                    time.sleep(wait)
        cap.release()


class HealthWorker(QThread):
    health = pyqtSignal(object)

    def __init__(self, api: ApiClient):
        super().__init__()
        self.api, self._running = api, True

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            try:
                self.health.emit(self.api.health())
            except Exception as exc:
                self.health.emit({"status": "offline", "reason": str(exc)})
            for _ in range(30):
                if not self._running:
                    return
                self.msleep(100)


def make_tone(path: Path, freq: float, ms: int = 180, repeats: int = 1):
    sr = 22050
    t = np.arange(int(sr * ms / 1000)) / sr
    beep = (0.4 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    gap = np.zeros(int(sr * 0.08), np.int16)
    data = np.concatenate([np.concatenate([beep, gap]) for _ in range(repeats)])
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


class AudioCues:
    """A distinct tone per tier (QtMultimedia if available, otherwise the system beep)."""

    def __init__(self):
        self.effects = {}
        try:
            from PyQt6.QtMultimedia import QSoundEffect
            d = Path(tempfile.mkdtemp(prefix="saferoute_audio_"))
            for tier, (freq, rep) in {1: (880, 2), 2: (660, 1), 3: (520, 1)}.items():
                make_tone(d / f"tier{tier}.wav", freq, repeats=rep)
                e = QSoundEffect()
                e.setSource(QUrl.fromLocalFile(str(d / f"tier{tier}.wav")))
                self.effects[tier] = e
        except Exception:
            self.effects = {}
        self.last = 0.0

    def play(self, tier: int):
        if time.monotonic() - self.last < 1.5:  # audio cues at least 1.5 s apart
            return
        self.last = time.monotonic()
        e = self.effects.get(tier)
        try:
            from PyQt6.QtMultimedia import QSoundEffect
            usable = e is not None and e.status() != QSoundEffect.Status.Error
        except Exception:
            usable = False
        if usable:
            e.play()
        else:  # no audio device or QtMultimedia missing: fall back to the system beep
            QApplication.beep()


class VideoLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background:#111; color:#aaa; font-size:16px;")
        self._pix = None

    def set_frame(self, bgr: np.ndarray):
        self._pix = to_pixmap(bgr)
        self._rescale()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))


class DriveTab(QWidget):
    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win
        self.worker: FrameWorker | None = None
        self.session_id: int | None = None
        self.audio = AudioCues()

        self.banner = Banner()
        self.video = VideoLabel("Choose a source: Webcam or Video file")

        self.btn_cam = QPushButton("Start webcam")
        self.cam_index = QSpinBox()
        self.cam_index.setRange(0, 9)
        self.cam_index.setPrefix("camera ")
        self.btn_file = QPushButton("Open video file")
        self.btn_stop = QPushButton("Stop session")
        self.btn_stop.setEnabled(False)
        self.chk_mute = QCheckBox("Mute audio")
        self.chk_ids = QCheckBox("Show track IDs")
        self.btn_cam.clicked.connect(self.start_camera)
        self.btn_file.clicked.connect(self.start_file)
        self.btn_stop.clicked.connect(self.stop)

        controls = QHBoxLayout()
        for w in (self.btn_cam, self.cam_index, self.btn_file, self.btn_stop, self.chk_mute, self.chk_ids):
            controls.addWidget(w)
        controls.addStretch()

        self.alert_big = QLabel("No alerts yet")
        self.alert_big.setWordWrap(True)
        self.alert_big.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alert_big.setMinimumHeight(110)
        self.alert_big.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.alert_big.setStyleSheet("border:2px solid #444; border-radius:8px; padding:8px;")
        self.badge = QLabel("--")
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.setFixedSize(120, 120)
        self.badge.setFont(QFont("Arial", 34, QFont.Weight.Bold))
        self.badge.setStyleSheet("border:10px solid #d32f2f; border-radius:60px; background:white; color:black;")
        self.history = QListWidget()
        self.history.setMaximumHeight(170)

        side = QVBoxLayout()
        side.addWidget(QLabel("Current alert"))
        side.addWidget(self.alert_big)
        side.addWidget(QLabel("Speed limit"))
        side.addWidget(self.badge, alignment=Qt.AlignmentFlag.AlignHCenter)
        side.addWidget(QLabel("Last 5 alerts"))
        side.addWidget(self.history)
        side.addStretch()
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(300)

        body = QHBoxLayout()
        body.addWidget(self.video, 1)
        body.addWidget(side_w)
        root = QVBoxLayout(self)
        root.addWidget(self.banner)
        root.addLayout(controls)
        root.addLayout(body, 1)

    # ---- session control ----
    def _start(self, source):
        if self.worker is not None:
            self.stop()
        try:
            h = self.win.api.health()
            if not h.get("detector_ready"):
                self.show_banner(f"Detector not loaded on the backend: {h.get('detector_reason') or h.get('reason')}")
                return
            version = f"cascade det:{h.get('detector_sha256', '')[:8]} cls:{h.get('model_sha256', '')[:8]}"
            self.session_id = self.win.api.start_session(version)
        except Exception as exc:
            self.show_banner(f"Backend unavailable: {exc}")
            return
        self.history.clear()
        self.badge.setText("--")
        self.alert_big.setText("Monitoring...")
        self.alert_big.setStyleSheet("border:2px solid #444; border-radius:8px; padding:8px;")
        self.worker = FrameWorker(self.win.api, source, self.session_id)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status.connect(self.on_status)
        self.worker.stats.connect(self.win.on_stats)
        self.worker.ended.connect(self.on_ended)
        self.worker.start()
        self.btn_stop.setEnabled(True)
        self.win.set_session(self.session_id)

    def start_camera(self):
        self._start(self.cam_index.value())

    def start_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", "Video (*.mp4 *.avi *.mkv *.mov)")
        if path:
            self._start(path)

    def stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
        if self.session_id is not None:
            try:
                s = self.win.api.close_session(self.session_id)
                self.win.statusBar().showMessage(
                    f"Session {self.session_id} closed: {s.get('alerts', 0)} alerts logged", 8000)
            except Exception:
                pass
            self.session_id = None
            self.win.set_session(None)
        self.btn_stop.setEnabled(False)

    # ---- worker signals ----
    def on_frame(self, frame, result, e2e_ms):
        if result is None:
            self.video.set_frame(frame)
            return
        self.video.set_frame(draw_detections(frame, result["detections"], self.chk_ids.isChecked()))
        for a in result.get("alerts", []):
            self.on_alert(a)

    def on_alert(self, a: dict):
        tier = a["tier"]
        b, g, r = TIER_COLORS.get(tier, TIER_COLORS[3])
        self.alert_big.setText(alert_text(a["label"]))
        self.alert_big.setStyleSheet(f"border:4px solid rgb({r},{g},{b}); border-radius:8px; "
                                     f"padding:8px; color:rgb({r},{g},{b});")
        v = speed_value(a["label"])
        if v is not None:
            self.badge.setText(str(v))
        self.history.insertItem(0, f"{time.strftime('%H:%M:%S')}  {pretty(a['label'])}  (tier {tier})")
        while self.history.count() > 5:
            self.history.takeItem(5)
        if not self.chk_mute.isChecked():
            self.audio.play(tier)

    def on_status(self, kind, msg):
        if kind == "ok":
            self.banner.clear()
        else:
            self.show_banner(msg)

    def on_ended(self, msg):
        if "disconnected" in msg.lower():
            self.show_banner(CAMERA_DISCONNECTED)
        elif "could not" in msg.lower():
            self.show_banner(msg)
        self.win.statusBar().showMessage(msg, 8000)
        self.stop()

    def show_banner(self, msg):
        self.banner.show_error(msg)


class MainWindow(QMainWindow):
    def __init__(self, api_url: str = DEFAULT_URL, start_health: bool = True):
        super().__init__()
        self.setWindowTitle("SafeRoute: Traffic Sign Detection and Driver Warning")
        self.resize(1280, 760)
        self.api = ApiClient(api_url)
        self.drive = DriveTab(self)
        self.image = UploadTab(api_url, start_health=start_health)
        tabs = QTabWidget()
        tabs.addTab(self.drive, "Drive (live)")
        tabs.addTab(self.image, "Image upload")
        self.setCentralWidget(tabs)

        self.lbl_backend = QLabel("Backend: checking...")
        self.lbl_stats = QLabel("FPS -- | inference -- ms | end-to-end -- ms")
        self.lbl_session = QLabel("No session")
        for w in (self.lbl_stats, self.lbl_session, self.lbl_backend):
            self.statusBar().addPermanentWidget(w)
        quit_act = QAction("Quit", self)
        quit_act.triggered.connect(self.close)
        self.menuBar().addMenu("File").addAction(quit_act)

        self.health_worker = HealthWorker(self.api)
        self.health_worker.health.connect(self.on_health)
        if start_health:
            self.health_worker.start()

    def on_health(self, h: dict):
        if h.get("status") == "ready":
            det = "detector OK" if h.get("detector_ready") else "detector MISSING"
            self.lbl_backend.setText(f"Backend: ready ({h.get('execution_provider', '?')}, {det})")
            self.lbl_backend.setStyleSheet("color:#1b8a3a;")
        else:
            self.lbl_backend.setText(f"Backend: {h.get('status', 'offline')}")
            self.lbl_backend.setStyleSheet("color:#b00020; font-weight:bold;")

    def on_stats(self, fps, inf, e2e):
        self.lbl_stats.setText(f"FPS {fps:.1f} | inference {inf:.1f} ms | end-to-end {e2e:.1f} ms")

    def set_session(self, sid):
        self.lbl_session.setText(f"Session {sid}" if sid else "No session")

    def closeEvent(self, e):
        self.drive.stop()
        self.image.health_timer.stop()
        self.health_worker.stop()
        self.health_worker.wait(2000)
        super().closeEvent(e)


def main():
    fix_qt_env()
    app = QApplication(sys.argv)
    w = MainWindow(os.getenv("SAFEROUTE_API", DEFAULT_URL))
    w.show()
    # allow Ctrl+C in the terminal to close the window
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
