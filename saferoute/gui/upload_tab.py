"""Image upload tab for the SafeRoute GUI.

Flow: pick an image -> send it to the backend -> show the result.

Default endpoint: POST /detect/image (detector + classifier cascade). Response:
    {"image": {...}, "detections": [{"box": [x1, y1, x2, y2], "det_conf": float,
      "label": str, "cls_conf": float, "raw_label": str, "tier": int}, ...],
     "rejected": int, "alerts": [...],
     "timings": {"detect_ms", "classify_ms", "pipeline_ms", "api_ms"}}
Boxes and labels are drawn on the image and each sign is listed with its confidence.

The older POST /predict/image (classifier only) still works:
    {"label": str, "confidence": float, "top_k": [...], "inference_ms": float, "api_ms": float}

Test by itself:
    python -m saferoute.gui.upload_tab
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from saferoute.gui.banners import BACKEND_OFFLINE, Banner
from saferoute.gui.overlay import pretty

API_URL = os.getenv("SAFEROUTE_API", "http://127.0.0.1:8000")
ENDPOINT = "/detect/image"
HEALTH_INTERVAL_MS = 5000

STATUS_MESSAGES = {
    413: "The image is empty or larger than 5 MB.",
    415: "Only JPEG or PNG images are accepted.",
    422: "The backend could not process this image.",
    503: "The model is not loaded in the backend yet.",
}


class _HealthThread(QThread):
    """Asks GET /health in the background. Emits (is_ready, message)."""

    result = pyqtSignal(bool, str)

    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url

    def run(self):
        try:
            data = httpx.get(f"{self.base_url}/health", timeout=2).json()
        except Exception:
            self.result.emit(False, BACKEND_OFFLINE)
            return
        if data.get("status") == "ready":
            self.result.emit(True, "")
        else:
            self.result.emit(False, f"Backend is up but not ready: {data.get('reason')}")


class _PredictThread(QThread):
    """Sends one image to the backend in the background."""

    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str, bool)  # message, backend_offline

    def __init__(self, url: str, image_path: str):
        super().__init__()
        self.url = url
        self.image_path = image_path

    def run(self):
        path = Path(self.image_path)
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        try:
            with open(path, "rb") as f:
                response = httpx.post(self.url, files={"file": (path.name, f, mime)}, timeout=15)
        except httpx.TransportError:
            self.failed.emit(BACKEND_OFFLINE, True)
            return
        except OSError as exc:
            self.failed.emit(f"Could not read the file: {exc}", False)
            return

        if response.status_code == 200:
            self.succeeded.emit(response.json())
            return
        try:
            detail = response.json().get("detail", "")
        except ValueError:
            detail = ""
        message = STATUS_MESSAGES.get(response.status_code, f"Backend error {response.status_code}")
        self.failed.emit(f"{message} {detail}".strip(), False)


def _conf(d: dict) -> float:
    """Classifier confidence from /detect/image, or plain confidence from older responses."""
    return float(d.get("cls_conf", d.get("confidence", 0.0)))


def _draw_detections(pixmap: QPixmap, detections: list[dict]) -> QPixmap:
    """Draw boxes and labels on a copy of the image."""
    pixmap = pixmap.copy()
    painter = QPainter(pixmap)
    pen = QPen(QColor("#e53935"))
    pen.setWidth(max(2, pixmap.width() // 300))
    painter.setPen(pen)
    font = painter.font()
    font.setPixelSize(max(12, pixmap.width() // 40))
    painter.setFont(font)
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        text = f"{pretty(d.get('label', '?'))} {_conf(d):.0%}"
        painter.drawText(x1 + 2, max(font.pixelSize(), y1 - 4), text)
    painter.end()
    return pixmap


class UploadTab(QWidget):
    def __init__(self, api_url: str = API_URL, endpoint: str = ENDPOINT, parent=None,
                 start_health: bool = True):
        super().__init__(parent)
        self.api_url = api_url
        self.endpoint = endpoint
        self._pixmap: QPixmap | None = None
        self._predict_thread: _PredictThread | None = None
        self._health_thread: _HealthThread | None = None

        self.banner = Banner()

        self.choose_button = QPushButton("Choose image...")
        self.choose_button.clicked.connect(self.choose_image)
        self.file_label = QLabel("No image selected")

        top = QHBoxLayout()
        top.addWidget(self.choose_button)
        top.addWidget(self.file_label, stretch=1)

        self.image_label = QLabel("Choose a JPEG or PNG image to analyze")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("border: 1px solid #888;")

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.result_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.banner)
        layout.addLayout(top)
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(self.result_label)

        # Check the backend now and then every few seconds.
        self.health_timer = QTimer(self)
        self.health_timer.timeout.connect(self.check_health)
        if start_health:
            self.health_timer.start(HEALTH_INTERVAL_MS)
            self.check_health()

    # --- backend health -------------------------------------------------
    def check_health(self):
        if self._health_thread is not None and self._health_thread.isRunning():
            return
        self._health_thread = _HealthThread(self.api_url)
        self._health_thread.result.connect(self._on_health)
        self._health_thread.start()

    def _on_health(self, ready: bool, message: str):
        if ready:
            self.banner.clear()
        else:
            self.banner.show_error(message)

    # --- choosing and sending an image ----------------------------------
    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an image", "", "Images (*.jpg *.jpeg *.png)"
        )
        if not path:
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.banner.show_error("Could not open this image file.")
            return
        self._pixmap = pixmap
        self.file_label.setText(Path(path).name)
        self.result_label.setText("Analyzing...")
        self._refresh_image(self._pixmap)
        self.choose_button.setEnabled(False)

        self._predict_thread = _PredictThread(self.api_url + self.endpoint, path)
        self._predict_thread.succeeded.connect(self._on_result)
        self._predict_thread.failed.connect(self._on_error)
        self._predict_thread.start()

    def _on_result(self, result: dict):
        self.choose_button.setEnabled(True)
        self.banner.clear()

        shown = self._pixmap
        detections = result.get("detections")
        if detections and shown is not None:
            shown = _draw_detections(shown, detections)
        self._refresh_image(shown)
        self.result_label.setText(self.format_result(result))

    @staticmethod
    def format_result(result: dict) -> str:
        """Text under the image, for /detect/image or /predict/image responses."""
        lines = []
        if "detections" in result:
            dets = result["detections"]
            if not dets:
                lines.append("<b>No traffic sign found</b>")
            for d in dets:
                lines.append(f"<b>{pretty(d['label'])}</b> ({_conf(d):.1%}), "
                             f"detector {d.get('det_conf', 0):.1%}")
            if result.get("rejected"):
                lines.append(f"{result['rejected']} box(es) rejected as not a sign")
            t = result.get("timings", {})
            names = (("detect_ms", "detector"), ("classify_ms", "classifier"), ("api_ms", "API"))
            timing = [f"{name} {t[key]:.1f} ms" for key, name in names if key in t]
        else:
            lines.append(f"<b>{result.get('label', '?')}</b> ({result.get('confidence', 0):.1%})")
            for item in result.get("top_k", [])[1:]:
                lines.append(f"{item['label']} ({item['confidence']:.1%})")
            timing = [f"{name} {result[key]} ms" for key, name in
                      (("inference_ms", "inference"), ("api_ms", "API")) if key in result]
        if timing:
            lines.append("<i>" + ", ".join(timing) + "</i>")
        return "<br>".join(lines)

    def _on_error(self, message: str, backend_offline: bool):
        self.choose_button.setEnabled(True)
        self.result_label.setText("")
        self.banner.show_error(message)

    # --- showing the image ----------------------------------------------
    def _refresh_image(self, pixmap: QPixmap | None):
        if pixmap is None:
            return
        self._shown = pixmap
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, "_shown", None) is not None:
            self._refresh_image(self._shown)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UploadTab()
    window.setWindowTitle("SafeRoute - Upload tab")
    window.resize(700, 600)
    window.show()
    sys.exit(app.exec())
