"""Image upload tab for the SafeRoute GUI.

Flow: pick an image -> send it to the backend -> show the result.

Backend response from POST /predict/image (classifier only):
    {"label": str, "confidence": float,
     "top_k": [{"label": str, "confidence": float}, ...],
     "inference_ms": float, "api_ms": float}

If the response also contains a "detections" list (the future /detect/image
endpoint), boxes and labels are drawn on the image. Each detection is expected
to look like {"box": [x1, y1, x2, y2], "label": str, "confidence": float}.
Adjust _draw_detections() once the real /detect/image format is final.

Test by itself:
    python -m saferoute.gui.upload_tab
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests
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

API_URL = "http://127.0.0.1:8000"
ENDPOINT = "/predict/image"  # change to "/detect/image" once that endpoint exists
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
            data = requests.get(f"{self.base_url}/health", timeout=2).json()
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
                response = requests.post(self.url, files={"file": (path.name, f, mime)}, timeout=15)
        except (requests.ConnectionError, requests.Timeout):
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
        text = f"{d.get('label', '?')} {d.get('confidence', 0):.0%}"
        painter.drawText(x1 + 2, max(font.pixelSize(), y1 - 4), text)
    painter.end()
    return pixmap


class UploadTab(QWidget):
    def __init__(self, api_url: str = API_URL, endpoint: str = ENDPOINT, parent=None):
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

        lines = [f"<b>{result.get('label', '?')}</b> ({result.get('confidence', 0):.1%})"]
        for item in result.get("top_k", [])[1:]:
            lines.append(f"{item['label']} ({item['confidence']:.1%})")
        timing = []
        if "inference_ms" in result:
            timing.append(f"inference {result['inference_ms']} ms")
        if "api_ms" in result:
            timing.append(f"API {result['api_ms']} ms")
        if timing:
            lines.append("<i>" + ", ".join(timing) + "</i>")
        self.result_label.setText("<br>".join(lines))

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
