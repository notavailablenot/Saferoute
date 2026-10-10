"""Error banners for the SafeRoute GUI.

A Banner is a red bar that is hidden by default. Call show_error(text) to show
it and clear() to hide it again. The same widget is reused for every error.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel

BACKEND_OFFLINE = "Backend offline: start the API (uvicorn saferoute.api.main:app) and try again."
CAMERA_DISCONNECTED = "Camera disconnected: check the webcam or video file and try again."


class Banner(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "background-color: #c62828; color: white; padding: 8px; font-weight: bold;"
        )
        self.hide()

    def show_error(self, text: str) -> None:
        self.setText(text)
        self.show()

    def clear(self) -> None:
        self.setText("")
        self.hide()
