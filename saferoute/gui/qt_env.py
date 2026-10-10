"""Keep OpenCV's bundled Qt plugins away from PyQt6.

`import cv2` (opencv-python wheel) sets QT_QPA_PLATFORM_PLUGIN_PATH to its own Qt "xcb" plugin.
PyQt6 then tries to load that incompatible plugin and aborts with
'Could not load the Qt platform plugin "xcb"'. Call fix_qt_env() before creating QApplication.
"""
import os


def fix_qt_env() -> None:
    for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        if "cv2" in os.environ.get(key, ""):
            os.environ.pop(key)
