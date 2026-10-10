import os

import numpy as np
import pytest

from saferoute.gui.overlay import alert_text, draw_detections, scale_detections, speed_value

DET = {"box": [10, 10, 60, 60], "det_conf": 0.9, "label": "SPEED_LIMIT_60", "cls_conf": 0.95,
       "tier": 2, "track_id": 3}


def test_overlay_helpers():
    out = draw_detections(np.zeros((100, 100, 3), np.uint8), [DET])
    assert out.shape == (100, 100, 3) and out.sum() > 0
    assert scale_detections([DET], 2.0)[0]["box"] == [20, 20, 120, 120]
    assert speed_value("SPEED_LIMIT_60") == 60 and speed_value("STOP") is None
    assert alert_text("SIGN") == "Sign ahead" and alert_text("NO_ENTRY") == "NO ENTRY ahead"


def test_main_window_renders_offscreen():
    pytest.importorskip("PyQt6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from saferoute.gui.app import MainWindow
    app = QApplication.instance() or QApplication([])
    w = MainWindow("http://127.0.0.1:9", start_health=False)
    res = {"detections": [DET], "alerts": [],
           "timings": {"detect_ms": 4.0, "classify_ms": 0.5, "pipeline_ms": 5.0, "api_ms": 6.0}}
    w.image.show_result(np.zeros((360, 640, 3), np.uint8), res, 8.0)
    assert w.image.table.rowCount() == 1
    w.drive.on_alert({"label": "SPEED_LIMIT_60", "tier": 2, "track_id": 3, "confidence": 0.95})
    assert w.drive.badge.text() == "60" and w.drive.history.count() == 1
    w.close()
    app.processEvents()
