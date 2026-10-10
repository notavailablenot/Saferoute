"""Label the class of every ground-truth sign in the detector TEST split (about 185 signs).

The PauloLab photos only say "sign", so this small PyQt6 tool lets a team member assign one of
the 10 SafeRoute classes (or OTHER) to each box. scripts/evaluate.py cascade then measures how
well the GTSRB-trained classifier works on real road photos (the domain-gap number for the report).

    python scripts/label_crops.py                    # resumes where you stopped
Keys:  0 STOP  1 YIELD  2 NO_ENTRY  3 SL20  4 SL30  5 SL60  6 SL80  7 SL100  8 PED  9 SCHOOL
       O OTHER (a sign that is not one of the 10)   S SKIP (unreadable)   B back   Q save and quit
Output: reports/cascade_labels.csv (image, box index, x1, y1, x2, y2, label); saved after every key.
Uses PyQt6 (same as the SafeRoute GUI) instead of cv2.imshow, whose bundled Qt "xcb" plugin
fails to load on some WSL setups.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from saferoute.gui.qt_env import fix_qt_env  # noqa: E402
FIELDS = ["image", "box", "x1", "y1", "x2", "y2", "label"]


def to_pixmap(bgr: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


def load_items(data: Path, split: str):
    items = []
    for img in sorted((data / "images" / split).glob("*")):
        lab = data / "labels" / split / (img.stem + ".txt")
        if not lab.exists():
            continue
        lines = [ln.split() for ln in lab.read_text().splitlines() if len(ln.split()) >= 5]
        for k, parts in enumerate(lines):
            items.append((str(img), k, [float(v) for v in parts[1:5]]))
    return items


class Labeler(QWidget):
    def __init__(self, items, out: Path, names: list[str]):
        super().__init__()
        self.items, self.out = items, out
        self.keymap = {getattr(Qt.Key, f"Key_{i}"): n for i, n in enumerate(names)}
        self.keymap |= {Qt.Key.Key_O: "OTHER", Qt.Key.Key_S: "SKIP"}
        self.done = {}
        if out.exists():
            for r in csv.DictReader(open(out)):
                self.done[(r["image"], int(r["box"]))] = r
        self.i = next((j for j, it in enumerate(items) if (it[0], it[1]) not in self.done), len(items))

        self.setWindowTitle("SafeRoute crop labeler")
        self.info = QLabel()
        self.info.setFont(QFont("Arial", 12))
        self.ctx = QLabel()
        self.crop = QLabel()
        self.crop.setFixedSize(360, 360)
        self.crop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        keys = QLabel("  ".join(f"{i}={n}" for i, n in enumerate(names)) +
                      "\nO=OTHER (not one of the 10)   S=SKIP (unreadable)   B=back   Q=save and quit")
        keys.setWordWrap(True)
        row = QHBoxLayout()
        row.addWidget(self.ctx)
        row.addWidget(self.crop)
        lay = QVBoxLayout(self)
        lay.addWidget(self.info)
        lay.addLayout(row)
        lay.addWidget(keys)
        print(f"{len(items)} signs, {len(self.done)} already labeled")
        self.show_item()

    def show_item(self):
        if self.i >= len(self.items):
            self.info.setText(f"All {len(self.items)} signs labeled. Press Q to quit.")
            return
        path, k, (cx, cy, bw, bh) = self.items[self.i]
        img = cv2.imread(path)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
        self.box = (x1, y1, x2, y2)
        m = 0.3 * max(x2 - x1, y2 - y1)
        crop = img[int(max(0, y1 - m)):int(min(h, y2 + m)), int(max(0, x1 - m)):int(min(w, x2 + m))]
        ctx = img.copy()
        cv2.rectangle(ctx, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), max(2, w // 300))
        self.ctx.setPixmap(to_pixmap(ctx).scaled(720, 540, Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation))
        if crop.size:
            self.crop.setPixmap(to_pixmap(crop).scaled(360, 360, Qt.AspectRatioMode.KeepAspectRatio,
                                                       Qt.TransformationMode.SmoothTransformation))
        prev = self.done.get((path, k), {}).get("label", "-")
        self.info.setText(f"{self.i + 1}/{len(self.items)}   {Path(path).name}  box {k}   current label: {prev}")

    def save(self):
        self.out.parent.mkdir(exist_ok=True)
        with open(self.out, "w", newline="") as f:
            wr = csv.DictWriter(f, FIELDS)
            wr.writeheader()
            wr.writerows(self.done.values())

    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key.Key_Q:
            self.save()
            print(f"saved {len(self.done)} labels to {self.out}")
            self.close()
        elif key == Qt.Key.Key_B:
            self.i = max(0, self.i - 1)
            self.show_item()
        elif key in self.keymap and self.i < len(self.items):
            path, k, _ = self.items[self.i]
            x1, y1, x2, y2 = self.box
            self.done[(path, k)] = {"image": path, "box": k, "x1": round(x1, 1), "y1": round(y1, 1),
                                    "x2": round(x2, 1), "y2": round(y2, 1), "label": self.keymap[key]}
            self.save()  # save after every key so nothing is lost
            self.i += 1
            self.show_item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/signs_v0")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="reports/cascade_labels.csv")
    a = ap.parse_args()
    data = yaml.safe_load((ROOT / "configs/classes.yaml").read_text())["classes"]
    names = [data[k]["name"] for k in sorted(data, key=int)]
    items = load_items(Path(a.data), a.split)
    if not items:
        raise SystemExit(f"no labeled images found in {a.data}/images/{a.split}")
    fix_qt_env()
    app = QApplication(sys.argv)
    w = Labeler(items, Path(a.out), names)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
