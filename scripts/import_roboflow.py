"""Import a Roboflow YOLO export as the SafeRoute single-class sign *detector* dataset.

What it does (and why):
  1. Reads <src>/{train,valid,test}/{images,labels} (Roboflow layout).
  2. Removes Roboflow augmentation duplicates: "X_jpg.rf.<hash>.jpg" copies of one photo
     collapse to ONE image (YOLO augments during training anyway).
  3. Re-splits 70/15/15 by *physical sign* (e.g. all "Placa774_*" photos stay together),
     so near-identical photos never sit in both train and test (data leakage).
  4. Converts any polygon labels to boxes, forces class 0 ("traffic_sign").
  5. Writes <dst>/images|labels/{train,val,test} and <dst>/data.yaml with an ABSOLUTE path
     (Ultralytics resolves relative paths against its own datasets folder).

Usage:
  python scripts/import_roboflow.py --src data/raw/ph_public_v0 --dst data/processed/signs_v0
"""
import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_DIRS = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def original_name(stem: str) -> str:
    """'Placa774_D1_JPG.rf.4edf29...' -> 'Placa774_D1_JPG' (one entry per original photo)."""
    return stem.split(".rf.")[0]


def group_key(orig: str) -> str:
    """Photos of the same physical sign share a group: 'Placa774_D1_JPG' -> 'Placa774'."""
    m = re.match(r"^([A-Za-z]+\d+)", orig)
    return m.group(1) if m else orig


def to_box_line(line: str) -> str | None:
    """Return a 'cls xc yc w h' line with class 0, converting polygons; None if unusable."""
    parts = line.split()
    if len(parts) < 5:
        return None
    vals = [float(v) for v in parts[1:]]
    if len(vals) == 4:
        xc, yc, w, h = vals
    elif len(vals) >= 6 and len(vals) % 2 == 0:  # polygon x1 y1 x2 y2 ...
        xs, ys = vals[0::2], vals[1::2]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        xc, yc, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    else:
        return None
    if w <= 0 or h <= 0:
        return None
    return f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def collect(src: Path) -> dict[str, tuple[Path, Path | None]]:
    """original name -> (image path, label path). First copy (sorted) wins for duplicates."""
    found: dict[str, tuple[Path, Path | None]] = {}
    for split in SPLIT_DIRS:
        img_dir = src / split / "images"
        if not img_dir.is_dir():
            continue
        for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
            orig = original_name(img.stem)
            if orig in found:
                continue
            lbl = src / split / "labels" / f"{img.stem}.txt"
            found[orig] = (img, lbl if lbl.exists() else None)
    return found


def split_groups(keys: list[str], seed: int = 42, val: float = 0.15, test: float = 0.15) -> dict[str, str]:
    groups = sorted(set(keys))
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_test, n_val = max(1, round(n * test)), max(1, round(n * val))
    assign = {}
    for i, g in enumerate(groups):
        assign[g] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return assign


def run(src: Path, dst: Path, class_name: str = "traffic_sign", seed: int = 42) -> dict:
    found = collect(src)
    assign = split_groups([group_key(o) for o in found], seed)
    stats = {"train": 0, "val": 0, "test": 0, "boxes": 0, "dropped_lines": 0, "background": 0}
    for orig, (img, lbl) in found.items():
        split = assign[group_key(orig)]
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst / "images" / split / f"{orig}{img.suffix.lower()}")
        lines = []
        if lbl is not None:
            for raw in lbl.read_text().splitlines():
                if not raw.strip():
                    continue
                conv = to_box_line(raw)
                if conv is None:
                    stats["dropped_lines"] += 1
                else:
                    lines.append(conv)
        if not lines:
            stats["background"] += 1
        stats["boxes"] += len(lines)
        (dst / "labels" / split / f"{orig}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats[split] += 1
    (dst / "data.yaml").write_text(
        f"path: {dst.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: {class_name}\n")
    stats["groups"] = len(set(assign))
    (dst / "import_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--class-name", default="traffic_sign")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    s = run(a.src, a.dst, a.class_name, a.seed)
    print(f"unique photos: train={s['train']} val={s['val']} test={s['test']} "
          f"(physical-sign groups={s['groups']}) boxes={s['boxes']} "
          f"background={s['background']} dropped_label_lines={s['dropped_lines']}")
    print(f"wrote {a.dst}/data.yaml")
