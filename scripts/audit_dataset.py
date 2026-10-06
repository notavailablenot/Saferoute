"""Sprint 1 dataset audit for a YOLO-format dataset (images/<split>, labels/<split>).

Checks: unreadable/corrupted images, images without labels, empty label files,
malformed lines, invalid class IDs, boxes outside [0, 1], tiny boxes, and the
per-split class distribution. Writes a JSON report and exits non-zero on hard errors.

Usage:
    python scripts/audit_dataset.py --root data/processed/saferoute_ph --num-classes 16 \
        --out reports/dataset_audit.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def audit(root: Path, num_classes: int, min_box_px: int = 12) -> dict:
    report = {"splits": {}, "errors": [], "warnings": []}
    for split in ("train", "val", "test"):
        img_dir, lbl_dir = root / "images" / split, root / "labels" / split
        if not img_dir.exists():
            report["warnings"].append(f"missing split: {split}")
            continue
        counts, n_img, n_bg = Counter(), 0, 0
        for img_path in sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT):
            n_img += 1
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception as exc:  # corrupted / truncated
                report["errors"].append(f"corrupted image {img_path}: {exc}")
                continue
            lbl = lbl_dir / img_path.relative_to(img_dir).with_suffix(".txt")
            if not lbl.exists():
                report["errors"].append(f"no label file for {img_path}")
                continue
            lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
            if not lines:
                n_bg += 1  # background image: allowed, but counted
            for i, ln in enumerate(lines, 1):
                parts = ln.split()
                if len(parts) != 5:
                    report["errors"].append(f"{lbl}:{i} malformed line '{ln}'")
                    continue
                try:
                    cls = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])
                except ValueError:
                    report["errors"].append(f"{lbl}:{i} non-numeric values '{ln}'")
                    continue
                if not 0 <= cls < num_classes:
                    report["errors"].append(f"{lbl}:{i} invalid class id {cls}")
                    continue
                if not all(0.0 <= v <= 1.0 for v in (xc, yc, bw, bh)):
                    report["errors"].append(f"{lbl}:{i} box outside [0,1]")
                    continue
                if bw * w < min_box_px or bh * h < min_box_px:
                    report["warnings"].append(f"{lbl}:{i} box smaller than {min_box_px}px")
                counts[cls] += 1
        report["splits"][split] = {
            "images": n_img,
            "background_images": n_bg,
            "instances": sum(counts.values()),
            "per_class": {str(k): counts.get(k, 0) for k in range(num_classes)},
        }
    # class-balance warnings on the training split
    train = report["splits"].get("train")
    if train:
        for k, v in train["per_class"].items():
            if v < 150:
                report["warnings"].append(f"class {k} has only {v} training instances (target >= 150)")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--num-classes", type=int, default=16)
    ap.add_argument("--min-box-px", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("reports/dataset_audit.json"))
    a = ap.parse_args(argv)
    rep = audit(a.root, a.num_classes, a.min_box_px)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=2))
    for split, s in rep["splits"].items():
        print(f"{split:5s} images={s['images']:5d} instances={s['instances']:6d} background={s['background_images']}")
    print(f"errors={len(rep['errors'])} warnings={len(rep['warnings'])} -> {a.out}")
    return 1 if rep["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
