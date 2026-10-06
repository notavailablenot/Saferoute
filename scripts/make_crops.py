"""Turn the YOLO detection dataset into an ImageFolder classification dataset of sign crops.

The crops feed the Sprint 1 baseline CNN (and later the optional second-stage
classifier in Proposal Section 5.2.2). Split membership is inherited from the
detection split, so the route-level split is preserved (no leakage).

Output layout: <out>/<split>/<CLASS_NAME>/<image_stem>_<i>.jpg
Usage:
    python scripts/make_crops.py --root data/processed/saferoute_ph \
        --classes configs/classes.yaml --out data/processed/crops --pad 0.15 --min-px 16
"""
import argparse
from pathlib import Path

import yaml
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def load_names(classes_yaml: Path) -> dict[int, str]:
    data = yaml.safe_load(classes_yaml.read_text())["classes"]
    return {int(k): v["name"] for k, v in data.items()}


def make_crops(root: Path, names: dict[int, str], out: Path, pad: float = 0.15, min_px: int = 16) -> int:
    n = 0
    for split in ("train", "val", "test"):
        img_dir, lbl_dir = root / "images" / split, root / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT):
            lbl = lbl_dir / img_path.relative_to(img_dir).with_suffix(".txt")
            if not lbl.exists():
                continue
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                W, H = im.size
                for i, ln in enumerate(l for l in lbl.read_text().splitlines() if l.strip()):
                    cls, xc, yc, bw, bh = ln.split()
                    cls = int(cls)
                    xc, yc, bw, bh = float(xc) * W, float(yc) * H, float(bw) * W, float(bh) * H
                    if bw < min_px or bh < min_px:
                        continue  # too small to classify reliably
                    side = max(bw, bh) * (1 + 2 * pad)  # square crop with context padding
                    x0, y0 = max(0, xc - side / 2), max(0, yc - side / 2)
                    x1, y1 = min(W, xc + side / 2), min(H, yc + side / 2)
                    dst = out / split / names[cls] / f"{img_path.stem}_{i}.jpg"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    im.crop((int(x0), int(y0), int(x1), int(y1))).save(dst, quality=95)
                    n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/crops"))
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--min-px", type=int, default=16)
    a = ap.parse_args()
    print(f"wrote {make_crops(a.root, load_names(a.classes), a.out, a.pad, a.min_px)} crops to {a.out}")
