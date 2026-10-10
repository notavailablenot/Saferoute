"""Label the class of every ground-truth sign in the detector TEST split (about 185 signs).

The PauloLab photos only say "sign", so this small tool lets a team member assign one of the
10 SafeRoute classes (or OTHER) to each box. scripts/evaluate.py cascade then measures how well
the GTSRB-trained classifier works on real road photos (the domain-gap number for the report).

    python scripts/label_crops.py                    # resumes where you stopped
Keys:  0 STOP  1 YIELD  2 NO_ENTRY  3 SL20  4 SL30  5 SL60  6 SL80  7 SL100  8 PED  9 SCHOOL
       o OTHER (a sign that is not one of the 10)   s SKIP (unreadable)   b back   q save and quit
Output: reports/cascade_labels.csv (image, box index, x1, y1, x2, y2, label)
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["image", "box", "x1", "y1", "x2", "y2", "label"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/signs_v0")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="reports/cascade_labels.csv")
    a = ap.parse_args()
    data = yaml.safe_load((ROOT / "configs/classes.yaml").read_text())["classes"]
    names = [data[k]["name"] for k in sorted(data, key=int)]
    keymap = {ord(str(i)): n for i, n in enumerate(names)} | {ord("o"): "OTHER", ord("s"): "SKIP"}

    items = []
    for img in sorted((Path(a.data) / "images" / a.split).glob("*")):
        lab = Path(a.data) / "labels" / a.split / (img.stem + ".txt")
        if not lab.exists():
            continue
        lines = [ln.split() for ln in lab.read_text().splitlines() if len(ln.split()) >= 5]
        for k, parts in enumerate(lines):
            items.append((str(img), k, [float(v) for v in parts[1:5]]))
    out = Path(a.out)
    out.parent.mkdir(exist_ok=True)
    done = {}
    if out.exists():
        for r in csv.DictReader(open(out)):
            done[(r["image"], int(r["box"]))] = r
    print(f"{len(items)} signs, {len(done)} already labeled")

    i = next((j for j, it in enumerate(items) if (it[0], it[1]) not in done), len(items))
    while 0 <= i < len(items):
        path, k, (cx, cy, bw, bh) = items[i]
        img = cv2.imread(path)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
        m = 0.3 * max(x2 - x1, y2 - y1)
        crop = img[int(max(0, y1 - m)):int(min(h, y2 + m)), int(max(0, x1 - m)):int(min(w, x2 + m))]
        if crop.size == 0:
            i += 1
            continue
        crop = cv2.resize(crop, (360, max(1, int(360 * crop.shape[0] / max(crop.shape[1], 1)))))
        ctx = img.copy()
        cv2.rectangle(ctx, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        ctx = cv2.resize(ctx, (640, int(640 * h / w)))
        canvas = np.zeros((max(ctx.shape[0], crop.shape[0]) + 40, 1000 + 20, 3), np.uint8)
        canvas[40:40 + ctx.shape[0], :640] = ctx
        canvas[40:40 + crop.shape[0], 660:660 + crop.shape[1]] = crop[:, :360]
        prev = done.get((path, k), {}).get("label", "")
        cv2.putText(canvas, f"{i + 1}/{len(items)}  {Path(path).name} box {k}  current: {prev}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("SafeRoute crop labeler (0-9, o, s, b, q)", canvas)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            i = max(0, i - 1)
            continue
        if key in keymap:
            done[(path, k)] = {"image": path, "box": k, "x1": round(x1, 1), "y1": round(y1, 1),
                               "x2": round(x2, 1), "y2": round(y2, 1), "label": keymap[key]}
            i += 1
            with open(out, "w", newline="") as f:  # save after every key so nothing is lost
                wr = csv.DictWriter(f, FIELDS)
                wr.writeheader()
                wr.writerows(done.values())
    cv2.destroyAllWindows()
    print(f"saved {len(done)} labels to {out}")


if __name__ == "__main__":
    main()
