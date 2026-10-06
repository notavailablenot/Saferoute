"""Build the 16-class SafeRoute *crop classifier* dataset from GTSRB (proposal Section 6.1).

GTSRB (German Traffic Sign Recognition Benchmark) is already cropped, so it feeds the
baseline CNN directly. Only GTSRB classes that match a SafeRoute class are kept; the 6
SafeRoute classes with no GTSRB equivalent stay empty until Philippine photos are added.

Leakage control: GTSRB "Training" images come in *tracks* of ~30 frames of the SAME physical
sign. We hold out whole tracks for validation (never single frames). GTSRB "Final_Test"
becomes our test split.

Output: <out>/{train,val,test}/<SAFEROUTE_CLASS>/<name>.png  (ImageFolder layout)
Usage:
  python scripts/import_gtsrb.py --root data/raw/gtsrb --out data/processed/crops
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# GTSRB class id -> SafeRoute class name (configs/classes.yaml)
GTSRB_TO_SAFEROUTE = {
    0: "SPEED_LIMIT_20",
    1: "SPEED_LIMIT_30",
    3: "SPEED_LIMIT_60",
    5: "SPEED_LIMIT_80",
    7: "SPEED_LIMIT_100",
    13: "YIELD",
    14: "STOP",
    17: "NO_ENTRY",
    27: "PEDESTRIAN_CROSSING",  # GTSRB "pedestrians" warning: stand-in until PH photos exist
    28: "SCHOOL_ZONE",          # GTSRB "children crossing": stand-in until PH photos exist
}


def track_id(path: str | Path) -> str:
    """Training files are named <track>_<frame>.ppm inside a per-class folder."""
    p = Path(path)
    return f"{p.parent.name}/{p.stem.split('_')[0]}"


def map_and_split(train_samples, test_samples, val_frac: float = 0.15, seed: int = 42):
    """samples: list of (path, gtsrb_class). Returns {split: [(path, saferoute_name), ...]}."""
    by_class_tracks = defaultdict(lambda: defaultdict(list))
    for path, cls in train_samples:
        if cls in GTSRB_TO_SAFEROUTE:
            by_class_tracks[cls][track_id(path)].append(path)
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    for cls, tracks in sorted(by_class_tracks.items()):
        ids = sorted(tracks)
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * val_frac))
        for i, t in enumerate(ids):
            split = "val" if i < n_val else "train"
            out[split] += [(p, GTSRB_TO_SAFEROUTE[cls]) for p in tracks[t]]
    out["test"] = [(p, GTSRB_TO_SAFEROUTE[c]) for p, c in test_samples if c in GTSRB_TO_SAFEROUTE]
    return out


def export(splits, out_dir: Path) -> dict:
    from PIL import Image
    counts = {s: Counter() for s in splits}
    for split, items in splits.items():
        for path, name in items:
            p = Path(path)
            dst = out_dir / split / name / f"{p.parent.name}_{p.stem}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(p) as im:
                im.convert("RGB").save(dst)
            counts[split][name] += 1
    return {s: dict(sorted(c.items())) for s, c in counts.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/gtsrb"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/crops"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    from torchvision.datasets import GTSRB  # downloads once (~350 MB), then works offline
    train = GTSRB(str(a.root), split="train", download=True)
    test = GTSRB(str(a.root), split="test", download=True)
    splits = map_and_split(train._samples, test._samples, a.val_frac, a.seed)
    counts = export(splits, a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "crop_counts.json").write_text(json.dumps(counts, indent=2))
    for s, c in counts.items():
        print(f"{s:5s} total={sum(c.values()):6d}  " + "  ".join(f"{k}={v}" for k, v in c.items()))
