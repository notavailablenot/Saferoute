import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import import_gtsrb  # noqa: E402
import import_roboflow  # noqa: E402


def _rf(root, split, stem, lines):
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48)).save(root / split / "images" / f"{stem}.jpg")
    (root / split / "labels" / f"{stem}.txt").write_text("\n".join(lines))


def test_roboflow_dedup_group_split_and_polygon(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    # three augmented copies of one photo -> must collapse to one
    for h in ("aaa", "bbb", "ccc"):
        _rf(src, "train", f"Placa1_D1_JPG.rf.{h}", ["0 0.5 0.5 0.1 0.1"])
    _rf(src, "valid", "Placa1_D2_JPG.rf.ddd", ["0 0.4 0.4 0.1 0.1"])  # same sign, other photo
    _rf(src, "train", "Placa2_D1_JPG.rf.eee", ["0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3"])  # polygon
    _rf(src, "test", "Placa3_E1_JPG.rf.fff", [])  # background image
    for i in range(4, 12):
        _rf(src, "train", f"Placa{i}_D1_JPG.rf.x{i}", ["0 0.5 0.5 0.2 0.2"])
    stats = import_roboflow.run(src, dst)
    assert stats["train"] + stats["val"] + stats["test"] == 12  # 14 files -> 12 unique photos (3 aug copies -> 1)
    # both photos of Placa1 land in the SAME split
    where = {p.parent.name for p in dst.glob("images/*/Placa1_*")}
    assert len(where) == 1
    poly = next(dst.glob("labels/*/Placa2_D1_JPG.txt")).read_text().split()
    assert poly[0] == "0" and abs(float(poly[3]) - 0.2) < 1e-6  # polygon -> box width 0.2
    assert stats["background"] == 1
    assert f"path: {dst.resolve()}" in (dst / "data.yaml").read_text()


def test_gtsrb_mapping_and_track_split(tmp_path):
    train, test = [], []
    for cls in (14, 2):  # 14=STOP (kept), 2=50 km/h (not a SafeRoute class -> dropped)
        for t in range(10):
            for f in range(3):
                p = tmp_path / "Training" / f"{cls:05d}" / f"{t:05d}_{f:05d}.ppm"
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (40, 40)).save(p)
                train.append((str(p), cls))
    tp = tmp_path / "Final_Test" / "00001.ppm"
    tp.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40)).save(tp)
    test = [(str(tp), 14), (str(tp), 2)]
    splits = import_gtsrb.map_and_split(train, test, val_frac=0.2)
    names = {n for s in splits.values() for _, n in s}
    assert names == {"STOP"}
    # no track appears in both train and val
    tr = {import_gtsrb.track_id(p) for p, _ in splits["train"]}
    va = {import_gtsrb.track_id(p) for p, _ in splits["val"]}
    assert tr.isdisjoint(va) and len(va) == 2
    counts = import_gtsrb.export(splits, tmp_path / "out")
    assert counts["train"]["STOP"] == 24 and counts["val"]["STOP"] == 6 and counts["test"]["STOP"] == 1
    assert (tmp_path / "out" / "test" / "STOP").is_dir()
