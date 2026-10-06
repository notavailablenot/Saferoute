import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_dataset  # noqa: E402
import make_crops  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _mk(root, split, name, lines, size=(640, 480)):
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 120, 120)).save(root / "images" / split / f"{name}.jpg")
    (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines))


def test_audit_flags_bad_labels_and_counts_classes(tmp_path):
    _mk(tmp_path, "train", "ok", ["0 0.5 0.5 0.1 0.1", "5 0.2 0.2 0.05 0.08"])
    _mk(tmp_path, "train", "badcls", ["99 0.5 0.5 0.1 0.1"])
    _mk(tmp_path, "train", "oob", ["1 1.4 0.5 0.1 0.1"])
    _mk(tmp_path, "val", "bg", [])
    (tmp_path / "images" / "test").mkdir(parents=True)
    (tmp_path / "images" / "test" / "corrupt.jpg").write_bytes(b"not an image")
    rep = audit_dataset.audit(tmp_path, num_classes=16)
    assert rep["splits"]["train"]["per_class"]["0"] == 1
    assert rep["splits"]["train"]["per_class"]["5"] == 1
    assert rep["splits"]["val"]["background_images"] == 1
    joined = " ".join(rep["errors"])
    assert "invalid class id 99" in joined and "outside" in joined and "corrupted" in joined
    out = tmp_path / "r.json"
    assert audit_dataset.main(["--root", str(tmp_path), "--out", str(out)]) == 1
    assert json.loads(out.read_text())["errors"]


def test_make_crops_preserves_split_and_class(tmp_path):
    _mk(tmp_path, "train", "a", ["0 0.5 0.5 0.1 0.1", "13 0.3 0.3 0.01 0.01"])  # second box too small
    _mk(tmp_path, "test", "b", ["5 0.5 0.5 0.2 0.2"])
    names = make_crops.load_names(ROOT / "configs" / "classes.yaml")
    n = make_crops.make_crops(tmp_path, names, tmp_path / "crops")
    assert n == 2
    assert (tmp_path / "crops" / "train" / "STOP" / "a_0.jpg").exists()
    assert (tmp_path / "crops" / "test" / "SPEED_LIMIT_40" / "b_0.jpg").exists()
    assert not (tmp_path / "crops" / "train" / "PEDESTRIAN_CROSSING").exists()
