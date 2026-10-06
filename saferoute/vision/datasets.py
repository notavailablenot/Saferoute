"""DataLoaders for the Sprint 1 baseline CNN (ImageFolder of sign crops from make_crops.py).

Transforms MUST match ClassifierEngine.preprocess at inference (resize 64, ImageNet mean/std).
Horizontal flips are deliberately NOT used: mirroring turns NO_LEFT_TURN into NO_RIGHT_TURN.
"""
from collections import Counter
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_transforms(img_size: int = 64, train: bool = False):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3),  # night/glare proxy
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),             # motion/rain blur proxy
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def class_names_from_yaml(path: str | Path = "configs/classes.yaml") -> list[str]:
    data = yaml.safe_load(Path(path).read_text())["classes"]
    return [data[k]["name"] for k in sorted(data, key=int)]


class OrderedImageFolder(datasets.ImageFolder):
    """ImageFolder whose class indices follow configs/classes.yaml, NOT alphabetical folder order.

    Plain ImageFolder would make NO_ENTRY index 0, while the API and YOLO data.yaml use STOP = 0,
    which would silently mislabel every prediction after ONNX export.
    """

    def __init__(self, root, class_names: list[str], transform=None):
        self._names = list(class_names)
        try:  # torchvision >= 0.17: tolerate classes with no crops in a split (common in val/test)
            super().__init__(root, transform=transform, allow_empty=True)
        except TypeError:
            super().__init__(root, transform=transform)

    def find_classes(self, directory):
        return self._names, {n: i for i, n in enumerate(self._names)}


def build_loaders(crops_root: str | Path, img_size: int = 64, batch_size: int = 64,
                  num_workers: int = 2, balance: bool = True,
                  classes_yaml: str | Path = "configs/classes.yaml"):
    root = Path(crops_root)
    names = class_names_from_yaml(classes_yaml)
    train_ds = OrderedImageFolder(root / "train", names, build_transforms(img_size, train=True))
    val_ds = OrderedImageFolder(root / "val", names, build_transforms(img_size))
    test_ds = OrderedImageFolder(root / "test", names, build_transforms(img_size))
    sampler = None
    if balance:  # counter class imbalance (rubric: verify class balance BEFORE training)
        counts = Counter(train_ds.targets)
        weights = [1.0 / counts[t] for t in train_ds.targets]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=sampler is None,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
    return train_dl, val_dl, test_dl, train_ds.classes
