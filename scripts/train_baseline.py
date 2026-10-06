"""Train the Sprint 1 baseline CNN and save loss/accuracy curves (rubric: Baseline Results).

Usage: python scripts/train_baseline.py --crops data/processed/crops --epochs 20
Outputs: models/baseline_cnn.pt, reports/baseline_history.json, reports/baseline_curves.png
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from saferoute.vision.baseline_cnn import BaselineSignCNN
from saferoute.vision.datasets import build_loaders


def run_epoch(model, loader, loss_fn, device, opt=None):
    training = opt is not None
    model.train(training)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="data/processed/crops")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl, test_dl, classes = build_loaders(a.crops)
    model = BaselineSignCNN(num_classes=len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss()
    hist, best = [], 0.0
    Path("models").mkdir(exist_ok=True); Path("reports").mkdir(exist_ok=True)
    for ep in range(1, a.epochs + 1):
        tl, ta = run_epoch(model, train_dl, loss_fn, device, opt)
        vl, va = run_epoch(model, val_dl, loss_fn, device)
        hist.append({"epoch": ep, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va})
        print(f"ep {ep:02d} train {tl:.3f}/{ta:.3f}  val {vl:.3f}/{va:.3f}")
        if va > best:
            best = va
            torch.save({"state_dict": model.state_dict(), "classes": classes}, "models/baseline_cnn.pt")
    _, test_acc = run_epoch(model, test_dl, loss_fn, device)  # report once; never used for selection
    Path("reports/baseline_history.json").write_text(json.dumps({"history": hist, "best_val_acc": best,
                                                                 "test_acc_last_epoch": test_acc}, indent=2))
    try:
        import matplotlib.pyplot as plt
        e = [h["epoch"] for h in hist]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(e, [h["train_loss"] for h in hist], label="train"); ax[0].plot(e, [h["val_loss"] for h in hist], label="val")
        ax[0].set_title("Loss"); ax[0].legend()
        ax[1].plot(e, [h["train_acc"] for h in hist], label="train"); ax[1].plot(e, [h["val_acc"] for h in hist], label="val")
        ax[1].set_title("Accuracy"); ax[1].legend()
        fig.tight_layout(); fig.savefig("reports/baseline_curves.png", dpi=150)
    except ImportError:
        print("matplotlib not installed; curves not plotted")


if __name__ == "__main__":
    main()
