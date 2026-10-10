"""Baseline CNN for traffic-sign crop classification (10 SafeRoute classes + NOT_SIGN rejection class).

Follows the Module 2 baseline: 3 blocks of Conv -> ReLU -> MaxPool (32 -> 64 -> 128 filters),
then Flatten -> Linear -> ReLU -> Dropout -> Linear. The model returns raw logits; softmax is
applied at inference time (ClassifierEngine) because nn.CrossEntropyLoss expects logits.

Shape walk-through for a 64 x 64 RGB input (batch N):
    input            N x 3   x 64 x 64
    block1 (32)      N x 32  x 32 x 32
    block2 (64)      N x 64  x 16 x 16
    block3 (128)     N x 128 x 8  x 8
    flatten          N x 8192
    fc1 + dropout    N x 256
    fc2 (logits)     N x 11
Trainable parameters: 2,193,483 (conv 93,248 + fc 2,100,235). Sprint 1 used 10 outputs (2,193,226).
"""
import torch
from torch import nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class BaselineSignCNN(nn.Module):
    def __init__(self, num_classes: int = 11, img_size: int = 64, dropout: float = 0.5):
        super().__init__()
        if img_size % 8:
            raise ValueError("img_size must be divisible by 8 (three 2x2 max-pools)")
        self.features = nn.Sequential(conv_block(3, 32), conv_block(32, 64), conv_block(64, 128))
        side = img_size // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * side * side, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


if __name__ == "__main__":
    # Rubric check: "verify shapes and parameter counts".
    model = BaselineSignCNN()
    x = torch.randn(4, 3, 64, 64)
    for name, layer in [("features", model.features), ("classifier", model.classifier)]:
        x = layer(x)
        print(f"{name:10s} -> {tuple(x.shape)}")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n:,}")
    assert x.shape == (4, 11), "output must equal the number of classes"
    assert n == 2_193_483
    try:
        from torchinfo import summary  # pip install torchinfo
        summary(model, input_size=(1, 3, 64, 64))
    except ImportError:
        pass
