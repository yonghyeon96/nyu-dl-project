"""ResNet for 32x32 images with configurable width/depth, plus the 5M trainable-parameter check.

Print a model's parameter count without training (from the repository root):
    python -m src.model --config configs/baseline.json
    python -m src.model --config configs/baseline.json --set "widths=[40,80,160,320]"
"""
import argparse

from torch import nn
from torch.nn import functional as F

PARAM_LIMIT = 5_000_000


class BasicBlock(nn.Module):
    """y = ReLU(F(x) + S(x)): two conv-BN layers plus an identity or 1x1 projection shortcut."""

    def __init__(self, c_in, c_out, stride, kernel_size):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    """CIFAR-style ResNet: 3x3 stem without max-pool, then stages that halve the resolution."""

    def __init__(self, widths, blocks, kernel_size=3, dropout=0.0, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        stages, c_in = [], widths[0]
        for i, (c_out, n_blocks) in enumerate(zip(widths, blocks)):
            layers = []
            for j in range(n_blocks):
                stride = 2 if (i > 0 and j == 0) else 1
                layers.append(BasicBlock(c_in, c_out, stride, kernel_size))
                c_in = c_out
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(c_in, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg):
    """Build the model from a Config and refuse anything over the competition limit."""
    model = ResNet(cfg.widths, cfg.blocks, cfg.kernel_size, cfg.dropout)
    n = count_params(model)
    if n > PARAM_LIMIT:
        raise ValueError(f"{n:,} trainable parameters exceeds the {PARAM_LIMIT:,} limit")
    return model


def describe(cfg):
    """Print parameters per stage and the total against the limit; return the total."""
    model = ResNet(cfg.widths, cfg.blocks, cfg.kernel_size, cfg.dropout)
    print(f"widths={cfg.widths} blocks={cfg.blocks} kernel={cfg.kernel_size}")
    print(f"  stem    {count_params(model.stem):>10,}")
    for i, stage in enumerate(model.stages):
        print(f"  stage{i + 1}  {count_params(stage):>10,}")
    print(f"  head    {count_params(model.head):>10,}")
    n = count_params(model)
    print(f"  total   {n:>10,}  ({n / PARAM_LIMIT:.1%} of the 5M limit) -> {'OK' if n <= PARAM_LIMIT else 'TOO LARGE'}")
    return n


if __name__ == "__main__":
    from .config import load_config

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    describe(load_config(args.config, args.set))
