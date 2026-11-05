"""Experiment configuration: defaults, JSON loading and command-line overrides."""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Config:
    name: str = "baseline"
    seed: int = 2026

    # Data
    val_per_class: int = 500        # 500 x 10 classes = 5,000 validation images
    full_train: bool = False        # True: train on all 50k images, no validation
    crop_padding: int = 4           # RandomCrop padding; 0 disables
    hflip: bool = True
    cutout: int = 0                 # Cutout square size in pixels; 0 disables
    color_jitter: float = 0.0       # brightness/contrast/saturation factor range 1 +- this; 0 disables
    trivial_augment: bool = False   # TrivialAugmentWide: one random op per image

    # Model
    widths: tuple = (32, 64, 128, 256)   # channels per stage
    blocks: tuple = (2, 2, 2, 2)         # residual blocks per stage
    kernel_size: int = 3
    dropout: float = 0.0                 # before the final linear layer

    # Optimization
    epochs: int = 100
    batch_size: int = 256
    lr: float = 0.1
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 5e-4
    warmup_epochs: int = 0
    label_smoothing: float = 0.0
    ema_decay: float = 0.0          # evaluate and save an EMA of the weights, e.g. 0.999; 0 disables

    # Runtime
    amp: bool = True
    channels_last: bool = True


def _parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_config(source=None, overrides=()):
    """Build a Config from a JSON path or dict, then apply "key=value" overrides."""
    if isinstance(source, dict):
        values = dict(source)
    elif source:
        values = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        values = {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override must be key=value, got {item!r}")
        values[key] = _parse_value(value)

    valid = {f.name for f in fields(Config)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    cfg = Config(**{**asdict(Config()), **values})
    cfg.widths = tuple(cfg.widths)
    cfg.blocks = tuple(cfg.blocks)
    if len(cfg.widths) != len(cfg.blocks):
        raise ValueError("widths and blocks must have the same length")
    return cfg
