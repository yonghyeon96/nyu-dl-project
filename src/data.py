"""Data pipeline: PNG -> .npy cache, stratified train/val split, and a GPU batch loader with augmentation.

The evaluation images are only decoded into the cache for prediction. They are never
used for training, validation or normalization statistics.
"""
import csv
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .augment import color_jitter, trivial_augment

ROOT = Path(__file__).resolve().parent.parent   # repository root
CACHE_DIR = ROOT / "cache"
SPLIT_DIR = ROOT / "splits"
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


# ---------------------------------------------------------------- locating data

def _find_in(base):
    # The data may sit one level down, e.g. when a zip contained a top-level folder.
    for cand in [base, *sorted(p for p in base.iterdir() if p.is_dir())]:
        if (cand / "train_labels.csv").is_file():
            return cand
    return None


def find_data_dir(explicit=None):
    """Folder holding train_labels.csv: explicit arg > $DATA_DIR > HPC scratch > ./data.

    If a candidate folder only holds the uploaded competition .zip, it is extracted
    there once (the zip is kept).
    """
    if explicit:
        bases = [Path(explicit)]
    elif os.environ.get("DATA_DIR"):
        bases = [Path(os.environ["DATA_DIR"])]
    else:
        user = os.environ.get("USER", "")
        bases = [Path(f"/scratch/{user}/cifar_data")] if user else []
        bases += [ROOT / "data"]
    for base in bases:
        if not base.is_dir():
            continue
        found = _find_in(base)
        if found is None and (zips := sorted(base.glob("*.zip"))):
            print(f"Extracting {zips[0]} ...")
            with zipfile.ZipFile(zips[0]) as zf:
                zf.extractall(base)
            found = _find_in(base)
        if found is not None:
            return found
    raise FileNotFoundError(
        f"train_labels.csv not found under {[str(b) for b in bases]}. "
        "Put the competition files (or the competition .zip) in one of these folders.")


# ---------------------------------------------------------------- .npy cache

def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_images(folder, filenames):
    out = np.empty((len(filenames), 32, 32, 3), dtype=np.uint8)
    for i, fn in enumerate(filenames):
        with Image.open(folder / fn) as im:
            out[i] = np.asarray(im.convert("RGB"))
    return out


def _channel_stats(x, chunk=5000):
    """Per-channel mean/std in [0, 1], computed in chunks to keep memory low."""
    s = np.zeros(3); ss = np.zeros(3); n = 0
    for i in range(0, len(x), chunk):
        part = x[i:i + chunk].reshape(-1, 3).astype(np.float64) / 255.0
        s += part.sum(0); ss += (part ** 2).sum(0); n += len(part)
    mean = s / n
    return mean.tolist(), np.sqrt(ss / n - mean ** 2).tolist()


def build_cache(data_dir, cache_dir=CACHE_DIR):
    """Decode every PNG once into .npy arrays. Rows follow the CSV order. Skips if already built."""
    cache_dir = Path(cache_dir)
    if (cache_dir / "meta.json").is_file():
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"Building cache from {data_dir} ...")

    train = _read_csv(data_dir / "train_labels.csv")
    test = _read_csv(data_dir / "test.csv")
    train_x = _load_images(data_dir / "train_images", [r["filename"] for r in train])
    test_x = _load_images(data_dir / "test_images", [r["filename"] for r in test])
    arrays = {
        "train_x": train_x,
        "train_y": np.array([int(r["label"]) for r in train], dtype=np.int64),
        "train_ids": np.array([int(r["id"]) for r in train], dtype=np.int64),
        "test_x": test_x,
        "test_ids": np.array([int(r["id"]) for r in test], dtype=np.int64),
    }
    for name, arr in arrays.items():
        np.save(cache_dir / f"{name}.npy", arr)

    # Normalization statistics come from the labeled training images only.
    mean, std = _channel_stats(train_x)
    meta = {"data_dir": str(Path(data_dir).resolve()), "n_train": len(train_x), "n_test": len(test_x),
            "mean": mean, "std": std}
    # meta.json is written last so its presence marks a complete cache.
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Cache built in {time.perf_counter() - t0:.0f}s -> {cache_dir}")


def load_cache(cache_dir=CACHE_DIR):
    cache_dir = Path(cache_dir)
    data = {p.stem: np.load(p) for p in cache_dir.glob("*.npy")}
    data["meta"] = json.loads((cache_dir / "meta.json").read_text())
    return data


# ---------------------------------------------------------------- train/val split

def make_split(labels, val_per_class, seed, split_dir=SPLIT_DIR):
    """Stratified split by row index of train_labels.csv (row index == id). Saved so every run reuses it."""
    path = Path(split_dir) / f"split_seed{seed}_val{val_per_class}.json"
    if path.is_file():
        return json.loads(path.read_text())
    rng = np.random.default_rng(seed)
    val = []
    for c in range(len(CLASSES)):
        idx = np.flatnonzero(labels == c)
        val += rng.choice(idx, size=val_per_class, replace=False).tolist()
    val = sorted(val)
    val_set = set(val)
    split = {"seed": seed, "val_per_class": val_per_class,
             "train_idx": [i for i in range(len(labels)) if i not in val_set], "val_idx": val}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split))
    return split


# ---------------------------------------------------------------- GPU loader

class GpuLoader:
    """Keeps uint8 images on the device and yields normalized, augmented batches.

    The whole dataset (~150 MB) fits in GPU memory, so there is no CPU decoding
    or host-to-device copy per batch and the GPU stays busy.
    """

    def __init__(self, x, y, batch_size, device, mean, std, train=False,
                 crop_padding=0, hflip=False, cutout=0, color_jitter=0.0, trivial_augment=False,
                 channels_last=False, seed=0):
        self.x = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).contiguous()   # N,3,32,32 uint8
        self.y = torch.from_numpy(y).to(device) if y is not None else None
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.batch_size = batch_size
        self.train = train
        self.crop_padding, self.hflip, self.cutout = crop_padding, hflip, cutout
        self.color_jitter, self.trivial_augment = color_jitter, trivial_augment
        self.channels_last = channels_last
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    def __len__(self):
        n = len(self.x)
        return n // self.batch_size if self.train else -(-n // self.batch_size)

    def __iter__(self):
        n, dev = len(self.x), self.x.device
        order = (torch.randperm(n, device=dev, generator=self.gen) if self.train
                 else torch.arange(n, device=dev))
        for b in range(len(self)):
            idx = order[b * self.batch_size:(b + 1) * self.batch_size]
            xb = self.x[idx].float() / 255.0
            if self.train:
                xb = self._augment_pixels(xb)
            xb = (xb - self.mean) / self.std
            if self.train:
                xb = self._augment(xb)
            if self.channels_last:
                xb = xb.contiguous(memory_format=torch.channels_last)
            yield xb, (self.y[idx] if self.y is not None else None)

    def _augment_pixels(self, x):
        # Photometric augmentation needs real pixel values in [0, 1], so it runs before normalization.
        if self.trivial_augment:
            x = trivial_augment(x, self.gen)
        if self.color_jitter:
            x = color_jitter(x, self.color_jitter, self.gen)
        return x

    def _augment(self, x):
        B, C, H, W = x.shape
        dev = x.device
        if self.crop_padding:
            # RandomCrop with zero padding (zero == channel mean after normalization).
            p = self.crop_padding
            xp = F.pad(x, (p, p, p, p))
            oy = torch.randint(0, 2 * p + 1, (B,), device=dev, generator=self.gen)
            ox = torch.randint(0, 2 * p + 1, (B,), device=dev, generator=self.gen)
            ar = torch.arange(H, device=dev)
            rows = (oy[:, None] + ar)[:, None, :, None]
            cols = (ox[:, None] + ar)[:, None, None, :]
            x = xp[torch.arange(B, device=dev)[:, None, None, None],
                   torch.arange(C, device=dev)[None, :, None, None], rows, cols]
        if self.hflip:
            flip = torch.rand(B, device=dev, generator=self.gen) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
        if self.cutout:
            s = self.cutout
            cy = torch.randint(0, H, (B, 1, 1), device=dev, generator=self.gen) - s // 2
            cx = torch.randint(0, W, (B, 1, 1), device=dev, generator=self.gen) - s // 2
            ys = torch.arange(H, device=dev)[None, :, None]
            xs = torch.arange(W, device=dev)[None, None, :]
            hole = (ys >= cy) & (ys < cy + s) & (xs >= cx) & (xs < cx + s)
            x = x * (~hole)[:, None]
        return x
