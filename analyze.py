"""Compare trained checkpoints on the validation images. Runs on CPU; give it about 16 GB of memory.

    python analyze.py --ckpts runs/<a>/best.pt runs/<b>/best.pt

Prints each model's accuracy with per-class results and its most common confusions, how often
each pair differs, and the accuracy of the averaged predictions. The ensemble line estimates how
much a distillation student could gain; the ensemble itself is not a submission candidate.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from predict import load_model, predict_probs
from src.data import CLASSES, GpuLoader, build_cache, find_data_dir, load_cache, make_split
from src.utils import setup_device


def report(name, pred, y):
    acc = (pred == y).mean()
    per_class = [(pred[y == c] == c).mean() for c in range(len(CLASSES))]
    print(f"\n{name}: {acc:.2%} ({(pred != y).sum()} wrong)")
    print("  " + " ".join(f"{CLASSES[c][:5]} {per_class[c]:.1%}" for c in np.argsort(per_class)))
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(confusion, (y, pred), 1)
    np.fill_diagonal(confusion, 0)
    top = np.argsort(confusion, axis=None)[::-1][:5]
    print("  top confusions: " + ", ".join(
        f"{CLASSES[t]}->{CLASSES[p]} {confusion[t, p]}" for t, p in zip(*np.unravel_index(top, confusion.shape))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpts", nargs="+", required=True, help="best.pt files from runs that kept a validation split")
    p.add_argument("--no-tta", action="store_true", help="skip the horizontal flip average")
    p.add_argument("--data-dir")
    args = p.parse_args()

    build_cache(find_data_dir(args.data_dir))
    data = load_cache()
    device, amp_dtype = setup_device()

    probs, split = {}, None
    for path in args.ckpts:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model, cfg = load_model(ckpt, device)
        if cfg.full_train:
            p.error(f"{path} trained on the validation images")
        this_split = make_split(data["train_y"], cfg.val_per_class, cfg.split_seed)["val_idx"]
        if split is not None and this_split != split:
            p.error(f"{path} uses different validation images")
        split = this_split
        val_idx = np.array(split)
        y = data["train_y"][val_idx]
        loader = GpuLoader(data["train_x"][val_idx], y, 500, device, ckpt["mean"], ckpt["std"],
                           channels_last=cfg.channels_last)
        probs[Path(path).parent.name] = predict_probs(model, loader, device, amp_dtype, tta=not args.no_tta)

    for name, prob in probs.items():
        report(name, prob.argmax(1), y)

    if len(probs) > 1:
        print("\npairs (a right & b wrong / a wrong & b right / predictions differ):")
        for a, b in combinations(probs, 2):
            pa, pb = probs[a].argmax(1), probs[b].argmax(1)
            print(f"  {a} vs {b}: {((pa == y) & (pb != y)).sum()} / {((pa != y) & (pb == y)).sum()} / {(pa != pb).sum()}")
        report("ensemble of all", np.mean(list(probs.values()), axis=0).argmax(1), y)


if __name__ == "__main__":
    main()
