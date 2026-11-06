"""Write submission.csv from a trained checkpoint and check its format against sample_submission.csv.

    python predict.py --ckpt runs/<run>/best.pt
    python predict.py --ckpt runs/<run>/final.pt --tta --out submission.csv
    python predict.py --ckpt runs/<run>/best.pt --compare-tta      # validation accuracy per TTA setting
    python predict.py --ckpt runs/<run>/best.pt --tta --shift 2    # flips plus shifts of up to 2 px
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.config import load_config
from src.data import CLASSES, GpuLoader, build_cache, find_data_dir, load_cache, make_split
from src.model import build_model, count_params
from src.utils import autocast, setup_device


def tta_views(x, tta, shift):
    """Each batch view to average: shifted by -shift, 0 and +shift px per axis, plus its flip with tta.

    Shifts pad with zeros, the channel mean after normalization, like the training crops.
    """
    H, W = x.shape[-2:]
    padded = F.pad(x, (shift, shift, shift, shift))
    steps = sorted({-shift, 0, shift})
    for dy in steps:
        for dx in steps:
            view = padded[..., shift + dy:shift + dy + H, shift + dx:shift + dx + W]
            yield view
            if tta:
                yield view.flip(3)


@torch.no_grad()
def predict(model, loader, device, amp_dtype, tta=False, shift=0):
    """Class index per image, in loader order, from softmax probabilities averaged over the TTA views."""
    model.eval()
    preds = []
    for x, _ in loader:
        with autocast(device, amp_dtype):
            probs = sum(model(view).float().softmax(1) for view in tta_views(x, tta, shift))
        preds.append(probs.argmax(1).cpu())
    return torch.cat(preds).numpy()


def compare_tta(model, data, cfg, ckpt, device, amp_dtype):
    """Print validation accuracy for each TTA setting, on the split this checkpoint did not train on."""
    val_idx = np.array(make_split(data["train_y"], cfg.val_per_class, cfg.seed)["val_idx"])
    val_y = data["train_y"][val_idx]
    loader = GpuLoader(data["train_x"][val_idx], val_y, 500, device, ckpt["mean"], ckpt["std"],
                       channels_last=cfg.channels_last)
    for tta, shift in ((False, 0), (True, 0), (True, 1), (True, 2), (True, 3), (True, 4)):
        labels = predict(model, loader, device, amp_dtype, tta=tta, shift=shift)
        print(f"flip={tta!s:<5} shift={shift}: val {(labels == val_y).mean():.2%}")


def write_submission(path, ids, labels):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        w.writerows(zip(ids.tolist(), labels.tolist()))


def validate_submission(path, sample_path):
    """Raise if the file does not match the sample's header, id order and label range."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    with open(sample_path, newline="") as f:
        sample = list(csv.reader(f))
    assert rows[0] == ["id", "label"], f"header is {rows[0]}"
    assert len(rows) == len(sample), f"{len(rows) - 1} rows, expected {len(sample) - 1}"
    assert [r[0] for r in rows[1:]] == [r[0] for r in sample[1:]], "ids differ from sample_submission.csv order"
    labels = [r[1] for r in rows[1:]]
    assert all(v.isdigit() and 0 <= int(v) <= 9 for v in labels), "labels must be integers 0-9"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="best.pt or final.pt from a run folder")
    p.add_argument("--out", help="output CSV (default: submission.csv next to the checkpoint)")
    p.add_argument("--tta", action="store_true", help="average predictions with horizontal flips")
    p.add_argument("--shift", type=int, default=0, help="also average over shifts of this many pixels")
    p.add_argument("--compare-tta", action="store_true", help="print validation accuracy per TTA setting and exit")
    p.add_argument("--data-dir")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = load_config(ckpt["config"])
    device, amp_dtype = setup_device(cfg.amp)

    data_dir = find_data_dir(args.data_dir)
    build_cache(data_dir)
    data = load_cache()

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compare_tta:
        if cfg.full_train:
            p.error("this checkpoint trained on the validation images, so it cannot be compared on them")
        compare_tta(model, data, cfg, ckpt, device, amp_dtype)
        return

    loader = GpuLoader(data["test_x"], None, 500, device, ckpt["mean"], ckpt["std"],
                       channels_last=cfg.channels_last)
    labels = predict(model, loader, device, amp_dtype, tta=args.tta, shift=args.shift)

    out = Path(args.out) if args.out else ckpt_path.parent / "submission.csv"
    write_submission(out, data["test_ids"], labels)
    validate_submission(out, data_dir / "sample_submission.csv")

    counts = Counter(labels.tolist())
    print(f"{cfg.name} ({count_params(model):,} params, epoch {ckpt['epoch']}, tta={args.tta}, shift={args.shift})")
    print("Predicted class counts:", {CLASSES[c]: counts.get(c, 0) for c in range(len(CLASSES))})
    print(f"Wrote {out} ({len(labels)} rows) - format check passed")


if __name__ == "__main__":
    main()
