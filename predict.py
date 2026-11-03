"""Write submission.csv from a trained checkpoint and check its format against sample_submission.csv.

    python predict.py --ckpt runs/<run>/best.pt
    python predict.py --ckpt runs/<run>/final.pt --tta --out submission.csv
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.data import CLASSES, GpuLoader, build_cache, find_data_dir, load_cache
from src.model import build_model, count_params
from src.utils import autocast, setup_device


@torch.no_grad()
def predict(model, loader, device, amp_dtype, tta=False):
    """Class index per image, in loader order. With tta, average with the horizontally flipped image."""
    model.eval()
    preds = []
    for x, _ in loader:
        with autocast(device, amp_dtype):
            probs = model(x).float().softmax(1)
            if tta:
                probs += model(x.flip(3)).float().softmax(1)
        preds.append(probs.argmax(1).cpu())
    return torch.cat(preds).numpy()


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
    loader = GpuLoader(data["test_x"], None, 500, device, ckpt["mean"], ckpt["std"],
                       channels_last=cfg.channels_last)
    labels = predict(model, loader, device, amp_dtype, tta=args.tta)

    out = Path(args.out) if args.out else ckpt_path.parent / "submission.csv"
    write_submission(out, data["test_ids"], labels)
    validate_submission(out, data_dir / "sample_submission.csv")

    counts = Counter(labels.tolist())
    print(f"{cfg.name} ({count_params(model):,} params, epoch {ckpt['epoch']}, tta={args.tta})")
    print("Predicted class counts:", {CLASSES[c]: counts.get(c, 0) for c in range(len(CLASSES))})
    print(f"Wrote {out} ({len(labels)} rows) - format check passed")


if __name__ == "__main__":
    main()
