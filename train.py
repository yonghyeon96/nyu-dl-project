"""Train a ResNet from a config file.

    python train.py --config configs/baseline.json
    python train.py --config configs/baseline.json --set epochs=200 cutout=16 name=cutout16
    python train.py --config configs/baseline.json --smoke        # quick CPU check: 2k images, 2 epochs
    python train.py --resume runs/<run>/last.pt                   # continue an interrupted run

Each run writes runs/<name>_<time>/ with config.json, env.json, log.csv, last.pt,
best.pt (best validation accuracy), final.pt and summary.json.
"""
import argparse
import copy
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.config import load_config
from src.data import ROOT, GpuLoader, build_cache, find_data_dir, load_cache, make_split
from src.model import build_model, count_params
from src.utils import autocast, env_info, seed_all, setup_device, sync

LOG_FIELDS = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_sec"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON config, e.g. configs/baseline.json")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    p.add_argument("--resume", help="path to a last.pt checkpoint")
    p.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline end to end")
    p.add_argument("--data-dir", help="folder with train_labels.csv (auto-detected if omitted)")
    p.add_argument("--out-dir", default=str(ROOT / "runs"))
    args = p.parse_args()
    if not (args.config or args.resume):
        p.error("give --config or --resume")
    return args


def cosine_with_warmup(total_steps, warmup_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    return factor


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    loss_sum = torch.zeros((), device=device); correct = torch.zeros((), device=device); n = 0
    for x, y in loader:
        with autocast(device, amp_dtype):
            logits = model(x)
        loss_sum += F.cross_entropy(logits.float(), y, reduction="sum")
        correct += (logits.argmax(1) == y).sum()
        n += len(y)
    return loss_sum.item() / n, correct.item() / n


class ModelEMA:
    """Exponential moving average of the weights and BatchNorm statistics, updated every step.

    The decay ramps up from 0 so the average is not dominated by the random initial weights.
    """

    def __init__(self, model, decay):
        self.decay, self.updates = decay, 0
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        # state_dict() tensors share storage with the modules, so these lists stay live.
        self._ema = [t for t in self.model.state_dict().values() if t.is_floating_point()]
        self._src = [t for t in model.state_dict().values() if t.is_floating_point()]

    @torch.no_grad()
    def update(self):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        torch._foreach_lerp_(self._ema, self._src, 1 - decay)

    def state_dict(self):
        return {"model": self.model.state_dict(), "updates": self.updates}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.updates = state["updates"]


def save_atomic(obj, path):
    # Write to a temp file first so an instance shutdown never leaves a half-written checkpoint.
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    args = parse_args()

    # ---- config and run folder
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = load_config(ckpt["config"])
        smoke = ckpt["smoke"]
        run_dir = Path(args.resume).resolve().parent
    else:
        cfg = load_config(args.config, args.set)
        smoke = args.smoke
        if smoke:
            cfg.epochs = 2
        run_dir = Path(args.out_dir) / f"{cfg.name}{'_smoke' if smoke else ''}_{time.strftime('%m%d-%H%M%S')}"
        run_dir.mkdir(parents=True)

    seed_all(cfg.seed)
    device, amp_dtype = setup_device(cfg.amp)

    # ---- data
    build_cache(find_data_dir(args.data_dir))
    data = load_cache()
    mean, std = data["meta"]["mean"], data["meta"]["std"]
    split = make_split(data["train_y"], cfg.val_per_class, cfg.seed)
    if cfg.full_train:
        train_idx, val_idx = np.arange(len(data["train_y"])), np.array([], dtype=np.int64)
    else:
        train_idx, val_idx = np.array(split["train_idx"]), np.array(split["val_idx"])
    if smoke:
        rng = np.random.default_rng(0)
        train_idx = rng.choice(train_idx, 2000, replace=False)
        val_idx = rng.choice(val_idx, min(500, len(val_idx)), replace=False)

    train_loader = GpuLoader(data["train_x"][train_idx], data["train_y"][train_idx], cfg.batch_size, device,
                             mean, std, train=True, crop_padding=cfg.crop_padding, hflip=cfg.hflip,
                             cutout=cfg.cutout, color_jitter=cfg.color_jitter, trivial_augment=cfg.trivial_augment,
                             channels_last=cfg.channels_last, seed=cfg.seed)
    val_loader = None
    if len(val_idx):
        val_loader = GpuLoader(data["train_x"][val_idx], data["train_y"][val_idx], 1000, device,
                               mean, std, channels_last=cfg.channels_last)

    # ---- model and optimizer
    model = build_model(cfg).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    n_params = count_params(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                nesterov=cfg.nesterov, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_with_warmup(cfg.epochs * steps_per_epoch, cfg.warmup_epochs * steps_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema_decay else None
    eval_model = ema.model if ema else model     # validated, and saved to best.pt / final.pt

    start_epoch, best_acc, best_epoch = 0, -1.0, 0
    if ckpt:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        train_loader.gen.set_state(ckpt["loader_rng"])
        if ema:
            ema.load_state_dict(ckpt["ema"])
        start_epoch, best_acc, best_epoch = ckpt["epoch"], ckpt["best_acc"], ckpt["best_epoch"]
        print(f"Resumed {run_dir.name} at epoch {start_epoch}")
    else:
        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        (run_dir / "env.json").write_text(json.dumps(
            {**env_info(device, amp_dtype), "n_params": n_params, "smoke": smoke,
             "n_train": len(train_idx), "n_val": len(val_idx),
             "split_file": f"splits/split_seed{cfg.seed}_val{cfg.val_per_class}.json"}, indent=2))

    log_path = run_dir / "log.csv"
    if ckpt and log_path.is_file():
        # Drop log rows written after the checkpoint we resume from.
        with open(log_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) <= start_epoch]
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, LOG_FIELDS); w.writeheader(); w.writerows(rows)

    print(f"{cfg.name}: {n_params:,} params | train {len(train_idx)} / val {len(val_idx)} | "
          f"{cfg.epochs} epochs on {device} (amp={amp_dtype}){' | val on EMA weights' if ema else ''} -> {run_dir}")

    # ---- training loop
    t_start = time.perf_counter()
    val_acc = None      # stays None when resuming a run that already finished
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=device); correct = torch.zeros((), device=device); seen = 0
        for x, y in train_loader:
            with autocast(device, amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema:
                ema.update()
            loss_sum += loss.detach().float() * len(y)
            correct += (logits.argmax(1) == y).sum()
            seen += len(y)

        val_loss, val_acc = evaluate(eval_model, val_loader, device, amp_dtype) if val_loader else (None, None)
        sync(device)
        row = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": loss_sum.item() / seen, "train_acc": correct.item() / seen,
               "val_loss": val_loss, "val_acc": val_acc, "epoch_sec": time.perf_counter() - t0}

        improved = val_acc is not None and val_acc > best_acc
        if improved:
            best_acc, best_epoch = val_acc, epoch + 1
            save_atomic({"config": asdict(cfg), "model": eval_model.state_dict(), "epoch": epoch + 1,
                         "val_acc": val_acc, "mean": mean, "std": std}, run_dir / "best.pt")
        save_atomic({"config": asdict(cfg), "smoke": smoke, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "scaler": scaler.state_dict(), "loader_rng": train_loader.gen.get_state(),
                     "ema": ema.state_dict() if ema else None,
                     "epoch": epoch + 1, "best_acc": best_acc, "best_epoch": best_epoch,
                     "mean": mean, "std": std}, run_dir / "last.pt")

        new_log = not log_path.is_file()
        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, LOG_FIELDS)
            if new_log: w.writeheader()
            w.writerow(row)
        val_txt = f"val {val_loss:.4f} / {val_acc:.2%}{' *' if improved else ''}" if val_acc is not None else "no val"
        print(f"[{epoch + 1:>3}/{cfg.epochs}] lr {row['lr']:.4f} | train {row['train_loss']:.4f} / "
              f"{row['train_acc']:.2%} | {val_txt} | {row['epoch_sec']:.1f}s", flush=True)

    # ---- wrap up
    save_atomic({"config": asdict(cfg), "model": eval_model.state_dict(), "epoch": cfg.epochs,
                 "val_acc": val_acc if val_loader else None, "mean": mean, "std": std}, run_dir / "final.pt")
    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))
    summary = {
        "name": cfg.name, "run_dir": run_dir.name, "n_params": n_params, "epochs": cfg.epochs,
        "best_val_acc": best_acc if val_loader else None, "best_epoch": best_epoch if val_loader else None,
        "final_val_acc": val_acc if val_loader else None,
        "final_train_acc": float(rows[-1]["train_acc"]),
        "train_minutes": sum(float(r["epoch_sec"]) for r in rows) / 60,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Done in {(time.perf_counter() - t_start) / 60:.1f} min this session.")


if __name__ == "__main__":
    main()
