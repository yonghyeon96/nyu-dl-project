"""Shared helpers: seeding, device/AMP setup and environment info."""
import os
import platform
import random
import sys
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def setup_device(use_amp=True):
    """Return (device, amp_dtype). amp_dtype is None on CPU or when AMP is off."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if use_amp:
            # L4 and A100 both support bf16, which needs no GradScaler.
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        # Use only the CPUs Slurm allocated; more threads than cores makes CPU runs crawl.
        n_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        torch.set_num_threads(max(1, min(4, n_cpus)))
    return device, amp_dtype


def autocast(device, amp_dtype):
    return torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else nullcontext()


def sync(device):
    if device.type == "cuda": torch.cuda.synchronize()


def _pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def env_info(device, amp_dtype):
    cuda = device.type == "cuda"
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if cuda else None,
        "torch_cuda_build": torch.version.cuda,
        "amp_dtype": str(amp_dtype) if amp_dtype else None,
        "versions": {p: _pkg_version(p) for p in ["torch", "numpy", "pillow"]},
    }
