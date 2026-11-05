"""Batched pixel-space augmentations on the GPU.

Images are float tensors in [0, 1] with shape N,3,H,W. Every image gets its own random
parameters, and nothing loops over images, so a whole batch costs a few kernel launches.
"""
import math

import torch
from torch.nn import functional as F


def _rand(n, gen, device):
    return torch.rand(n, device=device, generator=gen)


def _gray(x):
    w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True)


def _blend(x, other, factor):
    """factor * x + (1 - factor) * other, per image: factor 0 gives `other`, 1 leaves x unchanged."""
    return (other + factor.view(-1, 1, 1, 1) * (x - other)).clamp(0, 1)


def brightness(x, factor):
    return _blend(x, torch.zeros_like(x), factor)


def contrast(x, factor):
    return _blend(x, _gray(x).mean((1, 2, 3), keepdim=True), factor)


def saturation(x, factor):
    return _blend(x, _gray(x), factor)


def sharpness(x, factor):
    kernel = x.new_tensor([[1, 1, 1], [1, 5, 1], [1, 1, 1]]) / 13
    smooth = x.clone()
    smooth[..., 1:-1, 1:-1] = F.conv2d(x, kernel.expand(3, 1, 3, 3), groups=3)   # border pixels stay as is
    return _blend(x, smooth, factor)


def posterize(x, bits):
    keep = (256 - 2 ** (8 - bits)).to(torch.uint8).view(-1, 1, 1, 1)   # mask of the top `bits` bits
    return ((x * 255).round().to(torch.uint8) & keep).float() / 255


def solarize(x, threshold):
    return torch.where(x >= threshold.view(-1, 1, 1, 1), 1 - x, x)


def autocontrast(x):
    lo, hi = x.amin((2, 3), keepdim=True), x.amax((2, 3), keepdim=True)
    flat = hi <= lo
    scale = torch.where(flat, torch.ones_like(hi), 1 / (hi - lo).clamp(min=1e-8))
    return ((x - torch.where(flat, torch.zeros_like(lo), lo)) * scale).clamp(0, 1)


def equalize(x):
    """Per-channel histogram equalization, same rule as PIL.ImageOps.equalize."""
    B, C, H, W = x.shape
    v = (x * 255).round().long().reshape(B * C, H * W)
    hist = torch.zeros(B * C, 256, device=x.device).scatter_add_(1, v, torch.ones_like(v, dtype=torch.float))
    step = (hist.sum(1, keepdim=True) - hist.gather(1, v.amax(1, keepdim=True))).div(255, rounding_mode="floor")
    lut = (hist.cumsum(1) + step.div(2, rounding_mode="floor")).div(step.clamp(min=1), rounding_mode="floor")
    lut = F.pad(lut, (1, 0))[:, :-1].clamp(0, 255)
    out = torch.where(step > 0, lut.gather(1, v), v.float())
    return (out / 255).view(B, C, H, W)


def affine(x, theta):
    """Warp each image by its own 2x3 matrix (grid units); uncovered pixels become black."""
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="nearest", padding_mode="zeros", align_corners=False)


def color_jitter(x, strength, gen):
    """Brightness, contrast and saturation, each scaled by a random factor in [1 - strength, 1 + strength]."""
    for op in (brightness, contrast, saturation):
        x = op(x, 1 + strength * (2 * _rand(len(x), gen, x.device) - 1))
    return x


TRIVIAL_OPS = ("identity", "shear_x", "shear_y", "translate_x", "translate_y", "rotate",
               "brightness", "color", "contrast", "sharpness", "posterize", "solarize",
               "autocontrast", "equalize")


def trivial_augment(x, gen):
    """TrivialAugmentWide (Muller & Hutter, 2021): one random op at a random strength per image.

    Op set and strength ranges follow torchvision's TrivialAugmentWide (31 strength bins).
    """
    B, dev = len(x), x.device
    op = torch.randint(0, len(TRIVIAL_OPS), (B,), device=dev, generator=gen)
    m = torch.randint(0, 31, (B,), device=dev, generator=gen).float() / 30        # strength in [0, 1]
    sign = torch.where(_rand(B, gen, dev) < 0.5, -1.0, 1.0)
    is_op = {name: op == i for i, name in enumerate(TRIVIAL_OPS)}

    # Geometric ops: one affine warp per image (identity matrix for every other op).
    shear = sign * m * 0.99
    shift = sign * m * 32 * 2 / x.shape[-1]          # up to 32 px, converted to grid units
    angle = sign * m * math.radians(135)
    rot, cos, sin = is_op["rotate"], angle.cos(), angle.sin()
    zero, one = torch.zeros_like(m), torch.ones_like(m)
    theta = torch.stack([
        torch.stack([torch.where(rot, cos, one),
                     torch.where(rot, -sin, torch.where(is_op["shear_x"], shear, zero)),
                     torch.where(is_op["translate_x"], shift, zero)], 1),
        torch.stack([torch.where(rot, sin, torch.where(is_op["shear_y"], shear, zero)),
                     torch.where(rot, cos, one),
                     torch.where(is_op["translate_y"], shift, zero)], 1),
    ], 1)
    geometric = sum(is_op[k] for k in ("shear_x", "shear_y", "translate_x", "translate_y", "rotate")).bool()
    out = torch.where(geometric.view(-1, 1, 1, 1), affine(x, theta), x)

    # Color ops: compute each on the whole batch, keep it where it was drawn.
    factor = 1 + sign * m * 0.99
    color_ops = [
        ("brightness", brightness(x, factor)),
        ("color", saturation(x, factor)),
        ("contrast", contrast(x, factor)),
        ("sharpness", sharpness(x, factor)),
        ("posterize", posterize(x, 8 - (m * 6).round().long())),
        ("solarize", solarize(x, 1 - m)),
        ("autocontrast", autocontrast(x)),
        ("equalize", equalize(x)),
    ]
    for name, y in color_ops:
        out = torch.where(is_op[name].view(-1, 1, 1, 1), y, out)
    return out
