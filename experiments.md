# Experiment Log

One row per run. Change **one thing at a time** relative to the reference run, and note what changed.
Numbers come from `runs/<run>/summary.json`. `Public LB` stays blank unless the run was submitted.

Validation: stratified 45,000 / 5,000 split, `splits/split_seed2026_val500.json`. Every run uses the same split.

| # | Run folder | Change vs. reference | Params | Epochs | Best val acc (epoch) | Final train acc | Time (min) | GPU | Public LB | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | | baseline (`configs/baseline.json`) | 2,797,610 | 100 | | | | | | |

## Observations
<!-- After each batch of runs: what helped, what didn't, and why you think so. This becomes the report's results section. -->

## Next experiments
<!-- Decide from the baseline's train/val gap:
     - train ≈ 100% but val much lower → overfitting: cutout, stronger augmentation, label smoothing
     - both low → underfitting: wider/deeper model (within 5M), more epochs
     - val still rising at the end → more epochs -->
