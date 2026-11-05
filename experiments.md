# Experiment Log

One row per run. Change **one thing at a time** relative to the reference run, and note what changed.
Numbers come from `runs/<run>/summary.json`. `Public LB` stays blank unless the run was submitted.

Validation: stratified 45,000 / 5,000 split, `splits/split_seed2026_val500.json`. Every run uses the same split.

| # | Run folder | Change vs. reference | Params | Epochs | Best val acc (epoch) | Final train acc | Time (min) | GPU | Public LB | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 0a | baseline_1104-030203 | baseline (`configs/baseline.json`) | 2,797,610 | 100 | 94.18% (92) | 99.96% | 3.4 | A100 | 0.837 | TTA (hflip) |
| 0b | baseline_1104-043712 | baseline, repeat | 2,797,610 | 100 | 93.78% (89) | 99.97% | 3.3 | A100 | | |
| 0c | baseline_1105-215330 | baseline, repeat | 2,797,610 | 100 | 93.88% (88) | 99.98% | 3.2 | A100 | | |

## Observations
<!-- After each batch of runs: what helped, what didn't, and why you think so. This becomes the report's results section. -->
- Three identical baseline runs: 94.18 / 93.78 / 93.88% (mean 93.95%, std 0.21 pp; GPU non-determinism). Single-run differences below about 0.5 pp are noise.
- Train accuracy reaches about 100% while val stays at about 94%, so the model overfits. Regularization and augmentation should help.
- Public LB (0.837) is about 10 pp below val. The 1,000-image public split has a standard error of about 1.2 pp, so this is not noise. The pipeline checks out (the test images are 32×32 RGB PNGs like the training images, and the submission format is validated), and the test images are slightly brighter and higher in contrast than the training images (per-channel means 0.513/0.498/0.462 vs 0.494/0.483/0.447; std about 0.27 vs 0.25). The evaluation set most likely comes from a shifted distribution. Gains in val accuracy usually carry over to a shifted test set, so val remains the selection metric, with extra weight on augmentation that improves robustness.

## Next experiments
<!-- Decide from the baseline's train/val gap:
     - train ≈ 100% but val much lower → overfitting: cutout, stronger augmentation, label smoothing
     - both low → underfitting: wider/deeper model (within 5M), more epochs
     - val still rising at the end → more epochs -->
