# Experiment Log

One row per run. Change **one thing at a time** relative to the reference run, and note what changed.
Numbers come from `runs/<run>/summary.json`. `Public LB` stays blank unless the run was submitted.

Validation: stratified 45,000 / 5,000 split, `splits/split_seed2026_val500.json`. Every run uses the same split.

| # | Run folder | Change vs. reference | Params | Epochs | Best val acc (epoch) | Final train acc | Time (min) | GPU | Public LB | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 0a | baseline_1104-030203 | baseline (`configs/baseline.json`) | 2,797,610 | 100 | 94.18% (92) | 99.96% | 3.4 | A100 | 0.837 | TTA (hflip) |
| 0b | baseline_1104-043712 | baseline, repeat | 2,797,610 | 100 | 93.78% (89) | 99.97% | 3.3 | A100 | | |
| 0c | baseline_1105-215330 | baseline, repeat | 2,797,610 | 100 | 93.88% (88) | 99.98% | 3.2 | A100 | | |
| 0d | baseline_1105-221604 | baseline, repeat | 2,797,610 | 100 | 94.34% (99) | 99.97% | 3.2 | A100 | | |
| 1 | cutout16_1105-222201 | cutout 16 | 2,797,610 | 100 | 94.52% (99) | 97.45% | 14.4* | A100 | | |
| 2 | ls01_1105-222201 | label smoothing 0.1 | 2,797,610 | 100 | 93.52% (89) | 99.98% | 14.4* | A100 | | |
| 3 | wide40_1105-222200 | widths 40/80/160/320 | 4,368,690 | 100 | 94.10% (85) | 99.98% | 17.2* | A100 | | |
| 4 | ep200_1105-222200 | 200 epochs | 2,797,610 | 200 | 94.34% (196) | 100.00% | 20.3* | A100 | | |
| 5 | cj03_1105-222201 | color jitter 0.3 | 2,797,610 | 100 | 94.06% (89) | 99.94% | 18.7* | A100 | | |
| 6 | ta_1105-222200 | TrivialAugmentWide | 2,797,610 | 100 | 94.72% (96) | 91.06% | 20.1* | A100 | | |
| 7 | ta_ep200_1105-222201 | TrivialAugmentWide, 200 epochs | 2,797,610 | 200 | **95.58%** (195) | 92.79% | 24.3* | A100 | | |
| 8 | ema999_1105-222201 | EMA 0.999 | 2,797,610 | 100 | 93.88% (72) | 99.98% | 14.3* | A100 | | |

\* Runs 1–8 shared one A100 at the same time, so their times are not comparable to the baseline's.

## Observations
<!-- After each batch of runs: what helped, what didn't, and why you think so. This becomes the report's results section. -->

**Round 1 (runs 1–8, one change each vs. baseline):**
- TrivialAugmentWide is the clear winner: +0.7 pp at 100 epochs and +1.5 pp at 200 epochs (95.58%). Its train accuracy stays at 91–93%, so the model no longer memorizes the training set, and it keeps improving with longer training.
- Cutout 16 helps a little (+0.5 pp, borderline) and also lowers train accuracy (97.5%).
- No gain from changes that leave the model memorizing the training set (train ≈ 100%): 200 epochs alone (+0.3), a wider model alone (+0.05), color jitter (0.0), EMA (−0.2), label smoothing (−0.5). With cosine decay to zero, EMA adds nothing at the end of training.
- Takeaway: the baseline is limited by overfitting, not capacity. Capacity and training length only pay off once strong augmentation is in place, which sets up round 2.
- Color jitter does not help in-distribution val, but it could still help on the shifted evaluation set; only a leaderboard submission would show that.
- Four identical baseline runs: 94.18 / 93.78 / 93.88 / 94.34% (mean 94.05%, std 0.26 pp; GPU non-determinism). A single run needs about 94.5% or more to count as an improvement.
- Train accuracy reaches about 100% while val stays at about 94%, so the model overfits. Regularization and augmentation should help.
- Public LB (0.837) is about 10 pp below val. The 1,000-image public split has a standard error of about 1.2 pp, so this is not noise. The pipeline checks out (the test images are 32×32 RGB PNGs like the training images, and the submission format is validated), and the test images are slightly brighter and higher in contrast than the training images (per-channel means 0.513/0.498/0.462 vs 0.494/0.483/0.447; std about 0.27 vs 0.25). The evaluation set most likely comes from a shifted distribution. Gains in val accuracy usually carry over to a shifted test set, so val remains the selection metric, with extra weight on augmentation that improves robustness.

## Next experiments
Round 2: build on TrivialAugmentWide with 200 epochs (run 7) and change one thing each:
- 300 epochs
- widths 42/84/168/336 (4.82M params, closest to the limit)
- cutout 16 on top
- 3 stages, widths 64/128/256, blocks 3/3/3 (4.33M): wider early layers, an architecture variant for the report
