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
| 7 | ta_ep200_1105-222201 | TrivialAugmentWide, 200 epochs | 2,797,610 | 200 | **95.58%** (195) | 92.79% | 24.3* | A100 | 0.864 | TTA (hflip) |
| 8 | ema999_1105-222201 | EMA 0.999 | 2,797,610 | 100 | 93.88% (72) | 99.98% | 14.3* | A100 | | |
| 9 | ta_ep300_1106-052103 | run 7, 300 epochs | 2,797,610 | 300 | 96.04% (285) | 93.19% | 30.7* | A100 | | |
| 10 | ta_ep200_w42_1106-052103 | run 7, widths 42/84/168/336 | 4,815,940 | 200 | 96.02% (184) | 93.59% | 30.7* | A100 | | |
| 11 | ta_ep200_cut16_1106-052103 | run 7, cutout 16 | 2,797,610 | 200 | 95.36% (196) | 88.77% | 21.9* | A100 | | |
| 12 | ta_ep200_s3w64_1106-052103 | run 7, 3 stages, widths 64/128/256, blocks 3/3/3 | 4,327,754 | 200 | **96.52%** (191) | 94.50% | 31.3* | A100 | 0.872 | TTA (hflip) |

| 13 | s3w64_ep300_1106-072723 | run 12, 300 epochs | 4,327,754 | 300 | 97.00% (298) | 94.96% | 98.0* | A100 | 0.873 | TTA (hflip) |
| 14 | s3w68_1106-072723 | run 12, widths 68/136/272 | 4,884,926 | 200 | 96.64% (184) | 94.65% | 98.5* | A100 | | |
| 15 | s3w64_cutmix_1106-072723 | run 12, CutMix on 50% of batches | 4,327,754 | 200 | 97.00% (200) | 81.46% | 74.5* | A100 | 0.873 | TTA (hflip) |
| 16 | s3w64_mixup_1106-072723 | run 12, Mixup on 50% of batches | 4,327,754 | 200 | 96.38% (187) | 82.96% | 74.5* | A100 | | |
| 17 | s3w64_sam_1106-072723 | run 12, SAM rho 0.05 | 4,327,754 | 200 | 96.84% (182) | 94.50% | 101.6* | A100 | | Resumed at epoch 193 |
| 18 | s3w64_cj03_1106-072723 | run 12, color jitter 0.3 | 4,327,754 | 200 | 96.66% (184) | 94.30% | 81.1* | A100 | | |
| 19 | s3w68_ep300_1106-072723 | run 12, widths 68/136/272, 300 epochs | 4,884,926 | 300 | 96.80% (289) | 95.04% | 122.7* | A100 | | Resumed at epoch 203; two changes |
| 20 | s3w64_ep300_cutmix_1107-000424 | run 12, 300 epochs and CutMix | 4,327,754 | 300 | **97.10%** (291) | 81.34% | 38.8* | A100 | | Two changes |
| 21 | s3w64_ep300_cutmix_full_1107-000424 | run 20 on all 50,000 images | 4,327,754 | 300 | no val | 80.36% | 40.2* | A100 | 0.898 | final.pt; TTA (hflip + shift 1) |
| 22 | s3w64_ep300_cutmix_s2027_1107-010621 | run 20, seed 2027 | 4,327,754 | 300 | 96.86% (282) | 80.89% | 41.4* | A100 | | |
| 23 | s3w64_ep300_kd_1107-010621 | run 13, distillation from runs 13 and 15 | 4,327,754 | 300 | **97.48%** (269) | 94.61% | 51.3* | A100 | | Alpha 0.9, temperature 4 |
| 24 | s3w64_ep300_cutmix_full_s2027_1107-011808 | run 21, seed 2027 | 4,327,754 | 300 | no val | 81.79% | 42.0* | A100 | 0.896 | final.pt; TTA (hflip + shift 1) |
| 25 | s3w64_ep300_kd_full_1107-020711 | run 23 on all 50,000 images | 4,327,754 | 300 | no val | 94.48% | 23.5 | A100 | 0.875 | final.pt; no TTA |

\* Runs in the same round shared one A100 at the same time, so their times are not comparable to the baseline's.

## Observations
<!-- After each batch of runs: what helped, what didn't, and why you think so. This becomes the report's results section. -->

**Inference on run 20 (no retraining, 5,000 validation images):**
- Validation accuracy per TTA setting:

| Run | None | Flip | Shift 1 | Shift 2 | Shift 3 | Shift 4 |
|---|---|---|---|---|---|---|
| 20 | 97.10% | 97.30% | 97.48% | 97.40% | 97.32% | 97.36% |
| 22 | 96.86% | 97.14% | 97.06% | 97.08% | 97.04% | 96.96% |
| 23 | 97.48% | 97.32% | 97.36% | 97.34% | 97.36% | 97.24% |

- Flip helps the two CutMix runs by 0.20 to 0.28 pp. Shifts add no consistent gain beyond flip: +0.18, −0.08 and +0.04 pp. The distilled model scores best with no TTA.
- With flip TTA, runs 13, 15 and 20 score 97.02%, 97.36% and 97.30%. Their averaged predictions score 97.56%, or 10 more correct images than the best single model, so distillation has limited room.
- Cat and dog confusions are the largest group: 36 of the 135 errors of run 20 with flip TTA.

**Round 4 (runs 20–25):**
- Run 25 scored 0.875 public, about 2.2 pp below runs 21 and 24, even though its setting has the best validation accuracy. It has neither CutMix nor flip + shift 1 TTA, so the cause is open. Best validation accuracy did not pick the best public model.
- All public scores above 0.875 came from CutMix models trained on all images with flip + shift 1 TTA.
- Distillation gives the best validated model. Mean val of the last 10 epochs: run 23 97.42%, run 20 96.98%, run 22 96.79%, run 13 96.93%. That is +0.49 pp over run 13, its reference, from one run. Its validation loss is also the lowest (0.084).
- Repeating run 20 with another seed moved the last-10 mean by −0.19 pp, so the two seeds average 96.88%, close to runs 13 and 15.
- Run 21 scored 0.898 public, +2.5 pp over the previous best. It changed the training data and the TTA together, so this submission cannot separate their effects.
- Run 24 repeats run 21 with another seed and scored 0.896 public, so training on all 50,000 images gave about 0.897 across two seeds. Their predictions differ on 113 of 2,000 images.

**Round 3 (runs 13–19, changes vs. run 12):**
- Comparing the mean val of the last 10 epochs, which is less optimistic than best val: run 20 96.98%, run 13 96.93%, run 15 96.89%, run 19 96.74%, run 17 96.64%, run 18 96.58%, run 14 96.49%, run 12 96.44%, run 16 96.31%.
- 300 epochs (+0.49 pp) and CutMix (+0.45 pp) are the useful changes. Combining them adds little (+0.54 pp), and the top three runs are within 0.09 pp, so this recipe levels off near 97%.
- Wider models, SAM, color jitter and Mixup show no clear gain. SAM costs twice as much per step.
- Runs 13 and 15 scored 0.873 public, +0.1 pp over run 12, which is one image and within noise.

**Round 2 (runs 9–12, one change each vs. run 7):**
- The 3-stage model is the clear winner: 96.52%, +0.94 pp. It beats the 4-stage model with more parameters (run 10, 96.02%). It changed the stages, widths and blocks together, so the gain cannot be traced to one of them.
- 300 epochs (+0.46 pp) and widths 42/84/168/336 (+0.44 pp) are about the size of the difference between two repeated runs (0.40 pp), so these gains are not confirmed.
- Cutout on top of TrivialAugmentWide does not help (−0.22 pp): train accuracy falls to 88.8%, so regularization is already sufficient.
- Leaderboard: run 12 scored 0.872 public, +0.8 pp over run 7.

**Round 1 (runs 1–8, one change each vs. baseline):**
- TrivialAugmentWide is the clear winner: +0.7 pp at 100 epochs and +1.5 pp at 200 epochs (95.58%). Its train accuracy stays at 91–93%, so the model no longer memorizes the training set, and it keeps improving with longer training.
- Cutout 16 helps a little (+0.5 pp, borderline) and also lowers train accuracy (97.5%).
- No gain from changes that leave the model memorizing the training set (train ≈ 100%): 200 epochs alone (+0.3), a wider model alone (+0.05), color jitter (0.0), EMA (−0.2), label smoothing (−0.5). In this one baseline run, EMA gave no gain at the end of a cosine schedule.
- Takeaway: the baseline is limited by overfitting, not capacity. Capacity and training length only pay off once strong augmentation is in place, which sets up round 2.
- Leaderboard: run 7 scored 0.864 public vs 0.837 for the baseline. Validation accuracy stays the selection metric.
- Four identical baseline runs: 94.18 / 93.78 / 93.88 / 94.34% (mean 94.05%, std 0.26 pp; GPU non-determinism). A single run needs about 94.5% or more to count as an improvement.
- Train accuracy reaches about 100% while val stays at about 94%, so the model overfits. Regularization and augmentation should help.
- Public LB (0.837) is about 10 pp below val. The submission format is validated. The cause of the gap is not known, so validation accuracy remains the selection metric.

## Final model

Run 21 is the final model: the setting in `configs/final.json`, `final.pt`, flip + shift 1 TTA, 4,327,754 parameters, public 0.898. Run 24 repeats it with another seed (0.896).

The final choice uses the official public scores as well as validation accuracy, because the model with the best validation accuracy (run 23) did not score best in its full-data version (run 25, 0.875).

Open question: runs 21 and 24 changed the training data and the TTA together, and run 25 used neither CutMix nor TTA, so the source of the public gain is not separated.
