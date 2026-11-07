# Efficient ResNet on CIFAR-10

A ResNet with **at most 5M trainable parameters**, trained from scratch on the CIFAR-10 training images provided for the course Kaggle competition.

## Repository layout

```
main.ipynb          control panel: checks, training jobs, results, submission
train.py            training entry point (config-driven, checkpoint/resume)
predict.py          writes and validates submission.csv
analyze.py          per-class, pairwise and ensemble results on the validation images
src/
  config.py         experiment settings (JSON + command-line overrides)
  data.py           PNG -> .npy cache, fixed stratified split, GPU loader with augmentation
  augment.py        batched GPU augmentations (color jitter, TrivialAugmentWide, CutMix, Mixup)
  model.py          configurable ResNet + 5M parameter check
  utils.py          seeding, device/AMP setup, environment info
configs/            baseline.json and final.json (the submitted model)
slurm/train.sbatch  Slurm GPU job for NYU HPC
splits/             fixed train/validation split (45,000 / 5,000, stratified)
experiments.md      experiment log
results/            per-run configs, logs and summaries, plus result tables
```

Not in the repository: competition data (`data/`), the generated cache (`cache/`), checkpoints and prediction files.

## Setup

```bash
pip install -r requirements.txt
```
Place the competition files in `data/`: `train_images/`, `test_images/`, `train_labels.csv`, `test.csv` and `sample_submission.csv`. To use another folder, set `DATA_DIR`.

## Usage

```bash
python -m src.model --config configs/baseline.json                   # parameter count
python train.py --config configs/baseline.json                       # train
python train.py --config configs/baseline.json --set cutout=16 name=cutout16
python train.py --resume runs/<run>/last.pt                          # resume
python predict.py --ckpt runs/<run>/best.pt --tta                    # submission.csv
python predict.py --ckpt runs/<run>/best.pt --compare-tta            # validation accuracy per TTA setting
python analyze.py --ckpts runs/<a>/best.pt runs/<b>/best.pt          # per-class and ensemble validation results
```
On NYU HPC, run `sbatch slurm/train.sbatch --config configs/baseline.json`, or use `main.ipynb`.

### Reproduce the final submission

```bash
python -m src.model --config configs/final.json      # 4,327,754 trainable parameters
python train.py --config configs/final.json          # writes runs/final_<time>/final.pt
python predict.py --ckpt runs/final_<time>/final.pt --tta --shift 1
```

This trains on all 50,000 images for 300 epochs and averages predictions over 18 flipped and shifted views. GPU training is not fully deterministic, so a new run gives close but not identical predictions. With seed 2026 and seed 2027, this setting scored 0.898 and 0.896 on the public leaderboard.

Each run folder `runs/<name>_<time>/` holds the following files:
- `config.json`: the full configuration
- `env.json`: hardware, package versions and parameter count
- `log.csv`: per-epoch metrics
- `best.pt`, `last.pt` and `final.pt`: checkpoints
- `summary.json`: the run summary

## Rules compliance

- **Parameter limit:** `build_model` refuses any model over 5,000,000 trainable parameters.
- **Training from scratch:** weights are randomly initialized. No pretrained weights or external data are used.
- **Evaluation images:** the training and prediction code uses them only to produce predictions. Normalization statistics come from the 50,000 labeled training images.
- **Distillation:** teacher models are trained from scratch on the allowed training data only. The submitted model is the single student, under the parameter limit.
- **Reproducibility:** seeds are fixed, the validation split is committed, and each run records its config and environment. GPU training is not fully deterministic: repeated baseline runs differ by about 0.3 percentage points.

## Results

**Final model:** 3 stages, widths 64/128/256, blocks 3/3/3, **4,327,754 trainable parameters**. Trained with SGD (Nesterov), cosine learning rate, TrivialAugmentWide, CutMix on half of the batches and 300 epochs on all 50,000 images. Configuration: [configs/final.json](configs/final.json).

| Model | Parameters | Training images | TTA | Validation accuracy | Public score |
|---|---|---|---|---|---|
| Baseline, 4 stages | 2,797,610 | 45,000 | Flip | 94.18% | 0.837 |
| TrivialAugmentWide, 200 epochs | 2,797,610 | 45,000 | Flip | 95.58% | 0.864 |
| 3 stages, 200 epochs | 4,327,754 | 45,000 | Flip | 96.52% | 0.872 |
| 3 stages, 300 epochs | 4,327,754 | 45,000 | Flip | 97.00% | 0.873 |
| 3 stages, CutMix | 4,327,754 | 45,000 | Flip | 97.00% | 0.873 |
| **3 stages, 300 epochs, CutMix (final)** | **4,327,754** | **50,000** | **Flip + shift 1** | **None held out** | **0.898** |
| Same, seed 2027 | 4,327,754 | 50,000 | Flip + shift 1 | None held out | 0.896 |
| Distillation, 300 epochs | 4,327,754 | 50,000 | None | None held out | 0.875 |

The validation accuracies are without TTA. The setting with distillation reached 97.48% with 45,000 training images. Full history: [experiments.md](experiments.md). Per-run records: [results/](results/).

## References

- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
- DeVries and Taylor, "Improved Regularization of Convolutional Neural Networks with Cutout", 2017.
- Muller and Hutter, "TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation", ICCV 2021. Operations and strength ranges follow torchvision's `TrivialAugmentWide`.
- Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018.
- Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features", ICCV 2019.
- Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR 2021.
- Hinton et al., "Distilling the Knowledge in a Neural Network", 2015.
