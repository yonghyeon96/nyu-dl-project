# Efficient ResNet on CIFAR-10

A ResNet with **at most 5M trainable parameters**, trained from scratch on the CIFAR-10 training images provided for the course Kaggle competition.

## Repository layout

```
main.ipynb          control panel: checks, training jobs, results, submission
train.py            training entry point (config-driven, checkpoint/resume)
predict.py          writes and validates submission.csv
src/
  config.py         experiment settings (JSON + command-line overrides)
  data.py           PNG -> .npy cache, fixed stratified split, GPU loader with augmentation
  augment.py        batched GPU augmentations (color jitter, TrivialAugmentWide)
  model.py          configurable ResNet + 5M parameter check
  utils.py          seeding, device/AMP setup, environment info
configs/            experiment configs (baseline.json)
slurm/train.sbatch  Slurm GPU job for NYU HPC
splits/             fixed train/validation split (45,000 / 5,000, stratified)
experiments.md      experiment log
```

Not in the repository: competition data (`data/`), the generated cache (`cache/`), and run outputs (`runs/`).

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
```
On NYU HPC, run `sbatch slurm/train.sbatch --config configs/baseline.json`, or use `main.ipynb`.

Each run folder `runs/<name>_<time>/` holds the following files:
- `config.json`: the full configuration
- `env.json`: hardware, package versions and parameter count
- `log.csv`: per-epoch metrics
- `best.pt`, `last.pt` and `final.pt`: checkpoints
- `summary.json`: the run summary

## Rules compliance

- **Parameter limit:** `build_model` refuses any model over 5,000,000 trainable parameters.
- **Training from scratch:** weights are randomly initialized. No pretrained weights or external data are used.
- **Evaluation images:** they are used only to produce predictions. Normalization statistics come from the training images.
- **Reproducibility:** the seed is fixed, the validation split is committed, and each run records its config and environment.

## Results

See `experiments.md`.
