#!/bin/bash
# Submit from the repository root (main.ipynb does this for you):
#   sbatch slurm/train.sbatch --config configs/baseline.json
#   sbatch slurm/train.sbatch --config configs/baseline.json --set cutout=16 name=cutout16
#   sbatch slurm/train.sbatch --resume runs/<run>/last.pt
# Watch progress:  squeue -u $USER ;  tail -f runs/slurm-<jobid>.out
#SBATCH --job-name=cifar-resnet
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm-%j.out

set -euo pipefail

# Python with a CUDA build of torch. main.ipynb passes its own kernel's python as PY;
# otherwise use the python on PATH (sbatch exports the submitting shell's PATH), which on
# NYU HPC is /share/apps/pyenv/py3.9 with torch 2.6.0+cu126.
PY=${PY:-python}

cd "$SLURM_SUBMIT_DIR"
nvidia-smi || true
"$PY" -u train.py "$@"
