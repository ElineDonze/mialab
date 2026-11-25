#!/bin/bash
# Trains the 3D autoencoder (E7) on the training subjects and stores its features
# (scripts/train_autoencoder.py). Submit from inside the repository:
#
#     sbatch ubelix/train_autoencoder.sbatch
#     sbatch ubelix/train_autoencoder.sbatch --split test     # only before a final test
#
# Quick test on 2 subjects (separate cache, few iterations):
#     sbatch --time=00:30:00 ubelix/train_autoencoder.sbatch --limit 2 --iterations 50 --log_every 10 \
#         --cache_dir <scratch folder>
#
# On a GPU node instead (free accounts: H100 or RTX4090), add --device cuda and request the GPU, e.g.
#     sbatch --partition=gpu --gpus-per-node=h100:1 ubelix/train_autoencoder.sbatch --device cuda
#
# Needs the environment <project>/envs/deepfeat (requirements-deep.txt).

#SBATCH --job-name=autoencoder
#SBATCH --partition=epyc2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=../logs/%x_%j.out

set -euo pipefail

REPO_DIR="$SLURM_SUBMIT_DIR"
PROJECT="${MIALAB_PROJECT:-$(dirname "$REPO_DIR")}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD)"

"$PROJECT/envs/deepfeat/bin/python" -u scripts/train_autoencoder.py --device cpu "$@"
