#!/bin/bash
# Trains the U-Net of one fold on a GPU and stores its features (E9, scripts/train_unet_features.py).
# Submit from inside the repository, one job per fold (6 models in total):
#
#     for f in 0 1 2 3 4 all; do sbatch ubelix/train_unet_features.sbatch --fold $f; done
#     sbatch ubelix/train_unet_features.sbatch --fold all --predict_test     # only before a final test
#
# Timing test (debug queue, max 20 minutes, nothing stored in the cache):
#     sbatch --qos=job_debug --time=00:20:00 ubelix/train_unet_features.sbatch --fold 0 --steps 200 --smoke \
#         --out_dir <scratch folder>
#
# The job may be preempted on the free GPUs; --requeue starts it again and the script continues from its last
# checkpoint. Needs the environment <project>/envs/deepfeat (requirements-deep.txt, with monai).

#SBATCH --job-name=unet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=../logs/%x_%j.out

set -euo pipefail

REPO_DIR="$SLURM_SUBMIT_DIR"
PROJECT="${MIALAB_PROJECT:-$(dirname "$REPO_DIR")}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD), restart ${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PROJECT/envs/deepfeat/bin/python" -u scripts/train_unet_features.py --device cuda "$@"
