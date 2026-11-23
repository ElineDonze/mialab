#!/bin/bash
# Learning curve of the U-Net of one CV fold (scripts/unet_learning_curve.py): training vs held-out loss and Dice
# during training. Submit from inside the repository:
#
#     sbatch ubelix/unet_learning_curve.sbatch --fold 0 --out_dir <results folder>
#
# Short check (debug queue):
#     sbatch --qos=job_debug --time=00:20:00 ubelix/unet_learning_curve.sbatch --fold 0 --steps 20 \
#         --loss_every 10 --dice_every 20 --n_train_dice 1 --out_dir <scratch folder>
#
# The job may be preempted on the free GPUs; --requeue starts it again and the script continues from its last
# checkpoint. Needs the environment <project>/envs/deepfeat (requirements-deep.txt, with monai).

#SBATCH --job-name=unet-curve
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=../logs/%x_%j.out

set -euo pipefail

REPO_DIR="$SLURM_SUBMIT_DIR"
PROJECT="${MIALAB_PROJECT:-$(dirname "$REPO_DIR")}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD), restart ${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PROJECT/envs/deepfeat/bin/python" -u scripts/unet_learning_curve.py --device cuda "$@"
