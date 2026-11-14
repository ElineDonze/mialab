#!/bin/bash
# Stores the deep features of one pretrained network (scripts/extract_deep_features.py) on a GPU node.
# Submit from inside the repository, one job per network:
#
#     sbatch ubelix/extract_deep_features.sbatch resnet_imagenet
#     sbatch ubelix/extract_deep_features.sbatch radimagenet
#     sbatch ubelix/extract_deep_features.sbatch medicalnet
#     sbatch ubelix/extract_deep_features.sbatch radimagenet --split test   # only before a final test
#
# Needs the environment <project>/envs/deepfeat (requirements-deep.txt) and the weights in <project>/weights/:
#     radimagenet_resnet50.pt          RadImageNet ResNet-50 (PyTorch)
#     medicalnet_resnet_50_23dataset.pth   MedicalNet ResNet-50
# The ImageNet weights are downloaded by torchvision into <project>/weights/torch.

#SBATCH --job-name=deepfeat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=../logs/%x_%j.out

set -euo pipefail

NETWORK="$1"; shift
REPO_DIR="$SLURM_SUBMIT_DIR"
PROJECT="${MIALAB_PROJECT:-$(dirname "$REPO_DIR")}"
export TORCH_HOME="$PROJECT/weights/torch"

case "$NETWORK" in
    resnet_imagenet) WEIGHTS="" ;;
    radimagenet)     WEIGHTS="$PROJECT/weights/radimagenet_resnet50.pt" ;;
    medicalnet)      WEIGHTS="$PROJECT/weights/medicalnet_resnet_50_23dataset.pth" ;;
    *) echo "Unknown network $NETWORK"; exit 1 ;;
esac

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD), network $NETWORK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PROJECT/envs/deepfeat/bin/python" -u scripts/extract_deep_features.py --network "$NETWORK" \
    ${WEIGHTS:+--weights "$WEIGHTS"} "$@"
