#!/bin/bash
# Stores the deep features of one pretrained network (scripts/extract_deep_features.py).
# The networks are only applied, not trained, so a CPU node is enough (about 15 s per subject, 10-15 min for all
# 20 training subjects). Submit from inside the repository, one job per network:
#
#     sbatch ubelix/extract_deep_features.sbatch resnet_imagenet
#     sbatch ubelix/extract_deep_features.sbatch radimagenet
#     sbatch ubelix/extract_deep_features.sbatch medicalnet
#     sbatch ubelix/extract_deep_features.sbatch radimagenet_3planes      # 2.5D variants: _25d and _3planes
#     sbatch ubelix/extract_deep_features.sbatch radimagenet --split test   # only before a final test
#
# On a GPU node instead (free accounts: H100 or RTX4090), add --device cuda and request the GPU, e.g.
#     sbatch --partition=gpu --gpus-per-node=h100:1 ubelix/extract_deep_features.sbatch radimagenet --device cuda
#
# Needs the environment <project>/envs/deepfeat (requirements-deep.txt) and the weights in <project>/weights/:
#     radimagenet_resnet50.pt              RadImageNet ResNet-50 (ResNet50.pt of the official PyTorch release)
#     medicalnet_resnet_50_23dataset.pth   MedicalNet ResNet-50 (TencentMedicalNet/MedicalNet-Resnet50)
# The ImageNet weights are downloaded by torchvision into <project>/weights/torch.
# (2 subjects on CPU used about 13 GB of memory.)

#SBATCH --job-name=deepfeat
#SBATCH --partition=epyc2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=../logs/%x_%j.out

set -euo pipefail

NETWORK="$1"; shift
REPO_DIR="$SLURM_SUBMIT_DIR"
PROJECT="${MIALAB_PROJECT:-$(dirname "$REPO_DIR")}"
export TORCH_HOME="$PROJECT/weights/torch"

case "$NETWORK" in
    resnet_imagenet*) WEIGHTS="" ;;   # also the 2.5D variants _25d and _3planes
    radimagenet*)     WEIGHTS="$PROJECT/weights/radimagenet_resnet50.pt" ;;
    medicalnet)       WEIGHTS="$PROJECT/weights/medicalnet_resnet_50_23dataset.pth" ;;
    *) echo "Unknown network $NETWORK"; exit 1 ;;
esac

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD), network $NETWORK"

"$PROJECT/envs/deepfeat/bin/python" -u scripts/extract_deep_features.py --network "$NETWORK" --device cpu \
    ${WEIGHTS:+--weights "$WEIGHTS"} "$@"
