#!/bin/bash
# Runs pipeline.py (or another script of the repository) as a Slurm job on UBELIX.
# Submit it from inside the repository; all arguments are passed on:
#
#     sbatch ubelix/run_pipeline.sbatch --config configs/experiments/E0_baseline.toml
#     sbatch ubelix/run_pipeline.sbatch --smoke
#     sbatch --export=ALL,SCRIPT=analysis/correlation_map.py ubelix/run_pipeline.sbatch --config <config>
#
# The job log is written to <project>/logs/<job name>_<job id>.out.
# Change resources on the command line if needed, e.g. sbatch --time=12:00:00 --mem=32G ...
# (E0 with 7 features used about 1.3 GB of memory and 4 minutes.)

#SBATCH --job-name=mialab
#SBATCH --partition=epyc2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=../logs/%x_%j.out

set -euo pipefail

REPO_DIR="$SLURM_SUBMIT_DIR"
PROJECT="${MIALAB_PROJECT:-$(dirname "$REPO_DIR")}"
SCRIPT="${SCRIPT:-pipeline.py}"

cd "$REPO_DIR"
echo "Job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, commit $(git rev-parse --short HEAD)"
echo "Command: $SCRIPT $*"

"$PROJECT/envs/mialab/bin/python" -u "$SCRIPT" "$@"
