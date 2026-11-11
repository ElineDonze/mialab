#!/bin/bash
# Sets up the MIA Lab project on UBELIX, the same way for every team member.
# Run it once on a login node, from inside the cloned repository:
#
#     bash ubelix/setup_ubelix.sh
#
# It is safe to run again: existing folders, the environment and the data are kept, and missing parts are added.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${MIALAB_PROJECT:-$(dirname "$REPO_DIR")}"
ENV_DIR="$PROJECT/envs/mialab"

echo "Repository: $REPO_DIR"
echo "Project:    $PROJECT"

# 1. Folders (the same layout for everybody, see mialab/utilities/paths.py)
mkdir -p "$PROJECT"/{data,cache,runs,outputs,logs,envs}

# 2. uv (manages Python and the packages, as on the laptops)
if ! command -v uv >/dev/null 2>&1; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "Installing uv into ~/.local/bin ..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi
echo "uv: $(uv --version)"

# 3. Python environment with exactly the versions of requirements.txt
if [ ! -x "$ENV_DIR/bin/python" ]; then
    uv venv "$ENV_DIR" --python 3.11
fi
uv pip install --python "$ENV_DIR/bin/python" -r "$REPO_DIR/requirements.txt" -e "$REPO_DIR"
"$ENV_DIR/bin/python" "$REPO_DIR/test/test_install.py" >/dev/null && echo "Environment OK: $ENV_DIR"

# 4. Dataset (copied once from the laptop; it is not downloaded by this script)
DATA="$PROJECT/data/MIALab_2020_Dataset"
if [ -d "$DATA/train" ] && [ -d "$DATA/test" ] && [ -d "$DATA/atlas" ]; then
    echo "Dataset OK: $(ls "$DATA/train" | wc -l) training and $(ls "$DATA/test" | wc -l) test subjects"
else
    echo
    echo "Dataset missing. Copy it once from your laptop (run this on the laptop, in the folder that"
    echo "contains MIALab_2020_Dataset):"
    echo "    scp -r MIALab_2020_Dataset $USER@submit01.hpc.unibe.ch:$PROJECT/data/"
fi

echo
echo "Done. Quick check of the whole pipeline (about 10 minutes):"
echo "    cd $REPO_DIR && sbatch ubelix/run_pipeline.sbatch --smoke"
