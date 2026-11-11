"""Default locations of the data, cache, runs and outputs.

All paths are relative to the project folder, which is the folder that contains the repository. The same code
therefore works on every laptop and on every UBELIX account without editing paths:

    <project>/                     e.g. ~/MIA_Lab_Project on UBELIX
        code/ or mialab/           this repository
        data/MIALab_2020_Dataset/  the dataset (or <project>/MIALab_2020_Dataset/)
        cache/                     pre-processed images and features, computed once and reused
        runs/                      one small folder per run (configs, metrics); copied to OneDrive
        outputs/                   large files per run (segmentations); stay where they were computed

Every path can be changed with an environment variable, e.g. ``MIALAB_DATA=/some/where``.
"""
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.environ.get('MIALAB_PROJECT', os.path.dirname(REPO_DIR))


def _first_existing(candidates: list) -> str:
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


DATA_DIR = os.environ.get('MIALAB_DATA') or _first_existing([
    os.path.join(PROJECT_DIR, 'data', 'MIALab_2020_Dataset'),
    os.path.join(PROJECT_DIR, 'MIALab_2020_Dataset'),
])
CACHE_DIR = os.environ.get('MIALAB_CACHE', os.path.join(PROJECT_DIR, 'cache'))
RUNS_DIR = os.environ.get('MIALAB_RUNS', os.path.join(PROJECT_DIR, 'runs'))
OUTPUTS_DIR = os.environ.get('MIALAB_OUTPUTS', os.path.join(PROJECT_DIR, 'outputs'))

FOLDS_FILE = os.path.join(REPO_DIR, 'configs', 'folds.json')


def atlas_dir(data_dir: str = DATA_DIR) -> str:
    return os.path.join(data_dir, 'atlas')


def train_dir(data_dir: str = DATA_DIR) -> str:
    return os.path.join(data_dir, 'train')


def test_dir(data_dir: str = DATA_DIR) -> str:
    return os.path.join(data_dir, 'test')
