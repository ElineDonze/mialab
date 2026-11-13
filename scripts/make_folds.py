"""Creates configs/folds.json: the fixed 5-fold split of the training subjects used by every run.

Run this only once; the file is committed so that everybody uses exactly the same folds. Running it again with the
same seed and subjects gives the same file.

    python scripts/make_folds.py
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mialab.utilities.paths as paths  # noqa: E402


def main(n_folds: int, seed: int):
    subjects = sorted(d for d in os.listdir(paths.train_dir()) if os.path.isdir(os.path.join(paths.train_dir(), d)))
    order = np.random.default_rng(seed).permutation(len(subjects))
    folds = [sorted(subjects[i] for i in order[k::n_folds]) for k in range(n_folds)]
    content = {'description': 'Fixed subject-wise cross-validation folds of the training subjects. '
                              'Each list holds the validation subjects of one fold. Created by scripts/make_folds.py.',
               'seed': seed, 'n_folds': n_folds, 'folds': folds}
    with open(paths.FOLDS_FILE, 'w') as f:
        json.dump(content, f, indent=4)
        f.write('\n')
    print('Wrote', paths.FOLDS_FILE, [len(fold) for fold in folds])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create the fixed cross-validation folds')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.n_folds, args.seed)
