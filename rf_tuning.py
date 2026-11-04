"""Random forest hyperparameter tuning with subject-wise cross-validation.

The training subjects are pre-processed once, then every random forest setting in the grid file is evaluated
with cross-validation on the training subjects only. The test subjects are never used, so the selected settings
can afterwards be evaluated on the test set with pipeline.py without biasing the result.
"""
import argparse
import copy
import csv
import datetime
import itertools
import os
import sys
import timeit
import tomllib

import numpy as np
import sklearn.ensemble as sk_ensemble
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection

try:
    import mialab.utilities.config as cfg
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.utilities.config as cfg
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil

from pipeline import LOADING_KEYS

LABELS = {1: 'WhiteMatter', 2: 'GreyMatter', 3: 'Hippocampus', 4: 'Amygdala', 5: 'Thalamus'}


def load_training_data(config: dict, data_atlas_dir: str, data_train_dir: str, cache_file: str):
    """Pre-processes the training subjects and returns the sampled voxels, or loads them from the cache file.

    Returns:
        tuple: Features (n_voxels, n_features), labels (n_voxels,) and subject index per voxel (n_voxels,).
    """
    if cache_file and os.path.exists(cache_file):
        print('-' * 5, 'Loading training data from', cache_file)
        data = np.load(cache_file)
        return data['features'], data['labels'], data['groups']

    print('-' * 5, 'Pre-processing training subjects...')
    np.random.seed(config['seed'])
    putil.load_atlas_images(data_atlas_dir)
    crawler = futil.FileSystemDataCrawler(data_train_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())
    images = putil.pre_process_batch(crawler.data, cfg.pre_process_params(config), multi_process=False)

    features = np.concatenate([img.feature_matrix[0] for img in images])
    labels = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()
    groups = np.concatenate([np.full(len(img.feature_matrix[1]), i) for i, img in enumerate(images)])

    if cache_file:
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        np.savez_compressed(cache_file, features=features, labels=labels, groups=groups)
        print('-' * 5, 'Saved training data to', cache_file)

    return features, labels, groups


def grid_settings(grid: dict):
    """Yields one dict of random forest settings for every combination in the grid."""
    names = list(grid['random_forest'])
    for values in itertools.product(*(grid['random_forest'][name] for name in names)):
        yield dict(zip(names, values))


def main(config_file: str, grid_file: str, result_dir: str, data_atlas_dir: str, data_train_dir: str,
         cache_file: str):
    """Evaluates every random forest setting of the grid with subject-wise cross-validation."""
    config = cfg.load_config(config_file)
    with open(grid_file, 'rb') as f:
        grid = tomllib.load(f)

    features, labels, groups = load_training_data(config, data_atlas_dir, data_train_dir, cache_file)
    print('-' * 5, f'{len(labels)} voxels from {len(np.unique(groups))} subjects, {features.shape[1]} features')

    folds = list(sk_model_selection.GroupKFold(n_splits=grid.get('n_splits', 5)).split(features, labels, groups))
    settings = list(grid_settings(grid))

    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t + '_rf_tuning')
    os.makedirs(result_dir, exist_ok=True)

    rows = []
    for i, setting in enumerate(settings, 1):
        run_config = copy.deepcopy(config)
        run_config['random_forest'].update(setting)
        run_config['random_forest']['n_jobs'] = -1  # use all cores; does not change the results

        for fold, (train_idx, val_idx) in enumerate(folds):
            forest = sk_ensemble.RandomForestClassifier(**cfg.random_forest_params(run_config))

            start_time = timeit.default_timer()
            forest.fit(features[train_idx], labels[train_idx])
            fit_time = timeit.default_timer() - start_time

            start_time = timeit.default_timer()
            predictions = forest.predict(features[val_idx])
            predict_time = timeit.default_timer() - start_time

            # the F1 score of a label equals its Dice coefficient on the validation voxels
            dice = sk_metrics.f1_score(labels[val_idx], predictions, labels=list(LABELS), average=None,
                                       zero_division=0)
            row = {**setting, 'fold': fold, 'mean_dice': np.mean(dice)}
            row.update({name: d for name, d in zip(LABELS.values(), dice)})
            row.update({'fit_time_s': fit_time, 'predict_time_s': predict_time})
            rows.append(row)

        mean = np.mean([r['mean_dice'] for r in rows[-len(folds):]])
        print(f'[{i}/{len(settings)}] {setting} mean Dice: {mean:.4f}')

    write_csv(os.path.join(result_dir, 'tuning_results.csv'), rows)
    summary = summarize(rows, list(settings[0]))
    write_csv(os.path.join(result_dir, 'tuning_summary.csv'), summary)

    print('\nBest settings (mean over folds):')
    for row in summary[:10]:
        print(row)


def summarize(rows: list, setting_names: list) -> list:
    """Averages the fold results per setting, sorted by mean Dice (best first)."""
    metric_names = ['mean_dice', *LABELS.values(), 'fit_time_s', 'predict_time_s']
    summary = {}
    for row in rows:
        key = tuple(row[name] for name in setting_names)
        summary.setdefault(key, []).append(row)

    result = []
    for key, fold_rows in summary.items():
        entry = dict(zip(setting_names, key))
        for name in metric_names:
            entry[name] = round(float(np.mean([r[name] for r in fold_rows])), 4)
        entry['mean_dice_std'] = round(float(np.std([r['mean_dice'] for r in fold_rows])), 4)
        result.append(entry)
    return sorted(result, key=lambda r: r['mean_dice'], reverse=True)


def write_csv(path: str, rows: list):
    """Writes a list of dicts to a semicolon-separated file."""
    with open(path, 'w', newline='') as f:
        csv_writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=';')
        csv_writer.writeheader()
        csv_writer.writerows(rows)


if __name__ == '__main__':
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Random forest tuning with subject-wise cross-validation')
    parser.add_argument('--config', type=str,
                        default=os.path.normpath(os.path.join(script_dir, './configs/default.toml')),
                        help='Experiment config file (pre-processing, features and seed are taken from it).')
    parser.add_argument('--grid', type=str,
                        default=os.path.normpath(os.path.join(script_dir, './configs/rf_tuning_grid.toml')),
                        help='Grid of random forest settings (TOML).')
    parser.add_argument('--cache', type=str, default=None,
                        help='Optional .npz file to save the pre-processed training voxels to and reuse them.')
    parser.add_argument('--result_dir', type=str,
                        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
                        help='Directory for results.')
    parser.add_argument('--data_atlas_dir', type=str,
                        default=os.path.normpath(os.path.join(script_dir, '../MIALab_2020_Dataset/atlas')),
                        help='Directory with atlas data.')
    parser.add_argument('--data_train_dir', type=str,
                        default=os.path.normpath(os.path.join(script_dir, '../MIALab_2020_Dataset/train/')),
                        help='Directory with training data.')

    args = parser.parse_args()
    main(args.config, args.grid, args.result_dir, args.data_atlas_dir, args.data_train_dir, args.cache)
