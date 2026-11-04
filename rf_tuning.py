"""Random forest hyperparameter tuning with subject-wise cross-validation.

Every random forest setting in the grid file is trained on the sampled voxels of the training folds, like in
pipeline.py, and evaluated on all brain voxels of the held-out training subjects. This gives a realistic Dice:
on the sampled voxels the large structures are under-represented, which hides most of the false positives.
The test subjects are never used, so the selected settings can afterwards be evaluated on the test set with
pipeline.py without biasing the result.
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
import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
import sklearn.model_selection as sk_model_selection

try:
    import mialab.data.structure as structure
    import mialab.utilities.config as cfg
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.utilities.config as cfg
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil

from pipeline import LOADING_KEYS

LABELS = {1: 'WhiteMatter', 2: 'GreyMatter', 3: 'Hippocampus', 4: 'Amygdala', 5: 'Thalamus'}


def load_training_data(config: dict, data_atlas_dir: str, data_train_dir: str, cache_file: str) -> dict:
    """Pre-processes the training subjects, or loads the result from the cache file.

    Each subject is pre-processed twice: once for the sampled voxels used for training and once for all voxels.
    This is done subject by subject and only the brain voxels are kept, because the full images do not fit into
    memory together.

    Returns:
        dict: The arrays features, labels and groups of the sampled voxels (groups is the subject index per voxel),
            and brain_features, brain_labels and brain_groups of all brain voxels.
    """
    if cache_file and os.path.exists(cache_file):
        print('-' * 5, 'Loading training data from', cache_file)
        with np.load(cache_file) as data:
            return dict(data)

    print('-' * 5, 'Pre-processing training subjects...')
    np.random.seed(config['seed'])
    putil.load_atlas_images(data_atlas_dir)
    crawler = futil.FileSystemDataCrawler(data_train_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())
    params = cfg.pre_process_params(config)

    arrays = {name: [] for name in ('features', 'labels', 'groups', 'brain_features', 'brain_labels', 'brain_groups')}
    for i, (id_, paths) in enumerate(crawler.data.items()):
        sampled = putil.pre_process(id_, dict(paths), **params)
        features, labels = sampled.feature_matrix
        arrays['features'].append(features.astype(np.float32))
        arrays['labels'].append(labels.squeeze().astype(np.uint8))
        arrays['groups'].append(np.full(len(labels), i, dtype=np.uint8))

        full = putil.pre_process(id_, dict(paths), training=False, **params)
        brain = sitk.GetArrayFromImage(full.images[structure.BrainImageTypes.BrainMask]).ravel() > 0
        features, labels = full.feature_matrix
        arrays['brain_features'].append(features[brain].astype(np.float32))
        arrays['brain_labels'].append(labels[brain].squeeze().astype(np.uint8))
        arrays['brain_groups'].append(np.full(np.count_nonzero(brain), i, dtype=np.uint8))

    data = {name: np.concatenate(values) for name, values in arrays.items()}
    if cache_file:
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        np.savez(cache_file, **data)
        print('-' * 5, 'Saved training data to', cache_file)

    return data


def dice_per_label(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Computes the Dice coefficient of every label in LABELS."""
    dice = []
    for label in LABELS:
        prediction, ground_truth = predictions == label, labels == label
        total = np.count_nonzero(prediction) + np.count_nonzero(ground_truth)
        dice.append(2 * np.count_nonzero(prediction & ground_truth) / total if total > 0 else 0.0)
    return np.array(dice)


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

    data = load_training_data(config, data_atlas_dir, data_train_dir, cache_file)
    features, labels, groups = data['features'], data['labels'], data['groups']
    subjects = np.unique(groups)
    print('-' * 5, f'{len(labels)} sampled voxels, {len(data["brain_labels"])} brain voxels, '
          f'{len(subjects)} subjects, {features.shape[1]} features')

    n_splits = grid.get('n_splits', 5)
    folds = list(sk_model_selection.GroupKFold(n_splits=n_splits).split(subjects, groups=subjects))
    settings = list(grid_settings(grid))

    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t + '_rf_tuning')
    os.makedirs(result_dir, exist_ok=True)

    rows = []
    for i, setting in enumerate(settings, 1):
        run_config = copy.deepcopy(config)
        run_config['random_forest'].update(setting)
        run_config['random_forest']['n_jobs'] = -1  # use all cores; does not change the results

        for fold, (train_subjects, val_subjects) in enumerate(folds):
            train_idx = np.isin(groups, subjects[train_subjects])
            forest = sk_ensemble.RandomForestClassifier(**cfg.random_forest_params(run_config))

            start_time = timeit.default_timer()
            forest.fit(features[train_idx], labels[train_idx])
            fit_time = timeit.default_timer() - start_time

            # evaluate per subject on all brain voxels, like the evaluation of the test subjects in pipeline.py
            start_time = timeit.default_timer()
            dice = []
            for subject in subjects[val_subjects]:
                brain_idx = data['brain_groups'] == subject
                predictions = forest.predict(data['brain_features'][brain_idx])
                dice.append(dice_per_label(predictions, data['brain_labels'][brain_idx]))
            predict_time = timeit.default_timer() - start_time
            dice = np.mean(dice, axis=0)

            row = {**setting, 'fold': fold, 'mean_dice': np.mean(dice)}
            row.update({name: d for name, d in zip(LABELS.values(), dice)})
            row.update({'fit_time_s': fit_time, 'predict_time_s': predict_time})
            rows.append(row)

        mean = np.mean([r['mean_dice'] for r in rows[-len(folds):]])
        print(f'[{i}/{len(settings)}] {setting} mean Dice: {mean:.4f}', flush=True)

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
