"""A medical image analysis pipeline for brain tissue segmentation with a random forest.

One run evaluates one experiment config:

- Default (cross-validation): 5-fold subject-wise cross-validation on the 20 training subjects, with the folds of
  configs/folds.json. The test subjects are never loaded. Use this for every decision (features, settings).
- ``--final-test``: trains on all training subjects and evaluates on the test subjects. Only for the few runs
  fixed in advance, after all decisions are made.
- ``--smoke``: a quick check of the whole pipeline on 3 training subjects (train on 2, validate on 1).

Steps: pre-processing (registration, skull stripping, normalization), feature extraction (the feature groups
switched on in the config), random forest training on sampled voxels, classification of every brain voxel,
evaluation (Dice and HD95 per tissue and subject).

Pre-processed images and features are cached (see mialab/utilities/feature_cache.py), so they are computed only
once. Each run writes a small folder in runs/ and its segmentations in outputs/ (see run_log.py).

Example:
    python pipeline.py --config configs/experiments/rf_check/E0_baseline_rf50.toml
"""
import argparse
import json
import os
import sys
import timeit

import numpy as np
import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble

try:
    import mialab.data.structure as structure
    import mialab.features as features
    import mialab.utilities.config as cfg
    import mialab.utilities.feature_cache as fcache
    import mialab.utilities.paths as paths
    import mialab.utilities.pipeline_utilities as putil
    import mialab.utilities.run_log as run_log
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.features as features
    import mialab.utilities.config as cfg
    import mialab.utilities.feature_cache as fcache
    import mialab.utilities.paths as paths
    import mialab.utilities.pipeline_utilities as putil
    import mialab.utilities.run_log as run_log

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load (used by rf_tuning)

METRIC_NAMES = {'DICE': 'dice', 'HDRFDST': 'hd95'}


def load_folds(subject_ids: list, folds_file: str = paths.FOLDS_FILE) -> list:
    """Loads the fixed cross-validation folds and checks that they match the training subjects."""
    with open(folds_file) as f:
        folds = json.load(f)['folds']
    in_folds = sorted(s for fold in folds for s in fold)
    if in_folds != sorted(subject_ids):
        raise ValueError(f'{folds_file} does not match the training subjects in the data folder')
    return folds


def train_and_evaluate(config: dict, groups: list, train_cache: fcache.FeatureCache, train_ids: list,
                       eval_cache: fcache.FeatureCache, eval_ids: list, run: run_log.Run, fold: str):
    """Trains a forest on the sampled voxels of train_ids and evaluates it on every brain voxel of eval_ids.

    Returns:
        tuple: result rows (one per subject and tissue), feature importance rows, feature names.
    """
    overlap = set(train_ids) & set(eval_ids) if train_cache is eval_cache else set()
    if overlap:
        raise ValueError(f'Subjects {sorted(overlap)} are in both training and evaluation')

    for group in groups:
        train_cache.ensure_features(group, train_ids, fit_ids=train_ids)
        eval_cache.ensure_features(group, eval_ids, fit_cache=train_cache, fit_ids=train_ids)

    data, labels, _, names = train_cache.matrix(train_ids, groups, 'sample', fit_ids=train_ids)
    print('-' * 5, f'Fold {fold}: training on {len(train_ids)} subjects, {len(labels)} voxels, '
          f'{len(names)} features', flush=True)
    forest = sk_ensemble.RandomForestClassifier(**cfg.random_forest_params(config))
    start_time = timeit.default_timer()
    forest.fit(data, labels)
    print(' Time elapsed:', round(timeit.default_timer() - start_time, 1), 's', flush=True)

    rows = []
    evaluator = putil.init_evaluator()
    segmentation_dir = os.path.join(run.outputs_dir, 'segmentations')
    os.makedirs(segmentation_dir, exist_ok=True)
    for subject in eval_ids:
        data, _, _, _ = eval_cache.matrix([subject], groups, 'brain', fit_ids=train_ids)
        predictions = forest.predict(data).astype(np.uint8)

        # voxels outside the brain mask are background
        reference = eval_cache.reference_image(subject)
        volume = np.zeros(sitk.GetArrayFromImage(reference).shape, dtype=np.uint8)
        volume.ravel()[eval_cache.voxels(subject)['brain_idx']] = predictions
        segmentation = sitk.GetImageFromArray(volume)
        segmentation.CopyInformation(reference)
        sitk.WriteImage(segmentation, os.path.join(segmentation_dir, f'{subject}_SEG.mha'), True)

        ground_truth = eval_cache.load_images(subject, labels=True)['GroundTruth']
        evaluator.evaluate(segmentation, ground_truth, subject)
        per_label = {}
        for result in evaluator.results:
            per_label.setdefault(result.label, {})[METRIC_NAMES.get(result.metric, result.metric)] = result.value
        evaluator.clear()
        for tissue, values in per_label.items():
            rows.append({'fold': fold, 'subject': subject, 'tissue': tissue, **values})
        dice = np.mean([v['dice'] for v in per_label.values()])
        print(f'  {subject}: mean Dice {dice:.4f}', flush=True)

    importances = [{'fold': fold, 'feature': name, 'importance': float(value)}
                   for name, value in zip(names, forest.feature_importances_)]
    return rows, importances, names


def summarize(rows: list) -> list:
    """Mean, standard deviation and median over subjects, per tissue and for the mean over the 5 tissues."""
    tissues = list(dict.fromkeys(r['tissue'] for r in rows))
    subjects = list(dict.fromkeys(r['subject'] for r in rows))
    summary = []
    for tissue in tissues + ['MeanOfTissues']:
        entry = {'tissue': tissue}
        for metric in ('dice', 'hd95'):
            if tissue == 'MeanOfTissues':
                values = [np.mean([r[metric] for r in rows if r['subject'] == s]) for s in subjects]
            else:
                values = [r[metric] for r in rows if r['tissue'] == tissue]
            values = np.asarray(values, dtype=float)
            entry.update({f'{metric}_mean': round(float(np.nanmean(values)), 4),
                          f'{metric}_std': round(float(np.nanstd(values)), 4),
                          f'{metric}_median': round(float(np.nanmedian(values)), 4)})
        entry['n_subjects'] = len(subjects)
        summary.append(entry)
    return summary


def check_final_test_allowed(config: dict, runs_dir: str):
    """Refuses a second final test run of the same config, so the test set is only used once per config."""
    if not os.path.isdir(runs_dir):
        return
    earlier = [d for d in os.listdir(runs_dir) if d.endswith('_FINALTEST') and f'_{config["name"]}_' in d]
    if earlier:
        raise RuntimeError(f'Config "{config["name"]}" was already evaluated on the test set: {earlier}. '
                           'The test set is used only once per config.')


def main(args):
    config = cfg.load_config(args.config)
    mode = 'final_test' if args.final_test else 'smoke' if args.smoke else 'cv'
    print('-' * 5, 'Experiment:', config['name'], f'({args.config}), mode: {mode}')
    if mode == 'smoke':
        config['name'] += '_smoke'
    if mode == 'final_test':
        check_final_test_allowed(config, args.runs_dir)

    groups = [features.create(name, config['feature_params'][name]) for name in cfg.enabled_feature_groups(config)]
    if not groups:
        raise ValueError('No feature group is switched on in the config')
    train_cache = fcache.FeatureCache(config, 'train', args.data_dir, args.cache_dir, args.workers)

    run = run_log.Run(config, mode, args.runs_dir, args.outputs_dir)
    print('-' * 5, 'Run folder:', run.dir, flush=True)
    try:
        rows, importances = [], []
        if mode == 'final_test':
            test_cache = fcache.FeatureCache(config, 'test', args.data_dir, args.cache_dir, args.workers)
            r, imp, names = train_and_evaluate(config, groups, train_cache, train_cache.subject_ids,
                                               test_cache, test_cache.subject_ids, run, 'test')
            rows, importances = r, imp
        else:
            if mode == 'smoke':
                ids = train_cache.subject_ids[:3]
                folds = [ids[2:]]
                all_ids = ids
            else:
                folds = load_folds(train_cache.subject_ids)
                all_ids = train_cache.subject_ids
            for k, val_ids in enumerate(folds):
                train_ids = [s for s in all_ids if s not in val_ids]
                r, imp, names = train_and_evaluate(config, groups, train_cache, train_ids,
                                                   train_cache, val_ids, run, str(k))
                rows += r
                importances += imp

        summary = summarize(rows)
        run.write_csv('results_per_subject.csv', rows)
        run.write_csv('results_summary.csv', summary)
        run.write_csv('feature_importance.csv', importances)
        with open(run.path('features.txt'), 'w') as f:
            f.write('\n'.join(names) + '\n')

        print('\nSummary over subjects:')
        for entry in summary:
            print(f'  {entry["tissue"]:<14} Dice {entry["dice_mean"]:.4f} +- {entry["dice_std"]:.4f}   '
                  f'HD95 {entry["hd95_mean"]:.2f}')
        mean = summary[-1]
        run.finish(n_features=len(names), feature_groups=[g.name for g in groups],
                   mean_dice=mean['dice_mean'], mean_hd95=mean['hd95_mean'])
    except BaseException as error:
        run.fail(error)
        raise


if __name__ == '__main__':
    """The program's entry point."""
    slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '0'))

    parser = argparse.ArgumentParser(description='Brain tissue segmentation with a random forest')
    parser.add_argument('--config', type=str,
                        default=os.path.join(paths.REPO_DIR, 'configs', 'default.toml'),
                        help='Experiment config file (TOML).')
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--final-test', action='store_true',
                            help='Train on all training subjects and evaluate on the test subjects. '
                                 'Only for the final runs fixed in advance.')
    mode_group.add_argument('--smoke', action='store_true',
                            help='Quick check on 3 training subjects (train on 2, validate on 1).')
    parser.add_argument('--workers', type=int, default=max(1, slurm_cpus // 4) if slurm_cpus else 1,
                        help='Subjects pre-processed in parallel (default: 1, or CPUs/4 in a Slurm job).')
    parser.add_argument('--data_dir', type=str, default=paths.DATA_DIR, help='Dataset folder.')
    parser.add_argument('--cache_dir', type=str, default=paths.CACHE_DIR, help='Cache folder.')
    parser.add_argument('--runs_dir', type=str, default=paths.RUNS_DIR, help='Folder for the small run folders.')
    parser.add_argument('--outputs_dir', type=str, default=paths.OUTPUTS_DIR, help='Folder for large outputs.')

    main(parser.parse_args())
