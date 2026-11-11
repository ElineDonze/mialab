"""Random forest tuning: runs the cross-validation of pipeline.py once for every setting of a grid.

Every setting becomes a normal run (its own folder in runs/, same folds, cache and evaluation as pipeline.py), so
the results can be compared with analysis/compare_runs.py and appear in runs_index.csv. The test subjects are never
used.

    python rf_tuning.py --config configs/experiments/E0_baseline.toml --grid configs/rf_tuning_grid.toml

The run names are <config name>_rf<trees>_d<max depth>_mf<max features>, e.g. E0_baseline_rf100_dnone_mfsqrt.
"""
import argparse
import copy
import itertools
import os
import sys
import tomllib

try:
    import mialab.utilities.config as cfg
    import mialab.utilities.paths as paths
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.utilities.config as cfg
    import mialab.utilities.paths as paths

import pipeline


def grid_settings(grid: dict):
    """Yields one dict of random forest settings for every combination in the grid."""
    names = list(grid['random_forest'])
    for values in itertools.product(*(grid['random_forest'][name] for name in names)):
        yield dict(zip(names, values))


def setting_name(setting: dict) -> str:
    short = {'n_estimators': 'rf', 'max_depth': 'd', 'max_features': 'mf'}
    return '_'.join(f'{short.get(k, k)}{v}' for k, v in setting.items())


def main(args):
    config = cfg.load_config(args.config)
    with open(args.grid, 'rb') as f:
        grid = tomllib.load(f)

    results = []
    for setting in grid_settings(grid):
        run_config = copy.deepcopy(config)
        run_config['random_forest'].update(setting)
        run_config['name'] = f'{config["name"]}_{setting_name(setting)}'
        print('\n' + '=' * 5, run_config['name'], flush=True)
        run = pipeline.run_experiment(run_config, 'cv', args.config, args.data_dir, args.cache_dir, args.runs_dir,
                                      args.outputs_dir, args.workers)
        results.append({**setting, 'mean_dice': run.info['mean_dice'], 'mean_hd95': run.info['mean_hd95'],
                        'run': run.name})

    print('\nResults (mean over subjects; compare runs with analysis/compare_runs.py):')
    for row in sorted(results, key=lambda r: -r['mean_dice']):
        print('  ' + '  '.join(f'{k}={v}' for k, v in row.items()))


if __name__ == '__main__':
    slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '0'))
    parser = argparse.ArgumentParser(description='Random forest tuning with the cross-validation of pipeline.py')
    parser.add_argument('--config', type=str, default=os.path.join(paths.REPO_DIR, 'configs', 'experiments',
                                                                   'E0_baseline.toml'),
                        help='Experiment config (features, seed); the grid overrides its random forest settings.')
    parser.add_argument('--grid', type=str, default=os.path.join(paths.REPO_DIR, 'configs', 'rf_tuning_grid.toml'),
                        help='Grid of random forest settings (TOML).')
    parser.add_argument('--workers', type=int, default=max(1, slurm_cpus // 4) if slurm_cpus else 1)
    parser.add_argument('--data_dir', type=str, default=paths.DATA_DIR)
    parser.add_argument('--cache_dir', type=str, default=paths.CACHE_DIR)
    parser.add_argument('--runs_dir', type=str, default=paths.RUNS_DIR)
    parser.add_argument('--outputs_dir', type=str, default=paths.OUTPUTS_DIR)
    main(parser.parse_args())
