"""Feature-feature correlation map of the feature groups switched on in a config.

Uses only the training subjects: a random sample of brain voxels from each of them (no labels are used).
Fold-dependent groups are fitted on all training subjects; this is fine for a figure, but a correlation filter used
to select features (E4) must be computed inside each cross-validation fold instead.

Writes a run folder (runs/<date>_corr_<config name>_<user>/) with:

    correlation.csv        the full matrix (feature x feature)
    group_summary.csv      mean and max absolute correlation between and within feature groups
    high_pairs.csv         feature pairs with |r| >= --threshold
    r2.csv                 (with --r2_targets) how well each target feature is predicted linearly from all
                           features of the other groups: R² close to 1 means the feature adds little new information
    correlation_map.png    the heatmap, with lines between the groups

All numbers are saved, so the figure can be redrawn later (plot_saved) without computing anything again.

    python analysis/correlation_map.py --config configs/experiments/E3_all.toml
"""
import argparse
import csv
import os
import sys

import numpy as np
import scipy.stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mialab.features as features  # noqa: E402
import mialab.utilities.config as cfg  # noqa: E402
import mialab.utilities.feature_cache as fcache  # noqa: E402
import mialab.utilities.paths as paths  # noqa: E402
import mialab.utilities.run_log as run_log  # noqa: E402


def sample_features(cache: fcache.FeatureCache, groups: list, n_voxels: int, seed: int):
    """A random sample of brain voxels, spread equally over the training subjects."""
    ids = cache.subject_ids
    for group in groups:
        cache.ensure_features(group, ids, fit_ids=ids)
    rng = np.random.default_rng(seed)
    per_subject = max(1, n_voxels // len(ids))
    blocks = []
    for subject in ids:
        data, _, _, names = cache.matrix([subject], groups, 'brain', fit_ids=ids)
        blocks.append(data[rng.choice(len(data), size=min(per_subject, len(data)), replace=False)])
    return np.concatenate(blocks), names


def correlation(data: np.ndarray, method: str) -> np.ndarray:
    if method == 'spearman':
        data = np.apply_along_axis(scipy.stats.rankdata, 0, data)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.corrcoef(data, rowvar=False)  # constant features give nan


def group_of(name: str) -> str:
    return name.split(':', 1)[0]


def group_summary(matrix: np.ndarray, names: list) -> list:
    groups = list(dict.fromkeys(group_of(n) for n in names))
    index = {g: [i for i, n in enumerate(names) if group_of(n) == g] for g in groups}
    rows = []
    for a in groups:
        for b in groups:
            block = np.abs(matrix[np.ix_(index[a], index[b])])
            if a == b:
                block = block[~np.eye(len(index[a]), dtype=bool)]
            block = block[np.isfinite(block)]
            rows.append({'group_a': a, 'group_b': b,
                         'mean_abs_r': round(float(block.mean()), 4) if block.size else '',
                         'max_abs_r': round(float(block.max()), 4) if block.size else ''})
    return rows


def high_pairs(matrix: np.ndarray, names: list, threshold: float) -> list:
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.isfinite(matrix[i, j]) and abs(matrix[i, j]) >= threshold:
                rows.append({'feature_a': names[i], 'feature_b': names[j], 'r': round(float(matrix[i, j]), 4)})
    return sorted(rows, key=lambda r: -abs(r['r']))


def r2_from_other_groups(data: np.ndarray, names: list, targets: list) -> list:
    """R² of a linear regression that predicts each target feature from all features of the other groups."""
    rows = []
    for g in targets:
        target_idx = [i for i, n in enumerate(names) if group_of(n) == g]
        source_idx = [i for i, n in enumerate(names) if group_of(n) not in targets]
        if not target_idx or not source_idx:
            continue
        sources = np.column_stack([data[:, source_idx], np.ones(len(data))])
        for i in target_idx:
            y = data[:, i]
            coefficients, *_ = np.linalg.lstsq(sources, y, rcond=None)
            residual = y - sources @ coefficients
            total = np.sum((y - y.mean()) ** 2)
            r2 = 1 - np.sum(residual ** 2) / total if total > 0 else float('nan')
            rows.append({'feature': names[i], 'r2_from_other_groups': round(float(r2), 4)})
    return rows


def plot_saved(run_dir: str):
    """Draws correlation_map.png from correlation.csv (can be called again later with other settings)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(os.path.join(run_dir, 'correlation.csv'), newline='') as f:
        reader = csv.reader(f)
        names = next(reader)[1:]
        matrix = np.array([[float(v) if v else np.nan for v in row[1:]] for row in reader])

    size = min(4 + 0.18 * len(names), 30)
    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    image = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label='correlation')
    groups = [group_of(n) for n in names]
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color='black', linewidth=0.8)
            ax.axvline(i - 0.5, color='black', linewidth=0.8)
    if len(names) <= 80:
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_title('Feature correlation (training voxels)')
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, 'correlation_map.png'), dpi=150)
    plt.close(fig)


def write_csv(path: str, rows: list):
    if rows:
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(args):
    config = cfg.load_config(args.config)
    config['name'] = 'corr_' + config['name']
    groups = [features.create(n, config['feature_params'][n]) for n in cfg.enabled_feature_groups(config)]
    cache = fcache.FeatureCache(config, 'train', args.data_dir, args.cache_dir, args.workers)

    run = run_log.Run(config, 'correlation', args.runs_dir, args.outputs_dir)
    try:
        data, names = sample_features(cache, groups, args.n_voxels, config['seed'])
        matrix = correlation(data, args.method)
        with open(run.path('correlation.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([args.method] + names)
            for name, row in zip(names, matrix):
                writer.writerow([name] + ['' if not np.isfinite(v) else round(float(v), 4) for v in row])
        write_csv(run.path('group_summary.csv'), group_summary(matrix, names))
        write_csv(run.path('high_pairs.csv'), high_pairs(matrix, names, args.threshold))
        if args.r2_targets:
            write_csv(run.path('r2.csv'), r2_from_other_groups(data, names, args.r2_targets))
        plot_saved(run.dir)
        run.finish(n_features=len(names), n_voxels=len(data), method=args.method,
                   feature_groups=[g.name for g in groups])
    except BaseException as error:
        run.fail(error)
        raise


if __name__ == '__main__':
    slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '0'))
    parser = argparse.ArgumentParser(description='Feature correlation map on the training subjects')
    parser.add_argument('--config', type=str, required=True, help='Config whose switched-on groups are compared.')
    parser.add_argument('--method', choices=['spearman', 'pearson'], default='spearman')
    parser.add_argument('--n_voxels', type=int, default=100000, help='Brain voxels sampled in total.')
    parser.add_argument('--threshold', type=float, default=0.9, help='|r| from which a pair is listed.')
    parser.add_argument('--r2_targets', nargs='*', default=[],
                        help='Groups (e.g. learned features) to predict from all other groups (R²).')
    parser.add_argument('--workers', type=int, default=max(1, slurm_cpus // 4) if slurm_cpus else 1)
    parser.add_argument('--data_dir', type=str, default=paths.DATA_DIR)
    parser.add_argument('--cache_dir', type=str, default=paths.CACHE_DIR)
    parser.add_argument('--runs_dir', type=str, default=paths.RUNS_DIR)
    parser.add_argument('--outputs_dir', type=str, default=paths.OUTPUTS_DIR)
    main(parser.parse_args())
