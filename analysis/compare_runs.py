"""Compares two runs subject by subject with the paired Wilcoxon signed-rank test.

Both runs must contain the same subjects (e.g. two cross-validation runs with the same folds). Reads only the
saved results, so nothing is trained again.

    python analysis/compare_runs.py runs/<run A> runs/<run B>
"""
import argparse
import csv
import os

import numpy as np
import scipy.stats


def read_rows(run_dir: str) -> dict:
    with open(os.path.join(run_dir, 'results_per_subject.csv'), newline='') as f:
        return {(row['subject'], row['tissue']): row for row in csv.DictReader(f)}


def compare(run_a: str, run_b: str, metrics=('dice', 'hd95')) -> list:
    a, b = read_rows(run_a), read_rows(run_b)
    if set(a) != set(b):
        raise ValueError('The two runs do not contain the same subjects and tissues')
    subjects = sorted({s for s, _ in a})
    tissues = list(dict.fromkeys(t for _, t in a))
    result = []
    for metric in metrics:
        for tissue in tissues + ['MeanOfTissues']:
            def values(rows):
                if tissue == 'MeanOfTissues':
                    return np.array([np.mean([float(rows[(s, t)][metric]) for t in tissues]) for s in subjects])
                return np.array([float(rows[(s, tissue)][metric]) for s in subjects])
            va, vb = values(a), values(b)
            difference = vb - va
            p = scipy.stats.wilcoxon(va, vb).pvalue if np.any(difference != 0) else 1.0
            result.append({'metric': metric, 'tissue': tissue, 'mean_A': round(float(va.mean()), 4),
                           'mean_B': round(float(vb.mean()), 4), 'mean_B_minus_A': round(float(difference.mean()), 4),
                           'B_better_in': int(np.sum(difference > 0) if metric == 'dice' else np.sum(difference < 0)),
                           'n_subjects': len(subjects), 'p_wilcoxon': round(float(p), 4)})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paired Wilcoxon test between two runs')
    parser.add_argument('run_a', help='Run folder A (e.g. the baseline)')
    parser.add_argument('run_b', help='Run folder B')
    parser.add_argument('--out', help='Optional CSV file for the table')
    args = parser.parse_args()

    table = compare(args.run_a, args.run_b)
    print(f'A = {os.path.basename(os.path.normpath(args.run_a))}\nB = {os.path.basename(os.path.normpath(args.run_b))}')
    print('For Dice higher is better, for HD95 lower is better. B_better_in counts subjects.')
    for row in table:
        print('  '.join(f'{k}={v}' for k, v in row.items()))
    if args.out:
        with open(args.out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
