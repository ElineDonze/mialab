"""Compares every pair of a set of runs with the paired Wilcoxon test (see compare_runs.py), with Holm correction.

Use it for a family of runs that should be ranked against each other, e.g. all "E0 + learned features" runs.
With many pairs some p-values are small by chance, so the Holm-adjusted p-value (per metric and tissue, over all
pairs) is reported next to the raw one.

    python analysis/pairwise_wilcoxon.py runs/<run 1> runs/<run 2> runs/<run 3> --out pairwise.csv

The run name used in the table is the experiment name (the run folder without the date and the user).
"""
import argparse
import csv
import itertools
import os
import re

import numpy as np

from compare_runs import compare


def run_name(run_dir: str) -> str:
    name = os.path.basename(os.path.normpath(run_dir))
    name = re.sub(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_', '', name)
    return re.sub(r'_[^_]+$', '', name)  # the user at the end


def holm(p_values: list) -> list:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return [round(float(v), 4) for v in adjusted]


def pairwise(run_dirs: list) -> list:
    rows = []
    for a, b in itertools.combinations(run_dirs, 2):
        for row in compare(a, b):
            rows.append({'A': run_name(a), 'B': run_name(b), **row})
    for key in {(r['metric'], r['tissue']) for r in rows}:
        family = [r for r in rows if (r['metric'], r['tissue']) == key]
        for r, adjusted in zip(family, holm([r['p_wilcoxon'] for r in family])):
            r['p_holm'] = adjusted
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paired Wilcoxon tests between all pairs of runs (Holm-adjusted)')
    parser.add_argument('runs', nargs='+', help='Run folders (at least 2)')
    parser.add_argument('--out', required=True, help='CSV file for the table')
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error('Give at least 2 run folders')

    table = pairwise(args.runs)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f'{len(args.runs)} runs, {len(table) // len({(r["metric"], r["tissue"]) for r in table})} pairs, '
          f'saved to {args.out}')
    print('Mean of tissues (Dice higher, HD95 lower is better):')
    for r in table:
        if r['tissue'] == 'MeanOfTissues':
            print(f'  {r["metric"]:5s} {r["A"]} {r["mean_A"]} vs {r["B"]} {r["mean_B"]}: '
                  f'B better in {r["B_better_in"]}/{r["n_subjects"]}, p {r["p_wilcoxon"]}, Holm {r["p_holm"]}')
