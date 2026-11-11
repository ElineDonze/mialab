"""Boxplots of Dice and HD95 per tissue, for one run or several runs side by side.

Reads only results_per_subject.csv of the run folders, so nothing is trained again. The figure is saved in the
figures/ folder of the project (never inside a run folder, which is never changed after the run).

    python analysis/plot_results.py runs/<run A>                     one run
    python analysis/plot_results.py runs/<run A> runs/<run B> ...    runs side by side, e.g. E0 against E1
    python analysis/plot_results.py                                  the newest run in runs/
"""
import argparse
import csv
import datetime
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mialab.utilities.paths as paths  # noqa: E402

TISSUES = ['WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus']
METRICS = {'dice': 'Dice (higher is better)', 'hd95': 'HD95 in mm (lower is better)'}


def load_run(run_dir: str) -> dict:
    """Returns {metric: {tissue: [value per subject]}} of a run."""
    values = {metric: {tissue: [] for tissue in TISSUES} for metric in METRICS}
    with open(os.path.join(run_dir, 'results_per_subject.csv'), newline='') as f:
        for row in csv.DictReader(f):
            for metric in METRICS:
                values[metric][row['tissue']].append(float(row[metric]))
    return values


def short_name(run_dir: str) -> str:
    """The config name of a run: <date>_<time>_<config name>_<user> without date, time and user."""
    parts = os.path.basename(os.path.normpath(run_dir)).split('_')
    return '_'.join(parts[2:-1]) or parts[-1]


def plot(run_dirs: list, output_file: str):
    runs = [load_run(d) for d in run_dirs]
    labels = [short_name(d) for d in run_dirs]
    width = 0.8 / len(runs)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 + 2 * len(runs), 4.5))
    for ax, (metric, title) in zip(axes, METRICS.items()):
        for i, (run, label) in enumerate(zip(runs, labels)):
            positions = [t + (i - (len(runs) - 1) / 2) * width for t in range(len(TISSUES))]
            boxes = ax.boxplot([run[metric][t] for t in TISSUES], positions=positions, widths=width * 0.9,
                               patch_artist=True, showfliers=True)
            for box in boxes['boxes']:
                box.set_facecolor(colors[i % len(colors)])
                box.set_alpha(0.6)
            ax.plot([], [], color=colors[i % len(colors)], linewidth=6, alpha=0.6, label=label)
        ax.set_xticks(range(len(TISSUES)), TISSUES, rotation=20)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        if metric == 'dice':
            ax.set_ylim(0, 1)
    axes[0].legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print('Saved', output_file)


def main(args):
    run_dirs = args.runs
    if not run_dirs:
        candidates = sorted(d for d in os.listdir(args.runs_dir)
                            if os.path.isfile(os.path.join(args.runs_dir, d, 'results_per_subject.csv')))
        if not candidates:
            sys.exit(f'No finished run found in {args.runs_dir}')
        run_dirs = [os.path.join(args.runs_dir, candidates[-1])]
    output = args.out or os.path.join(
        paths.FIGURES_DIR, f'{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_' + '_vs_'.join(
            short_name(d) for d in run_dirs)[:150] + '.png')
    plot(run_dirs, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Boxplots of Dice and HD95 per tissue for one or more runs')
    parser.add_argument('runs', nargs='*', help='Run folders (default: the newest run).')
    parser.add_argument('--runs_dir', type=str, default=paths.RUNS_DIR)
    parser.add_argument('--out', type=str, help='Output file (default: figures/<date>_<runs>.png).')
    main(parser.parse_args())
