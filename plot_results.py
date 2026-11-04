"""Plots the results of a pipeline.py run.

Creates one boxplot per metric in results.csv (e.g. Dice), with one box per label, and saves it next to the
results. Without --result_dir, the newest run in mia-result is used.
"""
import argparse
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt

LABELS = ['WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus']
POST_PROCESSED_SUFFIX = '-PP'  # results.csv lists the post-processed segmentation of a subject as <subject>-PP


def load_results(result_file: str, post_processed: bool) -> dict:
    """Loads a results.csv file.

    Args:
        result_file (str): The path to results.csv.
        post_processed (bool): Whether to load the results with or without post-processing.

    Returns:
        dict: The values as {metric: {label: [value per subject]}}.
    """
    results = {}
    with open(result_file, newline='') as f:
        for row in csv.DictReader(f, delimiter=';'):
            if row['SUBJECT'].endswith(POST_PROCESSED_SUFFIX) != post_processed:
                continue
            for metric, value in row.items():
                if metric not in ('SUBJECT', 'LABEL'):
                    results.setdefault(metric, {}).setdefault(row['LABEL'], []).append(float(value))
    return results


def plot_metric(metric: str, values: dict, title: str, output_file: str):
    """Saves a boxplot of a metric with one box per label."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([values[label] for label in LABELS], tick_labels=LABELS)
    ax.set_title(title)
    ax.set_ylabel(metric)
    if metric == 'DICE':
        ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print('Saved', output_file)


def main(result_dir: str, post_processed: bool):
    """Plots every metric of a run."""
    if result_dir is None:
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        runs = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(script_dir, 'mia-result', '*', 'results.csv')))
        if not runs:
            sys.exit('No results.csv found in mia-result. Run pipeline.py first or use --result_dir.')
        result_dir = runs[-1]

    results = load_results(os.path.join(result_dir, 'results.csv'), post_processed)
    run_name = os.path.basename(os.path.normpath(result_dir))
    suffix = '_post_processed' if post_processed else ''

    for metric, values in results.items():
        plot_metric(metric, values, f'{run_name}{" (post-processed)" if post_processed else ""}',
                    os.path.join(result_dir, f'boxplot_{metric}{suffix}.png'))


if __name__ == '__main__':
    """The program's entry point."""

    parser = argparse.ArgumentParser(description='Plot the results of a pipeline run')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Directory with results.csv. By default, the newest run in mia-result.')
    parser.add_argument('--post_processed', action='store_true',
                        help='Plot the results with post-processing instead of without.')

    args = parser.parse_args()
    main(args.result_dir, args.post_processed)
