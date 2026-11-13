"""Builds runs_index.csv: one line per run folder, with its settings and main results.

The index is always rebuilt from the run folders and never edited by hand. Run it wherever the run folders are,
e.g. on the laptop in the OneDrive runs folder, or on UBELIX:

    python analysis/build_runs_index.py --runs_dir "C:/Users/<you>/OneDrive/MIA_Lab/runs"
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mialab.utilities.paths as paths  # noqa: E402


def read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def read_summary(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, newline='') as f:
        return {row['tissue']: row for row in csv.DictReader(f)}


def index_row(run_dir: str) -> dict:
    info = read_json(os.path.join(run_dir, 'run_info.json'))
    config = read_json(os.path.join(run_dir, 'config_used.json'))
    summary = read_summary(os.path.join(run_dir, 'results_summary.csv'))
    forest = config.get('random_forest', {})
    row = {'run': os.path.basename(run_dir),
           'status': info.get('status', ''),
           'mode': info.get('mode', ''),
           'config_name': info.get('config_name', ''),
           'config_file': info.get('config_file', ''),
           'user': info.get('user', ''),
           'started': info.get('started', ''),
           'duration_min': info.get('duration_min', ''),
           'feature_groups': ' '.join(info.get('feature_groups', [])),
           'n_features': info.get('n_features', ''),
           'n_estimators': forest.get('n_estimators', ''),
           'max_depth': forest.get('max_depth', ''),
           'max_features': forest.get('max_features', ''),
           'seed': config.get('seed', ''),
           'commit': info.get('commit', '')[:8],
           'uncommitted_changes': info.get('uncommitted_changes', ''),
           'slurm_job_id': info.get('slurm_job_id', '')}
    for tissue, values in summary.items():
        row[f'dice_{tissue}'] = values.get('dice_mean', '')
    for tissue, values in summary.items():
        row[f'hd95_{tissue}'] = values.get('hd95_mean', '')
    return row


def main(runs_dir: str):
    run_dirs = sorted(os.path.join(runs_dir, d) for d in os.listdir(runs_dir)
                      if os.path.isfile(os.path.join(runs_dir, d, 'run_info.json')))
    rows = [index_row(d) for d in run_dirs]
    if not rows:
        print('No run folders found in', runs_dir)
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path = os.path.join(runs_dir, 'runs_index.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {path} ({len(rows)} runs)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rebuild runs_index.csv from the run folders')
    parser.add_argument('--runs_dir', type=str, default=paths.RUNS_DIR)
    main(parser.parse_args().runs_dir)
