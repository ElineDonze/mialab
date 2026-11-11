"""Creates one folder per run and records everything needed to understand and repeat it.

Each run gets two folders with the same name::

    runs/<run name>/      small files: settings, code version, environment, metrics (copy this to OneDrive)
    outputs/<run name>/   large files: segmentations (stay where the run was computed)

The run name is ``<date>_<time>_<config name>_<user>``, e.g. ``2026-11-20_14-05-31_E1_texture_eline``.
An existing run folder is never overwritten.
"""
import csv
import datetime
import getpass
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys

import mialab.utilities.paths as paths


def _git(*args) -> str:
    try:
        return subprocess.run(['git', *args], cwd=paths.REPO_DIR, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ''


class Run:
    """A run folder in runs/ and outputs/."""

    def __init__(self, config: dict, mode: str, runs_dir: str = paths.RUNS_DIR,
                 outputs_dir: str = paths.OUTPUTS_DIR):
        self.started = datetime.datetime.now()
        user = os.environ.get('MIALAB_USER') or getpass.getuser()
        self.name = f'{self.started:%Y-%m-%d_%H-%M-%S}_{config["name"]}_{user}'
        if mode == 'final_test':
            self.name += '_FINALTEST'
        self.dir = os.path.join(runs_dir, self.name)
        self.outputs_dir = os.path.join(outputs_dir, self.name)
        if os.path.exists(self.dir) or os.path.exists(self.outputs_dir):
            raise FileExistsError(f'Run folder {self.name} exists already; it is never overwritten')
        os.makedirs(self.dir)
        os.makedirs(self.outputs_dir)

        self.info = {'run': self.name, 'mode': mode, 'config_name': config['name'], 'user': user,
                     'host': socket.gethostname(), 'started': self.started.isoformat(timespec='seconds'),
                     'slurm_job_id': os.environ.get('SLURM_JOB_ID', ''), 'command': ' '.join(sys.argv),
                     'status': 'running'}
        self.write_json('config_used.json', config)
        self._write_code_version()
        self._write_environment()
        self.write_json('run_info.json', self.info)

    def path(self, file_name: str) -> str:
        return os.path.join(self.dir, file_name)

    def write_json(self, file_name: str, value):
        with open(self.path(file_name), 'w') as f:
            json.dump(value, f, indent=4)

    def write_csv(self, file_name: str, rows: list):
        """Writes a list of dicts to a comma-separated file."""
        if not rows:
            return
        with open(self.path(file_name), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _write_code_version(self):
        commit = _git('rev-parse', 'HEAD')
        branch = _git('rev-parse', '--abbrev-ref', 'HEAD')
        changes = _git('status', '--porcelain', '--untracked-files=no')
        with open(self.path('commit.txt'), 'w') as f:
            f.write(f'commit: {commit or "unknown"}\nbranch: {branch}\n'
                    f'uncommitted changes: {"yes, see uncommitted.patch" if changes else "no"}\n')
        if changes:
            with open(self.path('uncommitted.patch'), 'w') as f:
                f.write(_git('diff', 'HEAD'))
        self.info.update({'commit': commit, 'uncommitted_changes': bool(changes)})

    def _write_environment(self):
        packages = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions()
                          if d.metadata['Name'])
        with open(self.path('environment.txt'), 'w') as f:
            f.write(f'python: {sys.version}\nplatform: {platform.platform()}\n\n')
            f.write('\n'.join(packages) + '\n')

    def finish(self, **summary):
        """Marks the run as finished and prints how to copy its small folder to the laptop."""
        ended = datetime.datetime.now()
        self.info.update({'status': 'finished', 'ended': ended.isoformat(timespec='seconds'),
                          'duration_min': round((ended - self.started).total_seconds() / 60, 1), **summary})
        self.write_json('run_info.json', self.info)
        print('\n' + '-' * 5, 'Run finished:', self.dir)
        if os.environ.get('SLURM_JOB_ID'):
            user = getpass.getuser()
            print('Copy the results to your laptop (run this on the laptop, inside your OneDrive runs folder):')
            print(f'    scp -r {user}@submit01.hpc.unibe.ch:{self.dir} .')

    def fail(self, error: BaseException):
        self.info.update({'status': 'failed', 'error': repr(error)})
        self.write_json('run_info.json', self.info)
