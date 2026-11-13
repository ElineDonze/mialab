"""Quick checks of the configs and folds, without data. Run before opening a pull request:

    python test/test_configs.py

Checks that every config file loads (no misspelled setting), that every experiment config is named like its file,
that every switched-on feature group exists with valid parameters, and that the folds are a valid 5-fold split.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mialab.features as features  # noqa: E402
import mialab.utilities.config as cfg  # noqa: E402
import mialab.utilities.paths as paths  # noqa: E402
import pipeline  # noqa: E402


def main():
    errors = []
    config_files = sorted(glob.glob(os.path.join(paths.REPO_DIR, 'configs', '**', '*.toml'), recursive=True))
    config_files = [f for f in config_files if not f.endswith('rf_tuning_grid.toml')]
    names = {}
    for config_file in config_files:
        try:
            config = cfg.load_config(config_file)
            pipeline.check_config_name(config, config_file)
            for name in cfg.enabled_feature_groups(config):
                features.create(name, config['feature_params'][name])
            names.setdefault(config['name'], []).append(config_file)
        except Exception as error:  # noqa: BLE001 (report every problem, not only the first)
            errors.append(f'{os.path.relpath(config_file, paths.REPO_DIR)}: {error}')
    for name, files in names.items():
        if len(files) > 1 and name != 'default':
            errors.append(f'config name "{name}" is used by several files: {files}')

    with open(paths.FOLDS_FILE) as f:
        folds = json.load(f)['folds']
    subjects = [s for fold in folds for s in fold]
    if len(folds) != 5 or len(subjects) != 20 or len(set(subjects)) != 20:
        errors.append('configs/folds.json is not a 5-fold split of 20 different subjects')

    print(f'Checked {len(config_files)} config files and the folds.')
    if errors:
        print('Problems:\n  ' + '\n  '.join(errors))
        sys.exit(1)
    print('Everything is fine.')


if __name__ == '__main__':
    main()
