"""The config module loads experiment settings from TOML files.

Each experiment is described by one file in the configs directory, so experiments can be changed and
repeated without editing the code.
"""
import copy
import tomllib

DEFAULT_CONFIG = {
    'name': 'default',
    'seed': 42,
    'preprocessing': {
        'registration': True,
        'skullstrip': True,
        'normalization': True,
    },
    'features': {
        'coordinates': True,
        'intensity': True,
        'gradient_intensity': True,
    },
    'random_forest': {
        'n_estimators': 1,
        'max_depth': 5,
        'max_features': 'all',
        'n_jobs': 1,
    },
    'postprocessing': {
        'simple': True,
    },
}


def load_config(path: str) -> dict:
    """Loads an experiment config and fills in the default value of every missing setting.

    Args:
        path (str): The path to the TOML file.

    Returns:
        dict: The complete config.

    Raises:
        KeyError: If the file contains a setting that does not exist (e.g. a typo).
    """
    with open(path, 'rb') as f:
        user_config = tomllib.load(f)

    config = copy.deepcopy(DEFAULT_CONFIG)
    _merge(config, user_config, '')
    return config


def _merge(config: dict, user_config: dict, prefix: str):
    """Overwrites the values in config with the values in user_config, rejecting unknown keys."""
    for key, value in user_config.items():
        if key not in config:
            raise KeyError(f'Unknown config setting "{prefix}{key}". Valid settings: {", ".join(config)}')
        if isinstance(config[key], dict):
            if not isinstance(value, dict):
                raise KeyError(f'Config setting "{prefix}{key}" must be a section, e.g. [{prefix}{key}]')
            _merge(config[key], value, f'{prefix}{key}.')
        else:
            config[key] = value


def pre_process_params(config: dict) -> dict:
    """Converts the config to the parameters expected by pipeline_utilities.pre_process_batch.

    Args:
        config (dict): The experiment config.

    Returns:
        dict: The pre-processing and feature extraction parameters.
    """
    return {'registration_pre': config['preprocessing']['registration'],
            'skullstrip_pre': config['preprocessing']['skullstrip'],
            'normalization_pre': config['preprocessing']['normalization'],
            'coordinates_feature': config['features']['coordinates'],
            'intensity_feature': config['features']['intensity'],
            'gradient_intensity_feature': config['features']['gradient_intensity']}


def random_forest_params(config: dict) -> dict:
    """Converts the config to the keyword arguments of sklearn's RandomForestClassifier.

    Args:
        config (dict): The experiment config.

    Returns:
        dict: The random forest parameters.
    """
    forest = config['random_forest']
    # 'all' considers every feature at each split; sklearn expresses this as None
    max_features = None if forest['max_features'] == 'all' else forest['max_features']
    return {'n_estimators': forest['n_estimators'],
            'max_depth': forest['max_depth'],
            'max_features': max_features,
            'n_jobs': forest['n_jobs'],
            'random_state': config['seed']}
