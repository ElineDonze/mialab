"""Feature selection inside a cross-validation fold (experiment E4).

Only the training data of the fold is used, so the evaluation subjects never influence which features are kept:

1. A ranking forest (the random forest settings of the config) is trained on all features and ranks them by
   impurity importance.
2. Correlation filter: going down the ranking, a feature is dropped if its absolute Spearman correlation with a
   feature already kept is above ``correlation_threshold`` (computed on a random subset of the training voxels).
3. The ``n_features`` best remaining features are kept ("all" keeps every feature that passed the filter).
"""
import numpy as np
import scipy.stats
import sklearn.ensemble as sk_ensemble

import mialab.utilities.config as cfg

CORRELATION_VOXELS = 20_000  # training voxels used for the correlation filter


def is_active(config: dict) -> bool:
    selection = config['feature_selection']
    return selection['n_features'] != 'all' or selection['correlation_threshold'] < 1.0


def select(config: dict, data: np.ndarray, labels: np.ndarray, names: list) -> tuple:
    """Selects features on the training data of one fold.

    Returns:
        tuple: indices of the kept columns (in ranking order), and one row per feature with its ranking
        importance, rank and whether and why it was kept or dropped.
    """
    selection = config['feature_selection']
    ranking_forest = sk_ensemble.RandomForestClassifier(**cfg.random_forest_params(config))
    ranking_forest.fit(data, labels)
    order = np.argsort(-ranking_forest.feature_importances_, kind='stable')

    rng = np.random.default_rng(config['seed'])
    subset = rng.choice(len(data), size=min(CORRELATION_VOXELS, len(data)), replace=False)
    with np.errstate(invalid='ignore', divide='ignore'):  # constant features give NaN, set to 0 below
        correlation = np.abs(scipy.stats.spearmanr(data[subset]).correlation) if data.shape[1] > 1 else np.ones((1, 1))
    correlation = np.nan_to_num(np.atleast_2d(correlation), nan=0.0)  # constant features have no correlation

    limit = data.shape[1] if selection['n_features'] == 'all' else int(selection['n_features'])
    kept, rows = [], []
    for rank, column in enumerate(order, start=1):
        too_close = [k for k in kept if correlation[column, k] > selection['correlation_threshold']]
        if too_close:
            status = f'dropped: |r| > {selection["correlation_threshold"]} with {names[too_close[0]]}'
        elif len(kept) >= limit:
            status = 'dropped: below the n_features best'
        else:
            kept.append(column)
            status = 'kept'
        rows.append({'feature': names[column], 'rank': rank,
                     'ranking_importance': float(ranking_forest.feature_importances_[column]), 'status': status})
    return np.array(kept), rows
