"""Feature groups: one file per group, switched on and off in the experiment config.

How to add a new feature group (see CONTRIBUTING.md for the full rules):

1. Create a new file in this folder, e.g. ``texture.py``, with a subclass of :class:`FeatureGroup`.
2. Give it a unique ``name``, the ``defaults`` of its parameters, and implement :meth:`FeatureGroup.compute`.
3. Add the class to ``GROUPS`` at the bottom of this file.
4. Switch it on in a config file: ``[features] texture = true``, and change parameters in
   ``[feature_params.texture]``.

Every group returns one scalar image per feature. The images are in atlas space (all subjects are registered to
the MNI atlas), so every voxel of every subject has the same size and position.
"""
import typing as t

import numpy as np
import SimpleITK as sitk


class FeatureGroup:
    """Base class of a feature group.

    Attributes:
        name (str): The unique name of the group, used in the config file and in the feature names.
        defaults (dict): The default value of every parameter of the group.
        enabled_by_default (bool): Whether the group is switched on when the config does not mention it.
        fold_dependent (bool): True if the group learns something from the training subjects (for example a
            probabilistic atlas built from their labels). Such a group is fitted again in every cross-validation
            fold, on the training subjects of that fold only, so no information of the validation subjects leaks
            into the features.
        version (int): Increase this number when you change the computation of a group. The cached features of the
            old version are then not used any more.
    """

    name = ''
    defaults = {}
    enabled_by_default = False
    fold_dependent = False
    version = 1

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise KeyError(f'Unknown parameter(s) {sorted(unknown)} for feature group "{self.name}". '
                           f'Valid parameters: {sorted(self.defaults)}')
        self.params = {**self.defaults, **params}

    def fit(self, train_ids: t.List[str], context: 'FitContext'):
        """Learns from the training subjects. Only called if ``fold_dependent`` is True.

        Args:
            train_ids (list of str): The ids of the training subjects of the current fold.
            context (FitContext): Gives access to the pre-processed images of the training subjects.
        """
        raise NotImplementedError(f'Feature group "{self.name}" is fold dependent but has no fit method')

    def compute(self, images: t.Dict[str, sitk.Image]) -> t.Dict[str, t.Union[sitk.Image, np.ndarray]]:
        """Computes the features of one subject.

        Args:
            images (dict): The pre-processed images of the subject: 'T1w', 'T2w' and 'BrainMask'. The labels
                ('GroundTruth') are never passed, so a feature can not use them by accident.

        Returns:
            dict: One entry per feature, in a fixed order: the feature name and a scalar image (or a numpy array
                with the shape of ``sitk.GetArrayFromImage`` of the input images).
        """
        raise NotImplementedError


class FitContext:
    """Gives a fold-dependent feature group access to the pre-processed training subjects."""

    def __init__(self, load_images: t.Callable[[str], t.Dict[str, sitk.Image]]):
        self._load_images = load_images

    def images(self, subject_id: str) -> t.Dict[str, sitk.Image]:
        """Returns the pre-processed images of a training subject, including its labels ('GroundTruth')."""
        return self._load_images(subject_id)


# The groups are imported here, after FeatureGroup is defined, because they subclass it.
from mialab.features import baseline  # noqa: E402

#: All feature groups, in the order of their columns in the feature matrix.
GROUPS = {group.name: group for group in [
    baseline.Coordinates,
    baseline.Intensity,
    baseline.GradientIntensity,
]}


def create(name: str, params: dict = None) -> FeatureGroup:
    """Creates the feature group with the given name and parameters."""
    if name not in GROUPS:
        raise KeyError(f'Unknown feature group "{name}". Valid groups: {", ".join(GROUPS)}')
    return GROUPS[name](**(params or {}))
