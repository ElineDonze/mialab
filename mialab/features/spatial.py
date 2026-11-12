"""Spatial features: a probabilistic atlas of the 5 tissues and the distance to the brain border (6 features).

- Probabilistic atlas: for every tissue, the share of training subjects that have this tissue at the voxel (all
  subjects are registered to the same atlas space). It is learned from the training labels, so the group is fold
  dependent: it is fitted on the training subjects of each fold only, and a training subject gets an atlas built
  from the other training subjects (leave-one-out, see feature_cache.py). The atlas is smoothed slightly
  (``sigma`` in mm) so that small registration differences matter less.
- Distance to the brain border (mm, positive inside the brain): deep structures such as the thalamus are far from
  the border, cortical grey matter is close to it.
"""
import numpy as np
import SimpleITK as sitk
import scipy.ndimage

from mialab.features import FeatureGroup

TISSUES = {1: 'WhiteMatter', 2: 'GreyMatter', 3: 'Hippocampus', 4: 'Amygdala', 5: 'Thalamus'}


class Spatial(FeatureGroup):
    """Probabilistic atlas of the 5 tissues (learned from the training subjects) and distance to the brain border."""

    name = 'spatial'
    defaults = {'sigma': 1.0}
    fold_dependent = True

    def fit(self, train_ids, context):
        counts = None
        for subject in train_ids:
            labels = sitk.GetArrayFromImage(context.image(subject, 'GroundTruth'))
            if counts is None:
                counts = np.zeros((len(TISSUES),) + labels.shape, dtype=np.float32)
            for i, label in enumerate(TISSUES):
                counts[i] += labels == label
        probabilities = counts / len(train_ids)
        if self.params['sigma'] > 0:
            for i in range(len(TISSUES)):
                probabilities[i] = scipy.ndimage.gaussian_filter(probabilities[i], self.params['sigma'])
        self.probabilities = probabilities

    def compute(self, images):
        result = {f'p_{name}': self.probabilities[i] for i, name in enumerate(TISSUES.values())}
        mask = sitk.Cast(images['BrainMask'] != 0, sitk.sitkUInt8)
        result['distance_to_border'] = sitk.SignedMaurerDistanceMap(mask, insideIsPositive=True,
                                                                    squaredDistance=False, useImageSpacing=True)
        return result
