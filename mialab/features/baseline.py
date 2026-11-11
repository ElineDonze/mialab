"""The baseline features of the course pipeline (7 features).

These are the same features as in ``FeatureExtractor`` of ``pipeline_utilities.py``: atlas coordinates (3),
T1w and T2w intensity (2) and T1w and T2w gradient magnitude (2).
"""
import SimpleITK as sitk

import mialab.filtering.feature_extraction as fltr_feat
from mialab.features import FeatureGroup


class Coordinates(FeatureGroup):
    """The physical x, y and z coordinates of each voxel in atlas space (mm)."""

    name = 'coordinates'
    enabled_by_default = True

    def compute(self, images):
        coordinates = fltr_feat.AtlasCoordinates().execute(images['T1w'])
        return {axis: sitk.VectorIndexSelectionCast(coordinates, i, sitk.sitkFloat32)
                for i, axis in enumerate(['x', 'y', 'z'])}


class Intensity(FeatureGroup):
    """The normalized T1w and T2w intensities."""

    name = 'intensity'
    enabled_by_default = True

    def compute(self, images):
        return {'T1w': images['T1w'], 'T2w': images['T2w']}


class GradientIntensity(FeatureGroup):
    """The gradient magnitude of the T1w and T2w images."""

    name = 'gradient_intensity'
    enabled_by_default = True

    def compute(self, images):
        return {'T1w': sitk.GradientMagnitude(images['T1w']), 'T2w': sitk.GradientMagnitude(images['T2w'])}
