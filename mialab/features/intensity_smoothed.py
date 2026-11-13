"""Smoothed intensities: the T1w and T2w images smoothed with Gaussians of several widths (6 features).

A smoothed intensity describes the tissue around a voxel instead of the single voxel, so it is less sensitive to
noise. The smoothing only uses voxels inside the brain mask (normalized convolution): the zeros outside the brain
are not averaged into voxels near the brain border.
"""
import numpy as np
import SimpleITK as sitk

from mialab.features import FeatureGroup


def masked_smooth(image: sitk.Image, mask: sitk.Image, sigma: float) -> np.ndarray:
    """Gaussian smoothing that only averages voxels inside the mask.

    Args:
        image: The image.
        mask: The brain mask (non-zero inside the brain).
        sigma: The standard deviation of the Gaussian in mm.

    Returns:
        np.ndarray: The smoothed image as array (0 outside the mask).
    """
    inside = sitk.Cast(mask != 0, sitk.sitkFloat32)
    weighted = sitk.SmoothingRecursiveGaussian(sitk.Cast(image, sitk.sitkFloat32) * inside, sigma)
    weights = sitk.SmoothingRecursiveGaussian(inside, sigma)
    values = sitk.GetArrayFromImage(weighted)
    norm = sitk.GetArrayFromImage(weights)
    inside_array = sitk.GetArrayFromImage(inside) > 0
    result = np.zeros_like(values, dtype=np.float32)
    result[inside_array] = values[inside_array] / np.maximum(norm[inside_array], 1e-6)
    return result


class IntensitySmoothed(FeatureGroup):
    """T1w and T2w intensity smoothed with Gaussians of standard deviation ``sigmas`` (mm)."""

    name = 'intensity_smoothed'
    defaults = {'sigmas': [1.0, 2.0, 3.0]}

    def compute(self, images):
        result = {}
        for modality in ('T1w', 'T2w'):
            for sigma in self.params['sigmas']:
                result[f'{modality}_s{sigma:g}'] = masked_smooth(images[modality], images['BrainMask'], sigma)
        return result
