"""Edges at several scales: gradient magnitude and Laplacian of Gaussian (LoG) of T1w and T2w (8 features).

- Gradient magnitude after Gaussian smoothing: how strongly the intensity changes around a voxel, i.e. how close it
  is to a tissue border. The baseline has the unsmoothed gradient; here the borders are measured at larger scales.
- Laplacian of Gaussian: positive in small dark spots, negative in small bright spots and zero on edges, so it
  describes blob-like structures of the size of the Gaussian.
"""
import SimpleITK as sitk

from mialab.features import FeatureGroup


class GradientMultiscale(FeatureGroup):
    """Gradient magnitude and LoG of T1w and T2w for every Gaussian width in ``sigmas`` (mm)."""

    name = 'gradient_multiscale'
    defaults = {'sigmas': [1.0, 2.0]}

    def compute(self, images):
        result = {}
        for modality in ('T1w', 'T2w'):
            image = sitk.Cast(images[modality], sitk.sitkFloat32)
            for sigma in self.params['sigmas']:
                result[f'{modality}_grad_s{sigma:g}'] = sitk.GradientMagnitudeRecursiveGaussian(image, sigma)
            for sigma in self.params['sigmas']:
                result[f'{modality}_log_s{sigma:g}'] = sitk.LaplacianRecursiveGaussian(image, sigma)
        return result
