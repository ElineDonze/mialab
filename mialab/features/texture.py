"""First-order texture: statistics of the intensities in a small cube around each voxel (16 per image, 32 features).

The 16 statistics are those of ``first_order_texture_features_function`` in the course code (mean, variance,
standard deviation, skewness, kurtosis, entropy, energy, SNR, min, max, range and the 10th, 25th, 50th, 75th and 90th
percentile), computed for T1w and T2w in a cube of (2 * radius + 1)^3 voxels centred on the voxel.

Differences to the course code, which is too slow for whole images (one Python call per voxel):

- All brain voxels are computed at once with numpy, so a subject takes seconds instead of hours.
- The cube is centred on the voxel (the course code pads only one side, so its window is shifted).
- Entropy and energy are computed from a histogram of the cube with fixed bins. The course code uses the intensities
  themselves as probabilities, which is not defined for normalized images with negative values.

Voxels outside the brain mask are 0 after skull stripping and count as part of the cube near the brain border.
"""
import numpy as np
import SimpleITK as sitk
from numpy.lib.stride_tricks import sliding_window_view

from mialab.features import FeatureGroup

STATISTICS = ['mean', 'variance', 'std', 'skewness', 'kurtosis', 'entropy', 'energy', 'snr', 'min', 'max', 'range',
              'p10', 'p25', 'p50', 'p75', 'p90']


def first_order_statistics(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """The 16 first-order statistics of every row of ``values`` (one row = the cube of one voxel).

    Args:
        values: Array of shape (voxels, values per cube).
        bin_edges: Inner edges of the histogram bins used for entropy and energy.

    Returns:
        np.ndarray: Array of shape (voxels, 16), in the order of STATISTICS.
    """
    values = values.astype(np.float64)
    n = values.shape[1]
    mean = values.mean(axis=1)
    centred = values - mean[:, None]
    variance = (centred ** 2).mean(axis=1)
    std = np.sqrt(variance)
    with np.errstate(divide='ignore', invalid='ignore'):
        # adjusted Fisher-Pearson skewness and kurtosis, as in the course code; 0 for a constant cube
        skewness = np.where(std > 0, np.sqrt(n * (n - 1)) / (n - 2) * (centred ** 3).sum(axis=1) / (n * std ** 3), 0)
        kurtosis = np.where(std > 0, (centred ** 4).sum(axis=1) / (n * std ** 4), 0)
        snr = np.where(std > 0, mean / std, 0)

    bins = np.digitize(values, bin_edges)
    probabilities = np.stack([(bins == b).sum(axis=1) for b in range(len(bin_edges) + 1)], axis=1) / n
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.where(probabilities > 0, probabilities * np.log2(probabilities), 0).sum(axis=1)
    energy = (probabilities ** 2).sum(axis=1)

    minimum, maximum = values.min(axis=1), values.max(axis=1)
    percentiles = np.percentile(values, [10, 25, 50, 75, 90], axis=1).T
    return np.column_stack([mean, variance, std, skewness, kurtosis, entropy, energy, snr, minimum, maximum,
                            maximum - minimum, percentiles]).astype(np.float32)


class Texture(FeatureGroup):
    """First-order statistics of T1w and T2w in a cube of (2 * ``radius`` + 1)^3 voxels.

    Entropy and energy use ``bins`` equal bins between ``bin_range`` (the images are normalized to mean 0 and
    standard deviation 1 inside the brain, so most values are between -3 and 3).
    """

    name = 'texture'
    defaults = {'radius': 1, 'bins': 8, 'bin_range': [-3.0, 3.0]}
    chunk_size = 200_000  # voxels computed at once, to limit memory

    def compute(self, images):
        radius = self.params['radius']
        size = 2 * radius + 1
        bin_edges = np.linspace(*self.params['bin_range'], self.params['bins'] + 1)[1:-1]
        brain = np.nonzero(sitk.GetArrayFromImage(images['BrainMask']) > 0)

        result = {}
        for modality in ('T1w', 'T2w'):
            array = sitk.GetArrayFromImage(images[modality]).astype(np.float32)
            windows = sliding_window_view(np.pad(array, radius), (size, size, size))
            features = np.zeros(array.shape + (len(STATISTICS),), dtype=np.float32)
            for start in range(0, len(brain[0]), self.chunk_size):
                index = tuple(axis[start:start + self.chunk_size] for axis in brain)
                cubes = windows[index].reshape(len(index[0]), -1)
                features[index] = first_order_statistics(cubes, bin_edges)
            for i, statistic in enumerate(STATISTICS):
                result[f'{modality}_{statistic}'] = features[..., i]
        return result
