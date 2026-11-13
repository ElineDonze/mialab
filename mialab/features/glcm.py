"""GLCM texture (gray level co-occurrence matrix), with the definitions of PyRadiomics (10 per image, 20 features).

For every brain voxel, a GLCM is computed in the cube of (2 * radius + 1)^3 voxels around it, as PyRadiomics does in
voxel-based extraction (kernelRadius = radius, maskedKernel = True: only cube voxels inside the brain are used):

- distance 1 and the 13 directions of 3D, one GLCM per direction, made symmetrical (symmetricalGLCM = True);
- each feature is computed per direction and averaged over the directions that have at least one pair;
- features (PyRadiomics names): Autocorrelation, ClusterProminence, ClusterShade, Contrast, Correlation (1 if a
  direction has only one gray level), DifferenceAverage, Id, Idm, JointEnergy, JointEntropy.

PyRadiomics itself has no release for our Python version (3.11) and is very slow for every voxel of the brain, so
the same computation is done here with numpy. test/check_glcm_pyradiomics.py compares it with PyRadiomics.

Gray levels: PyRadiomics bins intensities with a fixed bin width starting at the lowest value of each subject, so
the gray level numbers would depend on each subject's minimum. Here the bins are fixed for all subjects:
the normalized intensities (mean 0, standard deviation 1 in the brain) are clipped to ``value_range`` and cut into
bins of ``bin_width``, numbered from 1.
"""
import itertools

import numpy as np
import SimpleITK as sitk
from numpy.lib.stride_tricks import sliding_window_view

from mialab.features import FeatureGroup

FEATURES = ['Autocorrelation', 'ClusterProminence', 'ClusterShade', 'Contrast', 'Correlation', 'DifferenceAverage',
            'Id', 'Idm', 'JointEnergy', 'JointEntropy']

# the 13 directions of 3D (one of each pair of opposite directions), as (dz, dy, dx)
DIRECTIONS = [d for d in itertools.product((-1, 0, 1), repeat=3) if d > (0, 0, 0)]
EPSILON = np.spacing(1)  # PyRadiomics adds this inside the logarithms


def gray_levels(values: np.ndarray, value_range, bin_width: float) -> np.ndarray:
    """Discretizes intensities into gray levels 1, 2, ... with fixed bins."""
    low, high = value_range
    clipped = np.clip(values, low, high - 1e-6)
    return (np.floor((clipped - low) / bin_width) + 1).astype(np.int16)


def kernel_pairs(radius: int, direction) -> tuple:
    """Positions (flat index in the cube) of all voxel pairs (a, a + direction) inside a cube."""
    size = 2 * radius + 1
    first, second = [], []
    for position in itertools.product(range(size), repeat=3):
        neighbour = tuple(p + d for p, d in zip(position, direction))
        if all(0 <= n < size for n in neighbour):
            first.append(np.ravel_multi_index(position, (size,) * 3))
            second.append(np.ravel_multi_index(neighbour, (size,) * 3))
    return np.array(first), np.array(second)


def glcm_features(cubes: np.ndarray, n_levels: int, radius: int) -> np.ndarray:
    """The GLCM features of every cube.

    The features are computed directly from the list of gray level pairs of each direction instead of building the
    matrices: for the symmetrical GLCM, p(i, j) is the share of the pairs (a, b) and (b, a) with levels (i, j), so
    sums over p(i, j) are means over the pairs. JointEnergy and JointEntropy need the share of each pair value,
    which is counted by comparing the pairs with each other. This gives the same values as the matrices (see
    glcm_features_matrix) and is much faster.

    Args:
        cubes: Gray levels of the cubes, shape (voxels, (2 * radius + 1)^3); 0 marks voxels outside the mask.
        n_levels: The highest gray level.
        radius: The cube radius.

    Returns:
        np.ndarray: Shape (voxels, len(FEATURES)), averaged over the directions with at least one pair.
    """
    n_voxels = len(cubes)
    total = np.zeros((n_voxels, len(FEATURES)))
    n_valid = np.zeros(n_voxels)
    for direction in DIRECTIONS:
        first, second = kernel_pairs(radius, direction)
        a0, b0 = cubes[:, first].astype(np.float64), cubes[:, second].astype(np.float64)
        # symmetrical GLCM: every pair is counted as (a, b) and (b, a)
        a, b = np.concatenate([a0, b0], axis=1), np.concatenate([b0, a0], axis=1)
        w = ((a > 0) & (b > 0)).astype(np.float64)
        n = w.sum(axis=1)
        has_pairs = n > 0
        n_safe = np.where(has_pairs, n, 1)

        def mean(x):
            return (w * x).sum(axis=1) / n_safe

        mu = mean(a)  # mu_x = mu_y for a symmetrical GLCM
        variance = mean((a - mu[:, None]) ** 2)
        autocorrelation = mean(a * b)
        difference = np.abs(a - b)
        cluster = a + b - 2 * mu[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            correlation = np.where(variance > 0, (autocorrelation - mu ** 2) / variance, 1.0)

        # share of each pair value: how many pairs have the same (a, b) as this one
        code = np.where(w > 0, a * (n_levels + 1) + b, -1)
        same = ((code[:, :, None] == code[:, None, :]) & (w[:, :, None] > 0)).sum(axis=2)
        share = same / n_safe[:, None]
        joint_energy = mean(share)                                  # sum p^2 = mean over pairs of p
        joint_entropy = -mean(np.log2(np.where(w > 0, share, 1) + EPSILON))

        values = np.column_stack([autocorrelation, mean(cluster ** 4), mean(cluster ** 3), mean(difference ** 2),
                                  correlation, mean(difference), mean(1 / (1 + difference)),
                                  mean(1 / (1 + difference ** 2)), joint_energy, joint_entropy])
        total += np.where(has_pairs[:, None], values, 0)
        n_valid += has_pairs
    return (total / np.maximum(n_valid, 1)[:, None]).astype(np.float32)


def glcm_features_matrix(cubes: np.ndarray, n_levels: int, radius: int) -> np.ndarray:
    """Reference version of glcm_features that builds the matrices like PyRadiomics (slow, used for checks)."""
    n_voxels = len(cubes)
    levels = np.arange(1, n_levels + 1, dtype=np.float64)
    i, j = levels[:, None], levels[None, :]
    total = np.zeros((n_voxels, len(FEATURES)))
    n_valid = np.zeros(n_voxels)
    row = np.arange(n_voxels)[:, None]
    for direction in DIRECTIONS:
        first, second = kernel_pairs(radius, direction)
        a, b = cubes[:, first].astype(np.int64), cubes[:, second].astype(np.int64)
        valid = (a > 0) & (b > 0)
        index = np.concatenate([(row * n_levels + (a - 1)) * n_levels + (b - 1),
                                (row * n_levels + (b - 1)) * n_levels + (a - 1)], axis=1)
        weight = np.concatenate([valid, valid], axis=1)
        counts = np.bincount(index[weight], minlength=n_voxels * n_levels * n_levels)
        counts = counts.reshape(n_voxels, n_levels, n_levels).astype(np.float64)
        pair_sum = counts.sum(axis=(1, 2))
        has_pairs = pair_sum > 0
        p = counts / np.where(has_pairs, pair_sum, 1)[:, None, None]
        px, py = p.sum(axis=2), p.sum(axis=1)
        mu_x, mu_y = (px * levels).sum(axis=1), (py * levels).sum(axis=1)
        sigma_x = np.sqrt((px * (levels - mu_x[:, None]) ** 2).sum(axis=1))
        sigma_y = np.sqrt((py * (levels - mu_y[:, None]) ** 2).sum(axis=1))
        autocorrelation = (p * i * j).sum(axis=(1, 2))
        cluster = i + j - (mu_x + mu_y)[:, None, None]
        difference = np.abs(i - j)
        with np.errstate(divide='ignore', invalid='ignore'):
            correlation = np.where(sigma_x * sigma_y > 0, (autocorrelation - mu_x * mu_y) / (sigma_x * sigma_y), 1.0)
        values = np.column_stack([
            autocorrelation, (p * cluster ** 4).sum(axis=(1, 2)), (p * cluster ** 3).sum(axis=(1, 2)),
            (p * (i - j) ** 2).sum(axis=(1, 2)), correlation, (p * difference).sum(axis=(1, 2)),
            (p / (1 + difference)).sum(axis=(1, 2)), (p / (1 + difference ** 2)).sum(axis=(1, 2)),
            (p ** 2).sum(axis=(1, 2)), -(p * np.log2(p + EPSILON)).sum(axis=(1, 2))])
        total += np.where(has_pairs[:, None], values, 0)
        n_valid += has_pairs
    return (total / np.maximum(n_valid, 1)[:, None]).astype(np.float32)


class Glcm(FeatureGroup):
    """GLCM texture of T1w and T2w in a cube of (2 * ``radius`` + 1)^3 voxels, PyRadiomics definitions."""

    name = 'glcm'
    defaults = {'radius': 1, 'bin_width': 0.5, 'value_range': [-4.0, 4.0], 'modalities': ['T1w', 'T2w']}
    chunk_size = 20_000  # voxels computed at once, to limit memory

    def compute(self, images):
        radius = self.params['radius']
        size = 2 * radius + 1
        low, high = self.params['value_range']
        n_levels = int(np.ceil((high - low) / self.params['bin_width']))
        mask = sitk.GetArrayFromImage(images['BrainMask']) > 0
        brain = np.nonzero(mask)

        result = {}
        for modality in self.params['modalities']:
            values = sitk.GetArrayFromImage(images[modality]).astype(np.float32)
            levels = np.where(mask, gray_levels(values, (low, high), self.params['bin_width']), 0).astype(np.int16)
            windows = sliding_window_view(np.pad(levels, radius), (size, size, size))
            features = np.zeros(levels.shape + (len(FEATURES),), dtype=np.float32)
            for start in range(0, len(brain[0]), self.chunk_size):
                index = tuple(axis[start:start + self.chunk_size] for axis in brain)
                cubes = windows[index].reshape(len(index[0]), -1)
                features[index] = glcm_features(cubes, n_levels, radius)
            for k, feature in enumerate(FEATURES):
                result[f'{modality}_{feature}'] = features[..., k]
        return result
