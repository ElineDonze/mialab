"""Offset features: compare a voxel with boxes at a random offset (long-range context, Zikic et al.).

All other groups describe the voxel and its direct neighbourhood. Offset features look further away (up to
``max_offset_mm``) and so describe the anatomical context, e.g. "6 mm below is dark ventricle". They follow the
randomized features of Zikic, Glocker and Criminisi (MICCAI 2013 and Medical Image Analysis 2014). Their forests
draw new offsets in every tree node; a scikit-learn forest can not, so a fixed set of offsets is drawn once with a
fixed seed. All subjects are registered to the same atlas space, so the same offsets mean the same everywhere.

Two groups:

- ``offset_intensity``: on T1w and T2w (no labels, not fold dependent). For every offset one of the three
  difference features of Zikic et al. 2013 (section 2.2, types 2-4) is drawn at random:

  - ``pd``: I(x) - mean of the box of size s at x+u
  - ``bd``: mean of the box of size s at x - mean of the box of size s at x+u
  - ``tb``: mean of the box of size s at x+u - mean of the box of size r at x+v

  The same offsets are used for T1w and T2w. Box means only average voxels inside the brain mask (as in
  ``intensity_smoothed``): the images are z-scored inside the brain and 0 outside, so the zeros outside the brain
  would otherwise look like average tissue. A box that lies completely outside the brain has the mean 0.

- ``offset_atlas``: the probabilistic atlas of the ``spatial`` group read at offset positions, p_tissue(x+u), for
  every tissue and offset (Zikic et al. use the atlas priors as image channels in the same way). It is learned
  from the training labels, so it is fold dependent and fitted exactly like ``spatial`` (on the training subjects
  of the fold only, leave-one-out for the training subjects themselves).

Offsets are drawn in mm, uniformly in a ball of radius ``max_offset_mm`` (shorter than 1 mm is drawn again), and
rounded to whole mm. Positions outside the image read 0.
"""
import numpy as np
import scipy.ndimage
import SimpleITK as sitk

from mialab.features import FeatureGroup
from mialab.features.spatial import Spatial, TISSUES

KINDS = ('pd', 'bd', 'tb')  # the three difference features, see the module docstring


def draw_offsets(rng: np.random.Generator, n: int, max_mm: float) -> np.ndarray:
    """Draws ``n`` offsets (z, y, x) in whole mm, uniformly in a ball of radius ``max_mm``, at least 1 mm long."""
    offsets = []
    while len(offsets) < n:
        candidate = np.round(rng.uniform(-max_mm, max_mm, size=3))
        if 1.0 <= np.linalg.norm(candidate) <= max_mm:
            offsets.append(candidate.astype(int))
    return np.array(offsets, dtype=int).reshape(n, 3)


def to_voxels(offset_mm: np.ndarray, spacing_zyx: tuple) -> tuple:
    """Converts an offset (z, y, x) in mm to whole voxels."""
    return tuple(int(round(o / s)) for o, s in zip(offset_mm, spacing_zyx))


def shifted(array: np.ndarray, offset: tuple) -> np.ndarray:
    """Returns ``out`` with ``out[x] = array[x + offset]``; positions where x + offset is outside the array are 0."""
    out = np.zeros_like(array)
    destination, source = [], []
    for d, n in zip(offset, array.shape):
        if abs(d) >= n:
            return out
        destination.append(slice(max(0, -d), n - max(0, d)))
        source.append(slice(max(0, d), n - max(0, -d)))
    out[tuple(destination)] = array[tuple(source)]
    return out


def box_size_voxels(size_mm: float, spacing: float) -> int:
    """The side of a box in voxels: an odd number (so the box is centred on the voxel), at least 1."""
    return max(1, 2 * int(np.floor((size_mm / spacing - 1) / 2 + 0.5)) + 1)  # half rounds up (2 mm -> 3)


def masked_box_mean(values: np.ndarray, inside: np.ndarray, size: tuple) -> np.ndarray:
    """Mean of the voxels inside the mask within a box of ``size`` voxels around each voxel (0 if there are none)."""
    if all(s == 1 for s in size):
        return np.where(inside, values, 0).astype(np.float32)
    weights = scipy.ndimage.uniform_filter(inside.astype(np.float32), size, mode='constant')
    sums = scipy.ndimage.uniform_filter(np.where(inside, values, 0).astype(np.float32), size, mode='constant')
    mean = np.zeros_like(sums)
    has_brain = weights > 1e-6
    mean[has_brain] = sums[has_brain] / weights[has_brain]
    return mean


def _offset_name(offset_mm) -> str:
    return 'u' + '_'.join(f'{int(o)}' for o in offset_mm)  # z_y_x in mm, no commas (names go into CSV files)


class OffsetIntensity(FeatureGroup):
    """Random offset-difference features on T1w and T2w (``n_offsets`` per image)."""

    name = 'offset_intensity'
    defaults = {'n_offsets': 32, 'max_offset_mm': 10.0, 'sizes_mm': [1.0, 3.0, 5.0], 'seed': 0}

    def __init__(self, **params):
        super().__init__(**params)
        if self.params['n_offsets'] < 1 or self.params['max_offset_mm'] < 1 or not self.params['sizes_mm']:
            raise ValueError(f'"{self.name}" needs n_offsets >= 1, max_offset_mm >= 1 and at least one box size')
        self.draws = self.draw()

    def draw(self) -> list:
        """The random features: one (kind, u, s, v, r) per offset, the same for every subject and both images."""
        rng = np.random.default_rng(self.params['seed'])
        n, max_mm, sizes = self.params['n_offsets'], self.params['max_offset_mm'], self.params['sizes_mm']
        us = draw_offsets(rng, n, max_mm)
        vs = draw_offsets(rng, n, max_mm)
        kinds = rng.integers(0, len(KINDS), size=n)
        s_idx = rng.integers(0, len(sizes), size=n)
        r_idx = rng.integers(0, len(sizes), size=n)
        return [(KINDS[k], us[i], float(sizes[s_idx[i]]), vs[i], float(sizes[r_idx[i]]))
                for i, k in enumerate(kinds)]

    def feature_name(self, modality: str, i: int) -> str:
        kind, u, s, v, r = self.draws[i]
        name = f'{modality}_o{i:02d}_{kind}_s{s:g}_{_offset_name(u)}'
        if kind == 'tb':
            name += f'_r{r:g}_v' + _offset_name(v)[1:]
        return name

    def compute(self, images):
        spacing = images['T1w'].GetSpacing()[::-1]  # (z, y, x), like the numpy arrays
        inside = sitk.GetArrayFromImage(images['BrainMask']) != 0
        result = {}
        for modality in ('T1w', 'T2w'):
            values = sitk.GetArrayFromImage(images[modality]).astype(np.float32)
            means = {}  # box size in mm -> box mean image

            def box_mean(size_mm):
                if size_mm not in means:
                    size = tuple(box_size_voxels(size_mm, sp) for sp in spacing)
                    means[size_mm] = masked_box_mean(values, inside, size)
                return means[size_mm]

            point = np.where(inside, values, 0).astype(np.float32)
            for i, (kind, u, s, v, r) in enumerate(self.draws):
                at_u = shifted(box_mean(s), to_voxels(u, spacing))
                if kind == 'pd':
                    feature = point - at_u
                elif kind == 'bd':
                    feature = box_mean(s) - at_u
                else:
                    feature = at_u - shifted(box_mean(r), to_voxels(v, spacing))
                result[self.feature_name(modality, i)] = feature
        return result


class OffsetAtlas(FeatureGroup):
    """The probabilistic atlas of the 5 tissues read at ``n_offsets`` random offsets (5 x n_offsets features)."""

    name = 'offset_atlas'
    defaults = {'n_offsets': 8, 'max_offset_mm': 10.0, 'sigma': 1.0, 'seed': 1}
    fold_dependent = True

    def __init__(self, **params):
        super().__init__(**params)
        if self.params['n_offsets'] < 1 or self.params['max_offset_mm'] < 1:
            raise ValueError(f'"{self.name}" needs n_offsets >= 1 and max_offset_mm >= 1')
        rng = np.random.default_rng(self.params['seed'])
        self.offsets = draw_offsets(rng, self.params['n_offsets'], self.params['max_offset_mm'])
        self._atlas = Spatial(sigma=self.params['sigma'])  # same atlas as the spatial group

    def fit(self, train_ids, context):
        self._atlas.fit(train_ids, context)

    def compute(self, images):
        spacing = images['T1w'].GetSpacing()[::-1]
        probabilities = self._atlas.probabilities
        result = {}
        for i, u in enumerate(self.offsets):
            offset = to_voxels(u, spacing)
            for t, tissue in enumerate(TISSUES.values()):
                result[f'p_{tissue}_o{i:02d}_{_offset_name(u)}'] = shifted(probabilities[t], offset)
        return result
