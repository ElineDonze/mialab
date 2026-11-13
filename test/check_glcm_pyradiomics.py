"""Compares the GLCM features of mialab/features/glcm.py with PyRadiomics on a small piece of a real brain.

PyRadiomics has no release for Python 3.11, so this check runs in a separate small environment with Python 3.9:

    uv venv <env> --python 3.9
    uv pip install --python <env>/bin/python pyradiomics "numpy<2" SimpleITK
    <env>/bin/python test/check_glcm_pyradiomics.py <levels.npz>

The input file is written by the main environment (Python 3.11) with ``--export``:

    python test/check_glcm_pyradiomics.py --export <levels.npz>

It holds the gray levels of a 16 x 16 x 16 block inside the brain of the first training subject, the brain mask of
the block and the features of glcm.py. PyRadiomics then computes voxel-based GLCM features on the same gray levels
(binWidth 1, kernelRadius 1, maskedKernel), and both are compared voxel by voxel.
"""
import os
import sys

import numpy as np

BLOCK = 16


def export(path: str):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import SimpleITK as sitk
    from numpy.lib.stride_tricks import sliding_window_view
    import mialab.utilities.config as cfg
    import mialab.utilities.feature_cache as fcache
    from mialab.features import glcm

    config = cfg.load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs',
                                          'default.toml'))
    cache = fcache.FeatureCache(config, 'train')
    images = cache.load_images(cache.subject_ids[0])
    mask = sitk.GetArrayFromImage(images['BrainMask']) > 0
    values = sitk.GetArrayFromImage(images['T1w'])
    params = glcm.Glcm.defaults
    n_levels = int(np.ceil((params['value_range'][1] - params['value_range'][0]) / params['bin_width']))
    levels = np.where(mask, glcm.gray_levels(values, params['value_range'], params['bin_width']), 0)

    # a block at the brain border (so the masked kernel is tested too), around the centre of the brain mask
    z, y, x = [int(c) for c in np.argwhere(mask).mean(axis=0)]
    border = np.argwhere(mask[z, y, :])[0][0]
    block = (slice(z - BLOCK // 2, z + BLOCK // 2), slice(y - BLOCK // 2, y + BLOCK // 2),
             slice(border - 2, border - 2 + BLOCK))
    block_mask = mask[block]
    # PyRadiomics numbers the gray levels from the lowest level in the mask (level - lowest + 1); the block is
    # renumbered the same way here, so both compute on identical gray levels
    block_levels = levels[block].astype(np.int16)
    block_levels = np.where(block_mask, block_levels - block_levels[block_mask].min() + 1, 0).astype(np.int16)

    # features of glcm.py; the cube of a voxel at the block edge must not reach outside the block,
    # so the block is padded with 0 (= outside the mask), exactly as PyRadiomics sees it
    windows = sliding_window_view(np.pad(block_levels, 1), (3, 3, 3))
    inside = np.nonzero(block_mask)
    ours = glcm.glcm_features(windows[inside].reshape(len(inside[0]), -1), n_levels, 1)
    np.savez(path, levels=block_levels, mask=block_mask, ours=ours, names=np.array(glcm.FEATURES))
    print(f'Saved {path}: {len(inside[0])} brain voxels, gray levels {np.unique(block_levels[block_mask])}')


def compare(path: str):
    import SimpleITK as sitk
    from radiomics import featureextractor

    data = np.load(path)
    levels, mask, ours, names = data['levels'], data['mask'], data['ours'], list(data['names'])
    if np.unique(levels[mask]).min() != 1:
        sys.exit('The gray levels of the block must start at 1, as PyRadiomics numbers them')

    image = sitk.GetImageFromArray(levels.astype(np.float32))
    mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
    extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=1, kernelRadius=1, maskedKernel=True,
                                                           voxelBased=True, initValue=np.nan, force2D=False)
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(glcm=names)
    result = extractor.execute(image, mask_image, voxelBased=True)

    inside = np.nonzero(mask)
    print(f'{len(inside[0])} voxels compared')
    worst = 0.0
    for k, name in enumerate(names):
        # PyRadiomics returns the feature maps cropped to the mask; put them back on the grid of the block
        feature_map = sitk.Resample(result[f'original_glcm_{name}'], image, sitk.Transform(),
                                    sitk.sitkNearestNeighbor, np.nan, sitk.sitkFloat64)
        theirs = sitk.GetArrayFromImage(feature_map)[inside]
        valid = np.isfinite(theirs)
        difference = np.abs(theirs[valid] - ours[valid, k]) / np.maximum(1, np.abs(theirs[valid]))
        worst = max(worst, difference.max())
        print(f'  {name:<18} max relative difference {difference.max():.2e}  ({valid.sum()} voxels)')
    print('SAME as PyRadiomics' if worst < 1e-4 else 'DIFFERENT from PyRadiomics')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--export':
        export(sys.argv[2])
    elif len(sys.argv) == 2:
        compare(sys.argv[1])
    else:
        sys.exit(__doc__)
