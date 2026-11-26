"""Atlas figure for the final presentation: probabilistic atlas maps (learned from the training labels).

Makes coronal slices in the same style as the midterm feature maps (viridis on black, no axes), one PNG per tissue,
plus one overview strip. Uses the repository's own code, so the atlas is exactly the one the random forest saw for
a test subject (fitted on all 20 training subjects). No U-Net needed.

Run it from the root of the mialab repository, in the same Python environment you use for pipeline.py:

    python analysis/atlas_figure.py                 # first test subject, coronal slice y = 136
    python analysis/atlas_figure.py --slice 120     # another coronal slice

If your data is not in ../data/MIALab_2020_Dataset, set MIALAB_DATA=/path/to/MIALab_2020_Dataset first.
The first run registers the 20 training subjects and one test subject if they are not in the cache yet.
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())  # the repository root

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402

try:
    import mialab.features as features  # noqa: E402
    import mialab.utilities.config as cfg  # noqa: E402
    import mialab.utilities.feature_cache as fcache  # noqa: E402
    import mialab.utilities.paths as paths  # noqa: E402
except ImportError as e:
    sys.exit(f'Could not import mialab ({e}). Run this script from the root of the mialab repository.')

TISSUES = ['WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus']


def coronal(volume, y):
    """Coronal slice of a (z, y, x) array, head at the top."""
    return volume[:, y, :]


def save_map(slice_2d, brain_2d, path, cmap='viridis', vmin=None, vmax=None):
    """One map as a square-ish PNG: colour inside the brain, black outside, no axes."""
    data = np.ma.masked_where(~brain_2d, slice_2d)
    fig = plt.figure(figsize=(3, 3 * slice_2d.shape[0] / slice_2d.shape[1]), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('black')
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad('black')
    ax.imshow(data, cmap=cm, origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_axis_off()
    fig.savefig(path, facecolor='black')
    plt.close(fig)


def strip(panels, path, title):
    """Overview: several maps side by side with a title each."""
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.0), facecolor='white')
    for ax, (name, img, brain, kw) in zip(np.atleast_1d(axes), panels):
        cm = plt.get_cmap(kw.get('cmap', 'viridis')).copy()
        cm.set_bad('black')
        ax.imshow(np.ma.masked_where(~brain, img), cmap=cm, origin='lower', interpolation='nearest',
                  vmin=kw.get('vmin'), vmax=kw.get('vmax'))
        ax.set_facecolor('black')
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--subject', default='', help='test subject id (default: the first one)')
    parser.add_argument('--slice', type=int, default=136, help='coronal slice index y (default 136, as the error maps)')
    parser.add_argument('--out', default=os.path.join(paths.FIGURES_DIR, 'atlas_figure'))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    config = cfg.load_config(os.path.join(paths.REPO_DIR, 'configs', 'default.toml'))
    print('data :', paths.DATA_DIR, '\ncache:', paths.CACHE_DIR, flush=True)
    train = fcache.FeatureCache(config, 'train', paths.DATA_DIR, paths.CACHE_DIR)
    test = fcache.FeatureCache(config, 'test', paths.DATA_DIR, paths.CACHE_DIR)
    subject = args.subject or test.subject_ids[0]
    print('test subject:', subject, '| coronal slice y =', args.slice, flush=True)

    train.ensure_preprocessed(train.subject_ids)
    test.ensure_preprocessed([subject])
    images = test.load_images(subject)
    t1 = sitk.GetArrayFromImage(images['T1w']).astype(np.float32)
    brain = sitk.GetArrayFromImage(images['BrainMask']) > 0
    y = args.slice
    b2 = coronal(brain, y)

    # T1 of the same slice, for orientation
    t1s = coronal(t1, y)
    lo, hi = np.percentile(t1s[b2], [1, 99])
    save_map(t1s, b2, os.path.join(args.out, 'T1w.png'), cmap='gray', vmin=lo, vmax=hi)

    # ---- atlas: fitted on all 20 training subjects, as for the test set
    spatial = features.create('spatial')
    spatial.fit(train.subject_ids, features.FitContext(train.load_image))
    panels = [('T1w', t1s, b2, {'cmap': 'gray', 'vmin': lo, 'vmax': hi})]
    for i, tissue in enumerate(TISSUES):
        p = coronal(spatial.probabilities[i], y)
        save_map(p, b2, os.path.join(args.out, f'atlas_{tissue}.png'), vmin=0, vmax=1)
        panels.append((f'Atlas: {tissue}', p, b2, {'vmin': 0, 'vmax': 1}))
    strip(panels, os.path.join(args.out, 'atlas_overview.png'),
          'Probabilistic atlas: share of the 20 training subjects with this tissue at the voxel (0 to 1)')
    print('atlas pictures saved', flush=True)

    print('All pictures in', args.out)


if __name__ == '__main__':
    main()
