"""Deep features: per-voxel features from pretrained convolutional networks (16 features per network).

Three networks with the same architecture (ResNet-50) but different pretraining data are compared:

- ``deep_resnet_imagenet``: 2D, natural photos (ImageNet, torchvision weights)
- ``deep_radimagenet``: 2D, radiology images (RadImageNet: CT, MRI, ultrasound)
- ``deep_medicalnet``: 3D, medical volumes (MedicalNet, Tencent)

The networks are not trained or fine-tuned; they only describe the image around each voxel. Running them needs
PyTorch, so it is done once by ``scripts/extract_deep_features.py`` (a separate environment, see
``requirements-deep.txt``) and stored next to the cached images. The feature groups here only load the stored
features and need no PyTorch.

How the stored features are made (see the script for details):

1. T1w and T2w are passed through the network separately (2D networks: axial slice by slice).
2. Hypercolumns: the outputs of ``layer1`` (stride 4) and ``layer2`` (stride 8, upsampled to stride 4) are
   concatenated, so every position has local and slightly more global descriptors.
3. PCA reduces the channels to 16. The PCA is fitted on the 20 training subjects only (images, never labels);
   the test subjects are only projected.
4. The 16 maps are stored at stride 4 (value i at voxel 4 * i) and upsampled to the image grid here (linear
   interpolation).

No labels are used anywhere, so the groups are not fold dependent.
"""
import os

import numpy as np
import scipy.ndimage

from mialab.features import FeatureGroup

#: Increase when the extraction changes (layers, input scaling, PCA); old stored features are then not used.
EXTRACTION_VERSION = 1
N_COMPONENTS = 16
STRIDE = 4  # stored grid: every 4th voxel along the downsampled axes

#: network name -> axes of the image array (z, y, x) that are downsampled by the network
NETWORKS = {
    'resnet_imagenet': (1, 2),   # 2D on axial slices: y and x
    'radimagenet': (1, 2),
    'medicalnet': (0, 1, 2),     # 3D
}


def stored_path(subject_dir: str, network: str) -> str:
    """The file with the stored features of one subject and network."""
    return os.path.join(subject_dir, 'deep', f'{network}_v{EXTRACTION_VERSION}.npz')


def upsample(maps: np.ndarray, axes: tuple, shape: tuple) -> np.ndarray:
    """Upsamples stored maps (components first) by STRIDE along ``axes`` to the image ``shape``.

    Stored value i lies at voxel STRIDE * i (the networks pad only at the end, and the centre of their stride-4
    outputs is the input voxel 4 * i), so voxel v reads the stored grid at v / STRIDE with linear interpolation.
    """
    matrix = np.array([1.0 / STRIDE if axis in axes else 1.0 for axis in range(3)])
    result = np.empty((maps.shape[0],) + tuple(shape), dtype=np.float32)
    for i, component in enumerate(maps):
        result[i] = scipy.ndimage.affine_transform(component.astype(np.float32), matrix, output_shape=tuple(shape),
                                                   order=1, mode='nearest')
    return result


class _DeepFeatures(FeatureGroup):
    """Loads the stored features of one pretrained network (``n_components`` of the 16 PCA components)."""

    network = ''
    defaults = {'n_components': N_COMPONENTS}
    version = EXTRACTION_VERSION

    def __init__(self, **params):
        super().__init__(**params)
        if not 1 <= self.params['n_components'] <= N_COMPONENTS:
            raise ValueError(f'n_components of "{self.name}" must be between 1 and {N_COMPONENTS}')

    def compute(self, images):
        raise RuntimeError(f'Feature group "{self.name}" needs the subject folder (compute_for_subject)')

    def compute_for_subject(self, images, subject_dir):
        path = stored_path(subject_dir, self.network)
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} is missing. Run scripts/extract_deep_features.py --network '
                                    f'{self.network} first (see ubelix/extract_deep_features.sbatch)')
        with np.load(path) as stored:
            maps = stored['maps'][:self.params['n_components']]
            shape = tuple(int(s) for s in stored['shape'])
        expected = images['T1w'].GetSize()[::-1]
        if shape != expected:
            raise ValueError(f'{path} was made for an image of shape {shape}, but the image has shape {expected}')
        maps = upsample(maps, NETWORKS[self.network], shape)
        return {f'pc{i + 1:02d}': maps[i] for i in range(len(maps))}


class ResNetImageNet(_DeepFeatures):
    """ResNet-50 pretrained on ImageNet (natural photos), applied to axial slices."""
    name = 'deep_resnet_imagenet'
    network = 'resnet_imagenet'


class RadImageNet(_DeepFeatures):
    """ResNet-50 pretrained on RadImageNet (radiology images), applied to axial slices."""
    name = 'deep_radimagenet'
    network = 'radimagenet'


class MedicalNet(_DeepFeatures):
    """3D ResNet-50 pretrained by MedicalNet on 23 medical datasets."""
    name = 'deep_medicalnet'
    network = 'medicalnet'
