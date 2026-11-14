"""Computes the deep features of one pretrained network and stores them next to the cached images.

Needs PyTorch and a GPU (see requirements-deep.txt and ubelix/extract_deep_features.sbatch). The feature groups in
mialab/features/deep.py then load the stored features in the normal pipeline, without PyTorch.

    # training subjects: fits the PCA on them (images only, no labels) and stores the features
    python scripts/extract_deep_features.py --network radimagenet --weights <file>
    # test subjects, only before a final test run: uses the PCA of the training subjects
    python scripts/extract_deep_features.py --network radimagenet --weights <file> --split test

What is computed (one network, one subject):

1. T1w and T2w are scaled for the network and passed through it separately. 2D networks see axial slices
   (grey value copied to the 3 colour channels, ImageNet normalization); the 3D network sees the whole volume
   (z-score inside the brain, as in MedicalNet).
2. Hypercolumns: the outputs of layer1 (stride 4) and layer2 (stride 8, linearly upsampled to stride 4) are
   concatenated, for T1w and T2w: 2 x (256 + 512) = 1536 channels at every 4th voxel.
3. The channels are standardized and reduced to 16 with PCA. The PCA is fitted once, on randomly chosen brain
   positions of the 20 training subjects, and stored in the cache; the test subjects are only projected.
4. The 16 maps are stored as float16 at stride 4. Grid value i lies at voxel 4 * i: the images are padded at the
   end only, and the stride-4 outputs of a ResNet are centred on input voxel 4 * i.
"""
import argparse
import json
import os
import sys
import timeit

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mialab.features.deep as deep  # noqa: E402
import mialab.utilities.config as cfg  # noqa: E402
import mialab.utilities.feature_cache as fcache  # noqa: E402
import mialab.utilities.paths as paths  # noqa: E402

PCA_POSITIONS_PER_SUBJECT = 5000
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SLICES_PER_BATCH = 32
MULTIPLE = 8  # images are padded to a multiple of 8, so layer1 and layer2 have exactly 1/4 and 1/8 of the size


# ----- networks -----

def _check_loaded(result, network: str):
    """Fails if the weights did not fit the network. Only the unused classifier heads may differ."""
    heads = ('fc.', 'conv_seg.')
    missing = [k for k in result.missing_keys if not k.startswith(heads)]
    unexpected = [k for k in result.unexpected_keys if not k.startswith(heads)]
    if missing or unexpected:
        raise RuntimeError(f'The weights do not fit the {network} network. Missing: {missing[:10]}, '
                           f'unexpected: {unexpected[:10]}')


def _clean_state_dict(state: dict) -> dict:
    """Removes wrapper prefixes ('module.', 'backbone.', 'model.') that training code adds to the key names."""
    if 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']
    cleaned = {}
    for key, value in state.items():
        for prefix in ('module.', 'backbone.', 'model.'):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    # weights saved from nn.Sequential(*list(resnet.children())[:-1]): 0 = conv1, 1 = bn1, 4-7 = layer1-4
    names = {'0': 'conv1', '1': 'bn1', '4': 'layer1', '5': 'layer2', '6': 'layer3', '7': 'layer4'}
    if any(k.split('.')[0] in names for k in cleaned) and 'conv1.weight' not in cleaned:
        cleaned = {names.get(k.split('.')[0], k.split('.')[0]) + k[len(k.split('.')[0]):]: v
                   for k, v in cleaned.items()}
    return cleaned


def load_network(network: str, weights: str):
    """Returns a function that maps an input batch to (layer1, layer2) outputs."""
    if network in ('resnet_imagenet', 'radimagenet'):
        import torchvision
        if network == 'resnet_imagenet':
            model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = torchvision.models.resnet50(weights=None)
            state = _clean_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
            _check_loaded(model.load_state_dict(state, strict=False), network)
        stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
    elif network == 'medicalnet':
        import monai.networks.nets as monai_nets
        # MedicalNet: 7x7x7 stem convolution with stride 2, shortcut type B, no bias in the shortcut
        model = monai_nets.resnet50(spatial_dims=3, n_input_channels=1, conv1_t_size=7, conv1_t_stride=2,
                                    shortcut_type='B', bias_downsample=False, feed_forward=False)
        state = _clean_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        _check_loaded(model.load_state_dict(state, strict=False), network)
        activation = getattr(model, 'act', None) or getattr(model, 'relu')  # the name depends on the MONAI version
        stem = torch.nn.Sequential(model.conv1, model.bn1, activation, model.maxpool)
    else:
        raise ValueError(f'Unknown network "{network}"')
    model.eval()

    def forward(batch):
        layer1 = model.layer1(stem(batch))
        layer2 = model.layer2(layer1)
        return layer1, layer2
    return forward


# ----- input scaling -----

def scale_2d(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Grey values to 0..1 (0.5 and 99.5 percentile inside the brain), 0 outside the brain."""
    low, high = np.percentile(volume[mask], [0.5, 99.5])
    scaled = np.clip((volume - low) / max(high - low, 1e-6), 0, 1)
    scaled[~mask] = 0
    return scaled.astype(np.float32)


def scale_3d(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score inside the brain, 0 outside (MedicalNet normalization)."""
    values = volume[mask]
    scaled = (volume - values.mean()) / max(values.std(), 1e-6)
    scaled[~mask] = 0
    return scaled.astype(np.float32)


def pad_end(array: np.ndarray, axes: tuple) -> np.ndarray:
    """Pads with zeros at the end of ``axes`` to a multiple of MULTIPLE."""
    widths = [(0, (-array.shape[a]) % MULTIPLE if a in axes else 0) for a in range(array.ndim)]
    return np.pad(array, widths)


def upsample_layer2(layer2: torch.Tensor, size: tuple) -> torch.Tensor:
    """Layer2 (stride 8) to the layer1 grid (stride 4): layer1 position j reads layer2 at j / 2."""
    mode = 'bilinear' if layer2.dim() == 4 else 'trilinear'
    inner = tuple(2 * s - 1 for s in layer2.shape[2:])  # align_corners=True maps j to exactly j / 2
    up = F.interpolate(layer2, size=inner, mode=mode, align_corners=True)
    return F.pad(up, tuple(x for s, n in zip(reversed(size), reversed(inner)) for x in (0, s - n)),
                 mode='replicate')


# ----- hypercolumns -----

@torch.inference_mode()
def hypercolumns(forward, network: str, images: dict, device: str) -> np.ndarray:
    """Layer1 + layer2 outputs of T1w and T2w on the stride-4 grid: array (1536, *grid), float16."""
    mask = sitk.GetArrayFromImage(images['BrainMask']) > 0
    axes = deep.NETWORKS[network]
    columns = []
    for modality in ('T1w', 'T2w'):
        volume = sitk.GetArrayFromImage(images[modality]).astype(np.float32)
        if network == 'medicalnet':
            x = torch.from_numpy(pad_end(scale_3d(volume, mask), axes))[None, None].to(device)
            layer1, layer2 = forward(x)
            columns.append(torch.cat([layer1, upsample_layer2(layer2, layer1.shape[2:])], 1)[0].half().cpu())
        else:
            slices = torch.from_numpy(pad_end(scale_2d(volume, mask), (1, 2)))
            mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
            parts = []
            for start in range(0, len(slices), SLICES_PER_BATCH):
                batch = slices[start:start + SLICES_PER_BATCH, None].to(device).expand(-1, 3, -1, -1)
                layer1, layer2 = forward((batch - mean) / std)
                parts.append(torch.cat([layer1, upsample_layer2(layer2, layer1.shape[2:])], 1).half().cpu())
            columns.append(torch.cat(parts).permute(1, 0, 2, 3))  # (channels, z, y, x)
    return torch.cat(columns).numpy()


def grid_mask(mask: np.ndarray, grid_shape: tuple, axes: tuple) -> np.ndarray:
    """Brain mask at the grid positions (grid value i lies at voxel 4 * i)."""
    index = tuple(np.arange(n) * (deep.STRIDE if a in axes else 1) for a, n in enumerate(grid_shape))
    inside = tuple(i[i < s] for i, s in zip(index, mask.shape))
    result = np.zeros(grid_shape, dtype=bool)
    result[tuple(slice(0, len(i)) for i in inside)] = mask[np.ix_(*inside)]
    return result


# ----- main -----

def fit_pca(forward, args, cache, pca_path: str):
    """Fits standardization + PCA on random brain positions of the training subjects (no labels)."""
    rng = np.random.default_rng(cache.config['seed'])
    samples = []
    for subject in cache.subject_ids:
        images = cache.load_images(subject, labels=False)
        columns = hypercolumns(forward, args.network, images, args.device)
        mask = grid_mask(sitk.GetArrayFromImage(images['BrainMask']) > 0, columns.shape[1:],
                         deep.NETWORKS[args.network])
        positions = np.flatnonzero(mask)
        chosen = rng.choice(positions, size=min(PCA_POSITIONS_PER_SUBJECT, len(positions)), replace=False)
        samples.append(columns.reshape(len(columns), -1)[:, chosen].T.astype(np.float32))
        print(f'  PCA samples from {subject}: {len(chosen)}', flush=True)
    samples = np.concatenate(samples)
    mean = samples.mean(0)
    scale = samples.std(0)
    scale[scale < 1e-6] = 1.0  # channels that are constant (e.g. dead ReLUs) are only centred
    standardized = (samples - mean) / scale
    _, singular, components = np.linalg.svd(standardized - standardized.mean(0), full_matrices=False)
    explained = singular ** 2 / np.sum(singular ** 2)
    os.makedirs(os.path.dirname(pca_path), exist_ok=True)
    tmp = pca_path + f'.tmp{os.getpid()}.npz'
    np.savez(tmp, mean=mean, scale=scale, components=components[:deep.N_COMPONENTS],
             explained=explained[:deep.N_COMPONENTS], fit_subjects=np.array(cache.subject_ids))
    os.replace(tmp, pca_path)
    print(f'PCA fitted on {len(samples)} positions of {len(cache.subject_ids)} training subjects; '
          f'{deep.N_COMPONENTS} components explain {explained[:deep.N_COMPONENTS].sum():.1%} of the variance',
          flush=True)


def main(args):
    config = cfg.load_config(args.config)
    if args.limit and os.path.abspath(args.cache_dir) == os.path.abspath(paths.CACHE_DIR):
        raise ValueError('--limit is only for tests: pass a separate --cache_dir, so the real cache never gets '
                         'a PCA fitted on fewer subjects')
    cache = fcache.FeatureCache(config, args.split, args.data_dir, args.cache_dir)
    if args.limit:
        cache.subject_ids = cache.subject_ids[:args.limit]
    cache.ensure_preprocessed(cache.subject_ids)
    pca_path = os.path.join(args.cache_dir, cache.prep_key,
                            f'deep_{args.network}_v{deep.EXTRACTION_VERSION}_pca.npz')
    if args.network != 'resnet_imagenet' and not (args.weights and os.path.isfile(args.weights)):
        raise FileNotFoundError(f'--weights must point to the downloaded {args.network} weights')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('No GPU found; run this script on a GPU node (or pass --device cpu for a small test)')
    forward = load_network(args.network, args.weights)
    start = timeit.default_timer()

    if not os.path.exists(pca_path):
        if args.split != 'train':
            raise FileNotFoundError(f'{pca_path} is missing: run the training subjects first (--split train)')
        fit_pca(forward, args, cache, pca_path)
    with np.load(pca_path) as pca:
        mean, scale, components = pca['mean'], pca['scale'], pca['components']
        fit_subjects = list(pca['fit_subjects'])

    subjects = cache.subject_ids
    for subject in subjects:
        path = deep.stored_path(os.path.join(cache.root, subject), args.network)
        if os.path.exists(path) and not args.overwrite:
            continue
        images = cache.load_images(subject, labels=False)
        columns = hypercolumns(forward, args.network, images, args.device)
        grid_shape = columns.shape[1:]
        flat = (columns.reshape(len(columns), -1).T.astype(np.float32) - mean) / scale
        maps = (flat @ components.T).T.reshape((len(components),) + grid_shape).astype(np.float16)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f'.tmp{os.getpid()}.npz'
        np.savez(tmp, maps=maps, shape=np.array(sitk.GetArrayFromImage(images['T1w']).shape),
                 network=args.network, pca_fit_subjects=np.array(fit_subjects))
        os.replace(tmp, path)
        print(f'  {subject}: stored {maps.shape} ({timeit.default_timer() - start:.0f} s)', flush=True)

    info = {'network': args.network, 'split': args.split, 'subjects': len(subjects),
            'pca_file': pca_path, 'extraction_version': deep.EXTRACTION_VERSION,
            'device': torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu',
            'seconds': round(timeit.default_timer() - start)}
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stores the deep features of a pretrained network')
    parser.add_argument('--network', required=True, choices=sorted(deep.NETWORKS))
    parser.add_argument('--weights', type=str, default='', help='Weights file (radimagenet and medicalnet).')
    parser.add_argument('--split', choices=('train', 'test'), default='train')
    parser.add_argument('--config', type=str, default=os.path.join(paths.REPO_DIR, 'configs', 'default.toml'),
                        help='Config with the pre-processing settings (selects the cache folder).')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--limit', type=int, default=0, help='Only the first n subjects (for a quick test).')
    parser.add_argument('--overwrite', action='store_true', help='Recompute stored features.')
    parser.add_argument('--data_dir', type=str, default=paths.DATA_DIR)
    parser.add_argument('--cache_dir', type=str, default=paths.CACHE_DIR)
    main(parser.parse_args())
