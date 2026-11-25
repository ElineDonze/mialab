"""Computes the deep features of one pretrained network and stores them next to the cached images.

Needs PyTorch (see requirements-deep.txt and ubelix/extract_deep_features.sbatch). The networks are only applied,
not trained, so a CPU is enough (about 15 s per subject); a GPU is faster. The feature groups in
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
   concatenated, for T1w and T2w: 2 x (256 + 512) = 1536 channels at every 4th voxel (x 3 for the three slice
   directions of the "_3planes" networks; "_25d" networks see the neighbouring slices, see hypercolumns()).
3. The channels are standardized and reduced to 16 with PCA. The PCA is fitted once, on randomly chosen brain
   positions of the 20 training subjects, and stored in the cache; the test subjects are only projected.
4. The 16 maps are stored as float16 at stride 4. Grid value i lies at voxel 4 * i: the images are padded at the
   end only, and the stride-4 outputs of a ResNet are centred on input voxel 4 * i.

Layer comparison (networks "<2D network>_layer1" ... "_layer4", axial slices): only one layer is used instead of
the layer1 + layer2 hypercolumns (2 x 256 / 512 / 1024 / 2048 channels at stride 4 / 8 / 16 / 32). The PCA is
fitted and applied at the layer's own stride, and the 16 maps are then linearly upsampled to the stride-4 grid
(position i of a stride-s layer lies at voxel s * i). Standardization, PCA and linear interpolation are all
linear, so this gives the same maps as upsampling the channels first, at a fraction of the memory.
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
LAYER_STRIDES = {'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32}


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

    blocks = [model.layer1, model.layer2, model.layer3, model.layer4]

    def forward(batch, n_layers: int = 2):
        """The outputs of the first ``n_layers`` layers (default: layer1 and layer2)."""
        outputs = [blocks[0](stem(batch))]
        for block in blocks[1:n_layers]:
            outputs.append(block(outputs[-1]))
        return tuple(outputs)
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


def upsample_to_stride4(maps: np.ndarray, factor: int, size: tuple) -> np.ndarray:
    """Axial maps (c, z, h, w) of a stride-(4 * factor) layer to the stride-4 grid (c, z, *size).

    Stride-4 position j reads the layer at j / factor (bilinear, the edges repeat), as upsample_layer2.
    """
    x = torch.from_numpy(maps.astype(np.float32)).permute(1, 0, 2, 3)  # slices as the batch
    if factor > 1:
        inner = tuple(factor * (s - 1) + 1 for s in x.shape[2:])  # align_corners=True maps j to exactly j / f
        x = F.interpolate(x, size=inner, mode='bilinear', align_corners=True)
    # pad (replicate) or crop at the end to the stride-4 grid
    x = F.pad(x, tuple(p for s, n in zip(reversed(size), reversed(x.shape[2:])) for p in (0, max(s - n, 0))),
              mode='replicate')[..., :size[0], :size[1]]
    return x.permute(1, 0, 2, 3).numpy()


def single_layer(network: str) -> str:
    """'resnet_imagenet_layer3' -> 'layer3'; '' for the layer1 + layer2 hypercolumn networks."""
    suffix = network.rsplit('_', 1)[-1]
    return suffix if suffix in LAYER_STRIDES else ''


# ----- hypercolumns -----

def backbone_and_view(network: str) -> tuple:
    """'radimagenet_3planes' -> ('radimagenet', '3planes'); views: axial, 25d, 3planes (2D networks) or 3d."""
    if single_layer(network):  # layer comparison: axial slices
        return network[:-len('_layer1')], 'axial'
    for view in ('25d', '3planes'):
        if network.endswith('_' + view):
            return network[:-len(view) - 1], view
    return network, '3d' if network == 'medicalnet' else 'axial'


def run_2d(forward, slices: torch.Tensor, device: str) -> torch.Tensor:
    """2D network on slices (n, 3, h, w) scaled to 0..1: layer1 + upsampled layer2, (768, n, h / 4, w / 4)."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    parts = []
    for start in range(0, len(slices), SLICES_PER_BATCH):
        batch = slices[start:start + SLICES_PER_BATCH].to(device)
        layer1, layer2 = forward((batch - mean) / std)
        parts.append(torch.cat([layer1, upsample_layer2(layer2, layer1.shape[2:])], 1).half().cpu())
    return torch.cat(parts).permute(1, 0, 2, 3)


def run_2d_layer(forward, slices: torch.Tensor, device: str, layer: str) -> torch.Tensor:
    """2D network on slices (n, 3, h, w) scaled to 0..1: one layer at its own stride, (channels, n, h / s, w / s)."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    n_layers = int(layer[-1])
    parts = []
    for start in range(0, len(slices), SLICES_PER_BATCH):
        batch = slices[start:start + SLICES_PER_BATCH].to(device)
        parts.append(forward((batch - mean) / std, n_layers)[-1].half().cpu())
    return torch.cat(parts).permute(1, 0, 2, 3)


def grid_stride(network: str) -> int:
    """Stride of the grid on which the network's channels are computed (and the PCA is fitted)."""
    layer = single_layer(network)
    return LAYER_STRIDES[layer] if layer else deep.STRIDE


@torch.inference_mode()
def hypercolumns(forward, network: str, images: dict, device: str) -> np.ndarray:
    """Layer1 + layer2 outputs of T1w and T2w on the stored grid (float16, channels first).

    Views of the 2D networks:

    - axial: every axial slice, its grey value copied to the 3 colour channels (1536 channels).
    - 25d: every axial slice, with the slice below, the slice itself and the slice above as the 3 colour channels,
      so the network sees a little depth (1536 channels).
    - 3planes: axial, coronal and sagittal slices at every 4th voxel (grey value copied to the 3 channels); the
      outputs of the three directions are concatenated on a stride-4 grid in all axes (4608 channels).
    """
    mask = sitk.GetArrayFromImage(images['BrainMask']) > 0
    _, view = backbone_and_view(network)
    columns = []
    for modality in ('T1w', 'T2w'):
        volume = sitk.GetArrayFromImage(images[modality]).astype(np.float32)
        if view == '3d':
            x = torch.from_numpy(pad_end(scale_3d(volume, mask), (0, 1, 2)))[None, None].to(device)
            layer1, layer2 = forward(x)
            columns.append(torch.cat([layer1, upsample_layer2(layer2, layer1.shape[2:])], 1)[0].half().cpu())
        elif view in ('axial', '25d'):
            slices = torch.from_numpy(pad_end(scale_2d(volume, mask), (1, 2)))
            if view == 'axial':
                rgb = slices[:, None].expand(-1, 3, -1, -1)
            else:  # neighbours below and above; the first and last slice repeat themselves
                below = torch.cat([slices[:1], slices[:-1]])
                above = torch.cat([slices[1:], slices[-1:]])
                rgb = torch.stack([below, slices, above], 1)
            if single_layer(network):  # layer comparison: one layer at its own stride
                columns.append(run_2d_layer(forward, rgb, device, single_layer(network)))
            else:
                columns.append(run_2d(forward, rgb, device))  # (channels, z, y, x)
        else:  # 3planes
            padded = pad_end(scale_2d(volume, mask), (0, 1, 2))
            for axis in range(3):
                planes = torch.from_numpy(np.ascontiguousarray(np.moveaxis(padded, axis, 0)[::deep.STRIDE]))
                out = run_2d(forward, planes[:, None].expand(-1, 3, -1, -1), device)
                columns.append(out.movedim(1, axis + 1))  # back to (channels, z, y, x) on the stride-4 grid
    return torch.cat(columns).numpy()


def grid_mask(mask: np.ndarray, grid_shape: tuple, axes: tuple, stride: int = deep.STRIDE) -> np.ndarray:
    """Brain mask at the grid positions (grid value i lies at voxel stride * i)."""
    index = tuple(np.arange(n) * (stride if a in axes else 1) for a, n in enumerate(grid_shape))
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
                         deep.NETWORKS[args.network], grid_stride(args.network))
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
    if not args.network.startswith('resnet_imagenet') and not (args.weights and os.path.isfile(args.weights)):
        raise FileNotFoundError(f'--weights must point to the downloaded {args.network} weights')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('No GPU found; run this script on a GPU node (or pass --device cpu for a small test)')
    forward = load_network(backbone_and_view(args.network)[0], args.weights)
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
        if single_layer(args.network):  # PCA was applied at the layer's stride: upsample to the stride-4 grid
            shape = sitk.GetArrayFromImage(images['T1w']).shape
            size = tuple((n + (-n) % MULTIPLE) // deep.STRIDE for n in shape[1:])
            maps = upsample_to_stride4(maps, grid_stride(args.network) // deep.STRIDE, size).astype(np.float16)
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
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu (the last --device wins).')
    parser.add_argument('--limit', type=int, default=0, help='Only the first n subjects (for a quick test).')
    parser.add_argument('--overwrite', action='store_true', help='Recompute stored features.')
    parser.add_argument('--data_dir', type=str, default=paths.DATA_DIR)
    parser.add_argument('--cache_dir', type=str, default=paths.CACHE_DIR)
    main(parser.parse_args())
