"""Trains a 3D convolutional autoencoder on the training subjects and stores its features (E7).

The autoencoder learns to compress small blocks of the brain (T1w and T2w) into 16 numbers per 4 x 4 x 4 voxels
and to rebuild the block from them. Its bottleneck is then used as 16 learned features per voxel, in the same way
as the deep features of the pretrained networks (mialab/features/deep.py, group ``deep_autoencoder``).

Needs PyTorch (see requirements-deep.txt and ubelix/train_autoencoder.sbatch). A CPU is enough for this small
network; a GPU is faster.

    # training subjects: trains the autoencoder on them (images only, no labels) and stores the features
    python scripts/train_autoencoder.py --device cpu
    # test subjects, only before a final test run: uses the autoencoder trained on the training subjects
    python scripts/train_autoencoder.py --device cpu --split test

Details:

- Input: T1w and T2w as 2 channels, z-score inside the brain, 0 outside (as for MedicalNet).
- Training: random 64^3 blocks centred on brain voxels of the 20 training subjects. Denoising: the input gets a
  little Gaussian noise, the target is the clean block; the loss (mean squared error) counts brain voxels only.
  The autoencoder is trained once on all training subjects, like the PCA of the deep features; no labels are used.
- Encoder: 3x3x3 convolutions, two of them with stride 2 (padding 1), so bottleneck value i is centred on input
  voxel 4 * i, as the stride-4 outputs of the pretrained networks. The decoder is only needed for training.
- The 16 bottleneck maps are stored as float16 at stride 4 in all axes and upsampled in the pipeline.
"""
import argparse
import json
import os
import sys
import timeit

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mialab.features.deep as deep  # noqa: E402
import mialab.utilities.config as cfg  # noqa: E402
import mialab.utilities.feature_cache as fcache  # noqa: E402
import mialab.utilities.paths as paths  # noqa: E402
from extract_deep_features import pad_end, scale_3d  # noqa: E402

NETWORK = 'autoencoder'
PATCH = 64
MONITOR_PATCHES = 32  # fixed blocks of the training subjects, to follow the loss without noise


class AutoEncoder(nn.Module):
    """Small 3D convolutional autoencoder: 2 channels -> 16 channels at stride 4 -> 2 channels."""

    def __init__(self, latent: int = deep.N_COMPONENTS):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(64, latent, 1))  # bottleneck, no activation
        self.decoder = nn.Sequential(
            nn.Conv3d(latent, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(16, 2, 3, padding=1))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_subject(cache, subject: str) -> tuple:
    """(2, z, y, x) z-scored T1w and T2w, and the brain mask."""
    images = cache.load_images(subject, labels=False)
    mask = sitk.GetArrayFromImage(images['BrainMask']) > 0
    volume = np.stack([scale_3d(sitk.GetArrayFromImage(images[m]).astype(np.float32), mask)
                       for m in ('T1w', 'T2w')])
    return volume, mask


class PatchSampler:
    """Random 64^3 blocks centred on brain voxels (the volumes are padded with zeros by half a block)."""

    def __init__(self, volumes: list, masks: list):
        h = PATCH // 2
        self.volumes = [np.pad(v, ((0, 0), (h, h), (h, h), (h, h))) for v in volumes]
        self.masks = [np.pad(m, h) for m in masks]
        self.centres = [np.argwhere(m) for m in masks]  # in unpadded coordinates = corner in padded coordinates

    def sample(self, rng, n: int) -> tuple:
        x, m = [], []
        for _ in range(n):
            s = rng.integers(len(self.volumes))
            c = self.centres[s][rng.integers(len(self.centres[s]))]
            window = tuple(slice(i, i + PATCH) for i in c)
            x.append(self.volumes[s][(slice(None),) + window])
            m.append(self.masks[s][window])
        return torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(m)[:, None])


def masked_mse(output, target, mask):
    mask = mask.expand_as(target).float()
    return ((output - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)


def train(cache, args, model_path: str):
    torch.manual_seed(cache.config['seed'])
    rng = np.random.default_rng(cache.config['seed'])
    volumes, masks = zip(*(load_subject(cache, s) for s in cache.subject_ids))
    sampler = PatchSampler(list(volumes), list(masks))
    monitor_x, monitor_m = sampler.sample(np.random.default_rng(cache.config['seed'] + 1), MONITOR_PATCHES)

    model = AutoEncoder().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iterations)
    history = []
    start = timeit.default_timer()
    for iteration in range(1, args.iterations + 1):
        model.train()
        x, m = sampler.sample(rng, args.batch_size)
        x, m = x.to(args.device), m.to(args.device)
        noisy = x + args.noise * torch.randn_like(x) * m  # noise inside the brain only
        loss = masked_mse(model(noisy), x, m)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if iteration % args.log_every == 0 or iteration == args.iterations:
            model.eval()
            with torch.inference_mode():
                monitor = np.mean([masked_mse(model(monitor_x[i:i + 4].to(args.device)), monitor_x[i:i + 4]
                                              .to(args.device), monitor_m[i:i + 4].to(args.device)).item()
                                   for i in range(0, MONITOR_PATCHES, 4)])
            history.append({'iteration': iteration, 'train_loss': round(loss.item(), 5),
                            'monitor_loss': round(float(monitor), 5),
                            'seconds': round(timeit.default_timer() - start)})
            print(f'  iteration {iteration}: train loss {loss.item():.4f}, monitor loss {monitor:.4f} '
                  f'({timeit.default_timer() - start:.0f} s)', flush=True)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    tmp = model_path + f'.tmp{os.getpid()}'
    torch.save({'state_dict': model.state_dict(), 'fit_subjects': list(cache.subject_ids), 'history': history,
                'settings': {k: getattr(args, k) for k in ('iterations', 'batch_size', 'lr', 'noise')}}, tmp)
    os.replace(tmp, model_path)
    print(f'Autoencoder trained on {len(cache.subject_ids)} training subjects, saved to {model_path}', flush=True)


@torch.inference_mode()
def encode(model, cache, subject: str, device: str) -> tuple:
    volume, _ = load_subject(cache, subject)
    x = torch.from_numpy(pad_end(volume, (1, 2, 3)))[None].to(device)
    return model.encoder(x)[0].half().cpu().numpy(), volume.shape[1:]


def main(args):
    config = cfg.load_config(args.config)
    if args.limit and os.path.abspath(args.cache_dir) == os.path.abspath(paths.CACHE_DIR):
        raise ValueError('--limit is only for tests: pass a separate --cache_dir, so the real cache never gets '
                         'an autoencoder trained on fewer subjects')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('No GPU found; pass --device cpu or run on a GPU node')
    cache = fcache.FeatureCache(config, args.split, args.data_dir, args.cache_dir)
    if args.limit:
        cache.subject_ids = cache.subject_ids[:args.limit]
    cache.ensure_preprocessed(cache.subject_ids)
    model_path = os.path.join(args.cache_dir, cache.prep_key, f'deep_{NETWORK}_v{deep.EXTRACTION_VERSION}.pt')
    start = timeit.default_timer()

    if not os.path.exists(model_path):
        if args.split != 'train':
            raise FileNotFoundError(f'{model_path} is missing: train on the training subjects first (--split train)')
        train(cache, args, model_path)
    saved = torch.load(model_path, map_location='cpu', weights_only=False)
    model = AutoEncoder().to(args.device)
    model.load_state_dict(saved['state_dict'])
    model.eval()

    for subject in cache.subject_ids:
        path = deep.stored_path(os.path.join(cache.root, subject), NETWORK)
        if os.path.exists(path) and not args.overwrite:
            continue
        maps, shape = encode(model, cache, subject, args.device)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f'.tmp{os.getpid()}.npz'
        np.savez(tmp, maps=maps, shape=np.array(shape), network=NETWORK,
                 fit_subjects=np.array(saved['fit_subjects']))
        os.replace(tmp, path)
        print(f'  {subject}: stored {maps.shape} ({timeit.default_timer() - start:.0f} s)', flush=True)

    info = {'network': NETWORK, 'split': args.split, 'subjects': len(cache.subject_ids), 'model_file': model_path,
            'extraction_version': deep.EXTRACTION_VERSION, 'last_loss': saved['history'][-1] if saved['history']
            else None, 'device': torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu',
            'seconds': round(timeit.default_timer() - start)}
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains the autoencoder and stores its features')
    parser.add_argument('--split', choices=('train', 'test'), default='train')
    parser.add_argument('--config', type=str, default=os.path.join(paths.REPO_DIR, 'configs', 'default.toml'),
                        help='Config with the pre-processing settings and the seed (selects the cache folder).')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu (the last --device wins).')
    parser.add_argument('--iterations', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--noise', type=float, default=0.1, help='Standard deviation of the input noise.')
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--limit', type=int, default=0, help='Only the first n subjects (for a quick test).')
    parser.add_argument('--overwrite', action='store_true', help='Recompute stored features.')
    parser.add_argument('--data_dir', type=str, default=paths.DATA_DIR)
    parser.add_argument('--cache_dir', type=str, default=paths.CACHE_DIR)
    main(parser.parse_args())
