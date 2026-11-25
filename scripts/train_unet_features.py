"""Trains a small supervised 3D U-Net and stores its outputs as learned features per voxel (E9).

Unlike the pretrained networks (E6) and the autoencoder (E7), the U-Net is trained with the labels. So a subject
must never get features from a U-Net that was trained on its own labels. This is done with out-of-fold models
(stacking), using the same folds as the cross-validation of the random forest (configs/folds.json):

- ``--fold k`` (k = 0..4): trains on the 16 training subjects outside fold k and stores the features of the 4
  subjects of fold k (6 models in total with ``all``).
- ``--fold all``: trains on all 20 training subjects. With ``--predict_test`` it also stores the features of the
  10 test subjects (only before a final test run).

Limitation (see reports/H6_todo_list.md): in cross-validation fold k, the random forest is trained on features
that were made by U-Nets that saw the labels of fold k, so the CV numbers are slightly optimistic. The test
numbers are clean.

    python scripts/train_unet_features.py --fold 0
    python scripts/train_unet_features.py --fold all --predict_test   # only before a final test
    python scripts/train_unet_features.py --fold 0 --steps 200 --smoke  # timing test, stores nothing in the cache
    python scripts/train_unet_features.py --fold all --single   # one model for all subjects (final test only)

``--single`` (with ``--fold all``): the hidden layer of the one U-Net trained on all 20 training subjects is stored
as ``unet_single`` for all 20 training and all 10 test subjects. One model, so the channels mean the same for
every subject. The training subjects were seen by this U-Net, so use these features with
``pipeline.py --final-test`` only (no cross-validation). The out-of-fold files are not touched.

Details:

- Input: T1w and T2w as 2 channels, z-score inside the brain, 0 outside (as for the autoencoder).
- Output: 6 labels (background + 5 tissues), softmax; loss Dice + cross-entropy (MONAI DiceCELoss).
- Training: random 96^3 blocks. One third is centred on a random brain voxel, the rest on a voxel of a random
  tissue label (1-5), so the small structures are seen often. Augmentation: left-right flip and a small
  intensity scale and shift.
- Network: MONAI DynUNet (the nnU-Net architecture), filters 16/32/64/128, instance norm, deep supervision.
- Stored per subject, from sliding-window inference with the same 96^3 blocks, at full resolution:
  ``unet``: the input of the output layer, i.e. the last decoder layer (16 channels), and
  ``unet_prob``: the 6 softmax probabilities (background + 5 tissues).
- Only ``unet_prob`` is valid for stacking: the 16 hidden channels mean something different in every U-Net (each
  training run finds its own arbitrary channel order), so a random forest trained on the hidden channels of some
  out-of-fold models can not read those of another model. The probabilities mean the same in every model.
- The model is saved every ``--ckpt_every`` steps, so a preempted job continues where it stopped.
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
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import DynUNet

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mialab.features.deep as deep  # noqa: E402
import mialab.utilities.config as cfg  # noqa: E402
import mialab.utilities.feature_cache as fcache  # noqa: E402
import mialab.utilities.paths as paths  # noqa: E402
from extract_deep_features import scale_3d  # noqa: E402

NETWORK = 'unet'            # 16 channels of the last decoder layer
NETWORK_PROB = 'unet_prob'  # the 6 softmax probabilities (background + 5 tissues)
NETWORK_SINGLE = 'unet_single'  # hidden layer of the one model on all training subjects, for every subject
N_LABELS = 6
FILTERS = (16, 32, 64, 128)
SUPERVISION_WEIGHTS = (1.0, 0.5, 0.25)  # main output, then the two deep supervision heads


def build_model() -> DynUNet:
    return DynUNet(spatial_dims=3, in_channels=2, out_channels=N_LABELS, kernel_size=[3, 3, 3, 3],
                   strides=[1, 2, 2, 2], upsample_kernel_size=[2, 2, 2], filters=list(FILTERS),
                   norm_name='instance', deep_supervision=True, deep_supr_num=len(SUPERVISION_WEIGHTS) - 1)


class FeaturesAndLogits(nn.Module):
    """Returns the 16 channels of the last decoder layer and the 6 logits, concatenated (22 channels)."""

    def __init__(self, model: DynUNet):
        super().__init__()
        self.model = model
        self._features = None
        model.output_block.register_forward_pre_hook(self._keep)

    def _keep(self, _module, inputs):
        self._features = inputs[0]

    def forward(self, x):
        logits = self.model(x)  # eval mode: main output only
        return torch.cat([self._features, logits], dim=1)


def load_subject(cache, subject: str, labels: bool) -> tuple:
    """(2, z, y, x) z-scored T1w and T2w, the brain mask and (if asked) the labels."""
    images = cache.load_images(subject, labels=labels)
    mask = sitk.GetArrayFromImage(images['BrainMask']) > 0
    volume = np.stack([scale_3d(sitk.GetArrayFromImage(images[m]).astype(np.float32), mask)
                       for m in ('T1w', 'T2w')])
    label = sitk.GetArrayFromImage(images['GroundTruth']).astype(np.int64) if labels else None
    return volume, mask, label


class PatchSampler:
    """Random blocks: a third centred on a brain voxel, the rest on a voxel of a random tissue label."""

    def __init__(self, volumes: list, masks: list, labels: list, patch: int):
        h = patch // 2
        self.patch = patch
        self.volumes = [np.pad(v, ((0, 0), (h, h), (h, h), (h, h))) for v in volumes]
        self.labels = [np.pad(lab, h) for lab in labels]
        self.shapes = [m.shape for m in masks]
        # flat indices (unpadded image) of the possible centres: brain, then each tissue label
        self.centres = [[np.flatnonzero(m).astype(np.int32)] +
                        [np.flatnonzero(lab == k).astype(np.int32) for k in range(1, N_LABELS)]
                        for m, lab in zip(masks, labels)]

    def sample(self, rng, n: int) -> tuple:
        x, y = [], []
        for _ in range(n):
            s = rng.integers(len(self.volumes))
            kind = 0 if rng.random() < 1 / 3 else rng.integers(1, N_LABELS)
            candidates = self.centres[s][kind] if len(self.centres[s][kind]) else self.centres[s][0]
            c = np.unravel_index(candidates[rng.integers(len(candidates))], self.shapes[s])
            window = tuple(slice(i, i + self.patch) for i in c)  # centre c in padded coordinates = corner
            block, target = self.volumes[s][(slice(None),) + window], self.labels[s][window]
            if rng.random() < 0.5:  # left-right flip (x is the last array axis)
                block, target = block[..., ::-1], target[..., ::-1]
            block = block * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1) * (block != 0)
            x.append(np.ascontiguousarray(block, dtype=np.float32))
            y.append(np.ascontiguousarray(target)[None])
        return torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(y))


def supervised_loss(loss_fn, outputs, target):
    """Weighted Dice + CE over the main output and the deep supervision heads (all at full resolution)."""
    total = sum(w * loss_fn(outputs[:, i], target) for i, w in enumerate(SUPERVISION_WEIGHTS))
    return total / sum(SUPERVISION_WEIGHTS)


def dice_per_label(prediction: np.ndarray, truth: np.ndarray) -> dict:
    result = {}
    for k in range(1, N_LABELS):
        p, t = prediction == k, truth == k
        result[k] = round(float(2 * (p & t).sum() / max(p.sum() + t.sum(), 1)), 4)
    return result


def autocast(device: str):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else torch.autocast('cpu',
                                                                                                 enabled=False)


def train(train_cache, train_ids, args, model_path: str):
    seed = train_cache.config['seed'] + (0 if args.fold == 'all' else int(args.fold) + 1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    print(f'Loading {len(train_ids)} training subjects...', flush=True)
    data = [load_subject(train_cache, s, labels=True) for s in train_ids]
    sampler = PatchSampler(*(list(d) for d in zip(*data)), patch=args.patch)
    del data

    model = build_model().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.steps, power=0.9)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    step, history, seconds = 0, [], 0.0
    ckpt_path = model_path + '.ckpt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)  # the RNG state must stay on CPU
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        rng.bit_generator.state = ckpt['rng']
        torch.set_rng_state(ckpt['torch_rng'])
        step, history, seconds = ckpt['step'], ckpt['history'], ckpt['seconds']
        print(f'Resumed from step {step}', flush=True)

    def save(path):
        tmp = path + f'.tmp{os.getpid()}'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'rng': rng.bit_generator.state,
                    'torch_rng': torch.get_rng_state(), 'step': step, 'history': history,
                    'seconds': seconds + timeit.default_timer() - start, 'fit_subjects': list(train_ids),
                    'settings': {k: getattr(args, k) for k in ('steps', 'batch_size', 'lr', 'patch')}}, tmp)
        os.replace(tmp, path)

    start = timeit.default_timer()
    model.train()
    running = []
    while step < args.steps:
        x, y = sampler.sample(rng, args.batch_size)
        x, y = x.to(args.device), y.to(args.device)
        with autocast(args.device):
            outputs = model(x)
        loss = supervised_loss(loss_fn, outputs.float(), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
        optimizer.step()
        scheduler.step()
        step += 1
        running.append(loss.item())
        if step % args.log_every == 0 or step == args.steps:
            elapsed = seconds + timeit.default_timer() - start
            history.append({'step': step, 'loss': round(float(np.mean(running)), 5), 'seconds': round(elapsed)})
            print(f'  step {step}: loss {np.mean(running):.4f} ({elapsed:.0f} s)', flush=True)
            running = []
        if step % args.ckpt_every == 0 and step < args.steps:
            save(ckpt_path)
    save(model_path)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f'U-Net trained on {len(train_ids)} subjects, saved to {model_path}', flush=True)


@torch.inference_mode()
def predict(net: FeaturesAndLogits, volume: np.ndarray, args) -> tuple:
    x = torch.from_numpy(volume)[None].to(args.device)
    with autocast(args.device):
        out = sliding_window_inference(x, (args.patch,) * 3, sw_batch_size=2, predictor=net, overlap=0.5,
                                       mode='gaussian')
    out = out[0].float()
    logits = out[deep.N_COMPONENTS:]
    return (out[:deep.N_COMPONENTS].half().cpu().numpy(), torch.softmax(logits, 0).half().cpu().numpy(),
            logits.argmax(0).cpu().numpy())


def main(args):
    config = cfg.load_config(args.config)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('No GPU found; pass --device cpu or run on a GPU node')
    if args.smoke and not args.out_dir:
        raise ValueError('--smoke needs a separate --out_dir, so the real model folder never gets a test model')
    if args.predict_test and args.fold != 'all':
        raise ValueError('--predict_test needs --fold all (the U-Net trained on all training subjects)')
    if args.single and (args.fold != 'all' or args.predict_test or args.smoke):
        raise ValueError('--single needs --fold all and can not be combined with --predict_test or --smoke')
    with open(os.path.join(paths.REPO_DIR, 'configs', 'folds.json')) as f:
        folds = json.load(f)['folds']

    train_cache = fcache.FeatureCache(config, 'train', args.data_dir, args.cache_dir)
    all_train = list(train_cache.subject_ids)
    if args.fold == 'all':
        train_ids, held_out = all_train, []
    else:
        held_out = folds[int(args.fold)]
        train_ids = [s for s in all_train if s not in held_out]
    if sorted(train_ids + held_out) != sorted(all_train):
        raise ValueError('configs/folds.json does not match the training subjects')
    train_cache.ensure_preprocessed(all_train)

    model_dir = os.path.join(args.out_dir or os.path.join(args.cache_dir, train_cache.prep_key, 'unet'))
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'fold{args.fold}_v{deep.EXTRACTION_VERSION}.pt')
    start = timeit.default_timer()
    if not os.path.exists(model_path):
        train(train_cache, train_ids, args, model_path)
    saved = torch.load(model_path, map_location='cpu', weights_only=False)
    if sorted(saved['fit_subjects']) != sorted(train_ids):
        raise ValueError(f'{model_path} was trained on other subjects')
    model = build_model().to(args.device)
    model.load_state_dict(saved['model'])
    model.eval()
    net = FeaturesAndLogits(model)

    targets = [(train_cache, s, True) for s in (all_train if args.single else held_out)]
    if args.predict_test or args.single:
        test_cache = fcache.FeatureCache(config, 'test', args.data_dir, args.cache_dir)
        test_cache.ensure_preprocessed(test_cache.subject_ids)
        targets += [(test_cache, s, False) for s in test_cache.subject_ids]
    if args.smoke:
        targets = targets[:1]

    dice = {}
    dice_kind = 'seen' if args.single else 'held-out'  # --single: the model was trained on these subjects
    for cache, subject, has_labels in targets:
        subject_dir = os.path.join(cache.root, subject)
        if args.single:
            outputs = [(deep.stored_path(subject_dir, NETWORK_SINGLE), 0, NETWORK_SINGLE)]
        else:
            outputs = [(deep.stored_path(subject_dir, NETWORK), 0, NETWORK),
                       (deep.stored_path(subject_dir, NETWORK_PROB), 1, NETWORK_PROB)]
        if all(os.path.exists(p) for p, _, _ in outputs) and not args.overwrite and not args.smoke:
            continue
        volume, _, label = load_subject(cache, subject, labels=has_labels)
        t0 = timeit.default_timer()
        maps, probabilities, prediction = predict(net, volume, args)
        if has_labels:  # only training subjects; the test labels are never read here
            dice[subject] = dice_per_label(prediction, label)
        print(f'  {subject}: features {maps.shape}, {timeit.default_timer() - t0:.0f} s'
              + (f', {dice_kind} Dice {dice[subject]}' if has_labels else ''), flush=True)
        if args.smoke:
            continue
        os.makedirs(os.path.dirname(outputs[0][0]), exist_ok=True)
        for file, which, network in outputs:
            values = (maps, probabilities)[which]
            tmp = file + f'.tmp{os.getpid()}.npz'
            np.savez(tmp, maps=values, shape=np.array(volume.shape[1:]), network=network,
                     fit_subjects=np.array(saved['fit_subjects']))
            os.replace(tmp, file)

    key = dice_kind.replace('-', '_')
    info = {'network': NETWORK_SINGLE if args.single else NETWORK, 'fold': args.fold,
            'fit_subjects': len(train_ids), 'model_file': model_path,
            'settings': saved['settings'], 'train_seconds': round(saved['seconds']),
            'last_loss': saved['history'][-1] if saved['history'] else None, f'{key}_dice': dice,
            'device': torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu',
            'seconds': round(timeit.default_timer() - start)}
    if dice:
        info[f'{key}_mean_dice'] = {k: round(float(np.mean([d[k] for d in dice.values()])), 4)
                                    for k in range(1, N_LABELS)}
    info_name = f'fold{args.fold}_single_info.json' if args.single else f'fold{args.fold}_info.json'
    with open(os.path.join(model_dir, info_name), 'w') as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains the U-Net of one fold and stores its features')
    parser.add_argument('--fold', choices=('0', '1', '2', '3', '4', 'all'), required=True)
    parser.add_argument('--predict_test', action='store_true',
                        help='With --fold all: also store the features of the test subjects.')
    parser.add_argument('--single', action='store_true',
                        help='With --fold all: store the hidden layer of this one model for all training and test '
                             'subjects as unet_single (final test only).')
    parser.add_argument('--config', type=str, default=os.path.join(paths.REPO_DIR, 'configs', 'default.toml'))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patch', type=int, default=96)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--ckpt_every', type=int, default=500)
    parser.add_argument('--smoke', action='store_true',
                        help='Timing test: one held-out subject, nothing stored in the cache.')
    parser.add_argument('--out_dir', type=str, default='',
                        help='Folder for the models (default: <cache>/<prep>/unet). Use a separate one with --smoke.')
    parser.add_argument('--overwrite', action='store_true', help='Recompute stored features.')
    parser.add_argument('--data_dir', type=str, default=paths.DATA_DIR)
    parser.add_argument('--cache_dir', type=str, default=paths.CACHE_DIR)
    main(parser.parse_args())
