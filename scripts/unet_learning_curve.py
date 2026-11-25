"""Learning curve of the U-Net of one CV fold: training vs held-out loss and Dice during training.

The U-Net of ``train_unet_features.py`` only logged its training loss, so it can not show whether the network
overfits during training. This script trains the same U-Net again (same network, loss, optimiser, learning-rate
schedule, sampling, augmentation and seed as ``train_unet_features.py --fold <k>``) and evaluates it during
training on the training subjects and on the held-out subjects of the fold:

- every ``--loss_every`` steps: the loss (Dice + cross-entropy of the main output, eval mode) on a fixed set of
  random 96^3 blocks, drawn once with a separate seed, from the training subjects and from the held-out subjects;
- every ``--dice_every`` steps: the mean Dice of the 5 tissues on whole volumes (sliding window, as in
  ``train_unet_features.py``) of ``--n_train_dice`` training subjects and of all held-out subjects.

Overfitting shows as a training curve that keeps improving while the held-out curve stops or gets worse.
Nothing is written to the feature cache; the curve goes to ``--out_dir`` as ``fold<k>_curve.json``.
The test subjects are never read.
"""
import argparse
import json
import os
import sys
import timeit

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mialab.utilities.config as cfg  # noqa: E402
import mialab.utilities.feature_cache as fcache  # noqa: E402
import mialab.utilities.paths as paths  # noqa: E402
from train_unet_features import (N_LABELS, PatchSampler, autocast, build_model, dice_per_label,  # noqa: E402
                                 load_subject, supervised_loss)

EVAL_SEED_OFFSET = 1000  # the fixed evaluation blocks use their own generator, the training draws stay unchanged


@torch.inference_mode()
def block_loss(model, loss_fn, blocks, args) -> float:
    """Mean loss of the main output over fixed (x, y) blocks, in batches of ``--batch_size``."""
    x_all, y_all = blocks
    losses = []
    for i in range(0, len(x_all), args.batch_size):
        x, y = x_all[i:i + args.batch_size].to(args.device), y_all[i:i + args.batch_size].to(args.device)
        with autocast(args.device):
            logits = model(x)  # eval mode: main output only
        losses.append(loss_fn(logits.float(), y).item())
    return float(np.mean(losses))


@torch.inference_mode()
def volume_dice(model, subjects, args) -> dict:
    """Dice per tissue (1-5) of whole volumes, averaged over the subjects."""
    dice = []
    for volume, label in subjects:
        x = torch.from_numpy(volume)[None].to(args.device)
        with autocast(args.device):
            logits = sliding_window_inference(x, (args.patch,) * 3, sw_batch_size=2, predictor=model,
                                              overlap=0.5, mode='gaussian')
        dice.append(dice_per_label(logits[0].float().argmax(0).cpu().numpy(), label))
    per_label = {k: round(float(np.mean([d[k] for d in dice])), 4) for k in range(1, N_LABELS)}
    return {'mean': round(float(np.mean(list(per_label.values()))), 4), 'per_label': per_label}


def main(args):
    config = cfg.load_config(args.config)
    if args.loss_every < 1 or args.dice_every % args.loss_every != 0:
        raise ValueError('--dice_every must be a multiple of --loss_every (the Dice is computed at loss points)')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('No GPU found; pass --device cpu or run on a GPU node')
    with open(os.path.join(paths.REPO_DIR, 'configs', 'folds.json')) as f:
        folds = json.load(f)['folds']
    train_cache = fcache.FeatureCache(config, 'train', args.data_dir, args.cache_dir)
    all_train = list(train_cache.subject_ids)
    held_out = folds[int(args.fold)]
    train_ids = [s for s in all_train if s not in held_out]
    if sorted(train_ids + held_out) != sorted(all_train):
        raise ValueError('configs/folds.json does not match the training subjects')
    train_cache.ensure_preprocessed(all_train)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'fold{args.fold}_curve.json')
    ckpt_path = os.path.join(args.out_dir, f'fold{args.fold}_curve.ckpt')

    seed = train_cache.config['seed'] + int(args.fold) + 1  # as train_unet_features.py
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    print(f'Loading {len(train_ids)} training and {len(held_out)} held-out subjects...', flush=True)
    train_data = [load_subject(train_cache, s, labels=True) for s in train_ids]
    held_data = [load_subject(train_cache, s, labels=True) for s in held_out]
    train_dice_subjects = [(v, lab) for v, _, lab in train_data[:args.n_train_dice]]
    held_dice_subjects = [(v, lab) for v, _, lab in held_data]
    sampler = PatchSampler(*(list(d) for d in zip(*train_data)), patch=args.patch)
    held_sampler = PatchSampler(*(list(d) for d in zip(*held_data)), patch=args.patch)
    del train_data, held_data
    eval_rng = np.random.default_rng(seed + EVAL_SEED_OFFSET)
    train_blocks = sampler.sample(eval_rng, args.n_blocks)
    held_blocks = held_sampler.sample(eval_rng, args.n_blocks)
    del held_sampler

    model = build_model().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.steps, power=0.9)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    step, curve, train_log, seconds = 0, [], [], 0.0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        rng.bit_generator.state = ckpt['rng']
        torch.set_rng_state(ckpt['torch_rng'])
        step, curve, train_log, seconds = ckpt['step'], ckpt['curve'], ckpt['train_log'], ckpt['seconds']
        print(f'Resumed from step {step}', flush=True)
    start = timeit.default_timer()

    def info():
        return {'fold': args.fold, 'fit_subjects': train_ids, 'held_out_subjects': held_out,
                'train_dice_subjects': train_ids[:args.n_train_dice], 'n_blocks': args.n_blocks,
                'settings': {k: getattr(args, k) for k in ('steps', 'batch_size', 'lr', 'patch', 'loss_every',
                                                           'dice_every')},
                'seconds': round(seconds + timeit.default_timer() - start),
                'device': torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu',
                'curve': curve, 'train_log': train_log}

    def save_ckpt():
        tmp = ckpt_path + f'.tmp{os.getpid()}'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'rng': rng.bit_generator.state,
                    'torch_rng': torch.get_rng_state(), 'step': step, 'curve': curve, 'train_log': train_log,
                    'seconds': seconds + timeit.default_timer() - start}, tmp)
        os.replace(tmp, ckpt_path)

    def evaluate():
        model.eval()
        point = {'step': step, 'lr': optimizer.param_groups[0]['lr'],
                 'train_loss': round(block_loss(model, loss_fn, train_blocks, args), 5),
                 'held_out_loss': round(block_loss(model, loss_fn, held_blocks, args), 5)}
        if step > 0 and (step % args.dice_every == 0 or step == args.steps):
            point['train_dice'] = volume_dice(model, train_dice_subjects, args)
            point['held_out_dice'] = volume_dice(model, held_dice_subjects, args)
        model.train()
        curve.append(point)
        text = f'  step {step}: loss train {point["train_loss"]:.4f} held-out {point["held_out_loss"]:.4f}'
        if 'train_dice' in point:
            text += f', Dice train {point["train_dice"]["mean"]:.4f} held-out {point["held_out_dice"]["mean"]:.4f}'
        print(text + f' ({seconds + timeit.default_timer() - start:.0f} s)', flush=True)

    if step == 0:
        evaluate()
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
        if step % 50 == 0 or step == args.steps:  # same running training loss as train_unet_features.py
            train_log.append({'step': step, 'loss': round(float(np.mean(running)), 5)})
            running = []
        if step % args.loss_every == 0 or step == args.steps:
            evaluate()
            if step < args.steps:
                save_ckpt()  # after the evaluation, so a resumed job does not repeat or miss a point
    with open(out_path + '.tmp', 'w') as f:
        json.dump(info(), f, indent=2)
    os.replace(out_path + '.tmp', out_path)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f'Curve saved to {out_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training vs held-out learning curve of the U-Net of one fold')
    parser.add_argument('--fold', choices=('0', '1', '2', '3', '4'), required=True)
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Folder for the curve (never the feature cache).')
    parser.add_argument('--config', type=str, default=os.path.join(paths.REPO_DIR, 'configs', 'default.toml'))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patch', type=int, default=96)
    parser.add_argument('--loss_every', type=int, default=250)
    parser.add_argument('--dice_every', type=int, default=500)
    parser.add_argument('--n_blocks', type=int, default=32, help='Fixed evaluation blocks per subject group.')
    parser.add_argument('--n_train_dice', type=int, default=4, help='Training subjects for the volume Dice.')
    parser.add_argument('--data_dir', type=str, default=paths.DATA_DIR)
    parser.add_argument('--cache_dir', type=str, default=paths.CACHE_DIR)
    main(parser.parse_args())
