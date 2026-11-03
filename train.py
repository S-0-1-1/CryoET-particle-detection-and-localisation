import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cryoet.config import CLASS_LOSS_W, VERSIONS
from cryoet.data import CropSampler, load_prepared
from cryoet.inference import evaluate_tomogram
from cryoet.loss import heatmap_focal_loss
from cryoet.model import HeatmapUNet
from cryoet.utils import EMA, get_device, parallel, warmup_cosine


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prep', default='data/prep')
    parser.add_argument('--out', default='runs')
    parser.add_argument('--val-exp', default='TS_5_4')
    parser.add_argument('--steps', type=int, default=8000)
    parser.add_argument('--warmup', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--version-p', type=float, nargs=4, default=[0.5, 0.2, 0.2, 0.1])
    parser.add_argument('--aux-version', default='isonetcorrected')
    parser.add_argument('--aux-weight', type=float, default=0.2)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = get_device()

    used = sorted({v for v, p in zip(VERSIONS, args.version_p) if p > 0} | {'denoised', args.aux_version})
    experiments, points, volumes = load_prepared(args.prep, used, device)
    train_exps = [e for e in experiments if e != args.val_exp]
    val_gt = points[points.experiment == args.val_exp]
    sampler = CropSampler(volumes, points, train_exps, args.version_p, args.aux_version, device)

    model = HeatmapUNet().to(device)
    net = parallel(model)
    ema = EMA(model, args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
    class_w = torch.tensor(CLASS_LOSS_W, device=device)
    rng = np.random.default_rng(args.seed)

    best, run_loss, t0 = -1.0, 0.0, time.time()
    model.train()
    for step in range(1, args.steps + 1):
        lr = args.lr * warmup_cosine(step - 1, args.warmup, args.steps)
        for group in opt.param_groups:
            group['lr'] = lr
        x, y, a = sampler.batch(rng, args.batch)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            logits, aux = net(x)
        loss = heatmap_focal_loss(logits, y, class_w) + args.aux_weight * F.l1_loss(aux.float(), a)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        ema.update(model)
        run_loss += loss.detach()

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f'step {step} loss {run_loss.item() / 100:.4f} lr {lr:.2e} elapsed {elapsed / 60:.1f} min, left {elapsed / step * (args.steps - step) / 60:.1f} min', flush=True)
            run_loss = 0.0

        if len(val_gt) and step % args.eval_every == 0:
            val, per_class, thresholds = evaluate_tomogram(ema.module, volumes[(args.val_exp, 'denoised')], val_gt, args.val_exp, [()])
            print(f'step {step} val fbeta {val:.4f}', {c: round(s, 3) for c, s in per_class.items()}, flush=True)
            if val > best:
                best = val
                torch.save({'model': ema.module.state_dict(), 'thresholds': thresholds, 'score': val, 'step': step}, os.path.join(args.out, 'best.pt'))

    torch.save({'model': ema.module.state_dict(), 'thresholds': None, 'score': None, 'step': args.steps}, os.path.join(args.out, 'last.pt'))
    print(f'saved checkpoints to {args.out}' + (f', best val fbeta {best:.4f}' if best >= 0 else ''))


if __name__ == '__main__':
    main()
