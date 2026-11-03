import argparse
import json
import os

import torch

from cryoet.config import TTA_FLIPS
from cryoet.data import read_points, read_tomogram
from cryoet.inference import evaluate_tomogram
from cryoet.model import HeatmapUNet
from cryoet.utils import get_device, load_checkpoint, parallel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--exp', default='TS_5_4')
    parser.add_argument('--thresholds')
    parser.add_argument('--no-tta', action='store_true')
    parser.add_argument('--export')
    args = parser.parse_args()

    device = get_device()
    ckpt = load_checkpoint(args.weights)
    model = HeatmapUNet().to(device)
    model.load_state_dict(ckpt['model'])
    thresholds = None
    if args.thresholds:
        with open(args.thresholds) as f:
            thresholds = json.load(f)

    vol = torch.from_numpy(read_tomogram(args.root, 'train', args.exp)).to(device)
    gt = read_points(args.root, args.exp)
    tta = [()] if args.no_tta else TTA_FLIPS
    val, per_class, thresholds = evaluate_tomogram(parallel(model), vol, gt, args.exp, tta, thresholds)
    print(f'{args.exp} fbeta {val:.4f}', {c: round(s, 3) for c, s in per_class.items()})
    print('thresholds', {c: round(t, 3) for c, t in thresholds.items()})

    if args.export:
        if os.path.dirname(args.export):
            os.makedirs(os.path.dirname(args.export), exist_ok=True)
        torch.save({'model': ckpt['model'], 'thresholds': thresholds, 'score': val}, args.export)
        print(f'saved {args.export}')


if __name__ == '__main__':
    main()
