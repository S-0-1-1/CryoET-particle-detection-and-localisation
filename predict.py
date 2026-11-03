import argparse
import os
import time

import pandas as pd
import torch

from cryoet.config import TTA_FLIPS
from cryoet.data import list_experiments, read_tomogram
from cryoet.inference import extract_peaks, predict_volume
from cryoet.metric import apply_thresholds
from cryoet.model import HeatmapUNet
from cryoet.utils import get_device, load_checkpoint, load_thresholds, parallel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--out', default='submission.csv')
    parser.add_argument('--thresholds')
    parser.add_argument('--no-tta', action='store_true')
    args = parser.parse_args()

    device = get_device()
    ckpt = load_checkpoint(args.weights)
    thresholds = load_thresholds(args.thresholds, ckpt)
    model = HeatmapUNet().to(device)
    model.load_state_dict(ckpt['model'])
    net = parallel(model)
    tta = [()] if args.no_tta else TTA_FLIPS

    experiments = list_experiments(args.root, args.split)
    parts, t0 = [], time.time()
    for i, exp in enumerate(experiments, 1):
        vol = torch.from_numpy(read_tomogram(args.root, args.split, exp)).to(device)
        parts.append(apply_thresholds(extract_peaks(predict_volume(net, vol, tta), exp), thresholds))
        print(f'{i}/{len(experiments)} {exp} {len(parts[-1])} particles, {(time.time() - t0) / 60:.1f} min', flush=True)

    sub = pd.concat(parts)[['experiment', 'particle_type', 'x', 'y', 'z']].reset_index(drop=True)
    sub.insert(0, 'id', range(len(sub)))
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f'saved {args.out} with {len(sub)} rows')


if __name__ == '__main__':
    main()
