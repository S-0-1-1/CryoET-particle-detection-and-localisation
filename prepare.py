import argparse
import os

import numpy as np
import pandas as pd

from cryoet.config import VERSIONS
from cryoet.data import list_experiments, read_points, read_tomogram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', default='data/prep')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    points = []
    for exp in list_experiments(args.root, 'train'):
        for version in VERSIONS:
            vol = read_tomogram(args.root, 'train', exp, version)
            np.save(os.path.join(args.out, f'{exp}_{version}.npy'), vol.astype(np.float16))
        points.append(read_points(args.root, exp))
        print(f'{exp} {vol.shape}', flush=True)
    points = pd.concat(points, ignore_index=True)
    points.to_csv(os.path.join(args.out, 'points.csv'), index=False)
    print(f'saved {len(points)} particles to {args.out}')


if __name__ == '__main__':
    main()
