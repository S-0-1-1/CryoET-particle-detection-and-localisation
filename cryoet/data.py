import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import zarr

from .config import CLASS_ID, CLASS_LOSS_W, CLASSES, CROP, FG_PROB, MARGIN, OUT_STRIDE, RADIUS, SIGMA_RATIO, VERSIONS, VOXEL


def list_experiments(root, split):
    return sorted(os.listdir(os.path.join(root, split, 'static', 'ExperimentRuns')))


def read_tomogram(root, split, exp, version='denoised'):
    path = os.path.join(root, split, 'static', 'ExperimentRuns', exp, 'VoxelSpacing10.000', f'{version}.zarr')
    vol = zarr.open(path, mode='r')['0'][:].astype(np.float32)
    return (vol - vol.mean()) / (vol.std() + 1e-6)


def read_points(root, exp):
    rows = []
    for c in CLASSES:
        path = os.path.join(root, 'train', 'overlay', 'ExperimentRuns', exp, 'Picks', f'{c}.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for p in json.load(f)['points']:
                loc = p['location']
                rows.append((exp, c, loc['x'], loc['y'], loc['z']))
    return pd.DataFrame(rows, columns=['experiment', 'particle_type', 'x', 'y', 'z'])


def load_prepared(prep, versions, device):
    experiments = sorted(f[:-len('_denoised.npy')] for f in os.listdir(prep) if f.endswith('_denoised.npy'))
    points = pd.read_csv(os.path.join(prep, 'points.csv'))
    volumes = {(e, v): torch.from_numpy(np.load(os.path.join(prep, f'{e}_{v}.npy'))).to(device) for e in experiments for v in versions}
    return experiments, points, volumes


class Picks:
    def __init__(self, df, device):
        self.cls = df['particle_type'].map(CLASS_ID).values
        self.zyx = df[['z', 'y', 'x']].values / VOXEL
        w = np.array([CLASS_LOSS_W[k] for k in self.cls], dtype=np.float64)
        self.p = w / w.sum() if len(w) else w
        self.zyx_t = torch.tensor(self.zyx, dtype=torch.float32, device=device)
        self.cls_t = torch.tensor(self.cls, dtype=torch.long, device=device)


class CropSampler:
    def __init__(self, volumes, points, experiments, version_p, aux_version, device):
        self.volumes = volumes
        self.experiments = experiments
        self.version_p = version_p
        self.aux_version = aux_version
        self.device = device
        self.picks = {e: Picks(points[points.experiment == e], device) for e in experiments}
        self.n_out = CROP // OUT_STRIDE
        self.sigma = torch.tensor([SIGMA_RATIO * RADIUS[c] / (VOXEL * OUT_STRIDE) for c in CLASSES], device=device)
        self.pad = float(3 * self.sigma.max() * OUT_STRIDE)
        axis = torch.arange(self.n_out, device=device, dtype=torch.float32)
        self.grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)

    def transform(self, local, flips, k):
        for ax in flips:
            local[:, ax] = (CROP - 1) - local[:, ax]
        for _ in range(k):
            local = torch.stack([local[:, 0], (CROP - 1) - local[:, 2], local[:, 1]], 1)
        return local

    def rasterize(self, local, cls):
        heat = torch.zeros(len(CLASSES), self.n_out ** 3, device=self.device)
        if len(cls):
            d2 = ((self.grid[None] - local.round()[:, None]) ** 2).sum(-1)
            g = torch.exp(-d2 / (2 * self.sigma[cls, None] ** 2))
            heat = heat.scatter_reduce(0, cls[:, None].expand_as(g), g, reduce='amax')
        return heat.view(len(CLASSES), self.n_out, self.n_out, self.n_out)

    def sample(self, rng):
        exp = self.experiments[rng.integers(len(self.experiments))]
        version = VERSIONS[rng.choice(len(VERSIONS), p=self.version_p)]
        vol, aux, pts = self.volumes[(exp, version)], self.volumes[(exp, self.aux_version)], self.picks[exp]
        shape = np.array(vol.shape)
        if len(pts.cls) and rng.random() < FG_PROB:
            i = rng.choice(len(pts.cls), p=pts.p)
            origin = np.round(pts.zyx[i]).astype(int) - rng.integers(MARGIN, CROP - MARGIN, 3)
        else:
            origin = rng.integers(0, shape - CROP + 1)
        origin = np.clip(origin, 0, shape - CROP)
        z0, y0, x0 = origin
        x = torch.stack([vol[z0:z0 + CROP, y0:y0 + CROP, x0:x0 + CROP], aux[z0:z0 + CROP, y0:y0 + CROP, x0:x0 + CROP]]).float()
        local = pts.zyx_t - torch.tensor(origin, dtype=torch.float32, device=self.device)
        keep = ((local > -self.pad) & (local < CROP + self.pad)).all(1)
        local, cls = local[keep], pts.cls_t[keep]
        flips = [ax for ax in range(3) if rng.random() < 0.5]
        k = int(rng.integers(4))
        if flips:
            x = x.flip([ax + 1 for ax in flips])
        if k:
            x = torch.rot90(x, k, dims=(2, 3))
        local = self.transform(local, flips, k)
        x, aux = x[0], F.avg_pool3d(x[1][None], OUT_STRIDE)
        x = x * float(rng.uniform(0.8, 1.2)) + float(rng.uniform(-0.2, 0.2)) + torch.randn_like(x) * float(rng.uniform(0, 0.1))
        heat = self.rasterize((local - (OUT_STRIDE - 1) / 2) / OUT_STRIDE, cls)
        return x, heat, aux

    def batch(self, rng, size):
        xs, ys, auxs = zip(*[self.sample(rng) for _ in range(size)])
        return torch.stack(xs)[:, None], torch.stack(ys), torch.stack(auxs)
