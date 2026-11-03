import pandas as pd
import torch
import torch.nn.functional as F

from .config import CLASSES, MIN_CONF, NMS_RATIO, OUT_STRIDE, RADIUS, TILE, TILE_STRIDE, VOXEL
from .metric import apply_thresholds, score, tune_thresholds


def tile_starts(size, tile, stride):
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] + tile < size:
        starts.append(size - tile)
    return starts


@torch.no_grad()
def predict_volume(net, vol, tta):
    net.eval()
    d, h, w = vol.shape
    pad = [max(0, t - s) for t, s in zip(TILE, (d, h, w))]
    pad = [p + (s + p) % OUT_STRIDE for p, s in zip(pad, (d, h, w))]
    x = vol[None, None].float()
    if any(pad):
        x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]), mode='reflect')
    shape = x.shape[2:]
    acc = torch.zeros(len(CLASSES), *[s // OUT_STRIDE for s in shape], device=vol.device)
    cnt = torch.zeros(1, *acc.shape[1:], device=vol.device)
    starts = [tile_starts(s, t, st) for s, t, st in zip(shape, TILE, TILE_STRIDE)]
    for z0 in starts[0]:
        for y0 in starts[1]:
            for x0 in starts[2]:
                tile = x[:, :, z0:z0 + TILE[0], y0:y0 + TILE[1], x0:x0 + TILE[2]]
                batch = torch.cat([tile.flip(f) if f else tile for f in tta])
                with torch.autocast(vol.device.type, dtype=torch.float16, enabled=vol.device.type == 'cuda'):
                    out = net(batch)[0]
                out = torch.sigmoid(out.float())
                pred = sum(o.flip(f) if f else o for o, f in zip(out.split(1), tta)) / len(tta)
                sl = (slice(None),) + tuple(slice(a // OUT_STRIDE, (a + t) // OUT_STRIDE) for a, t in zip((z0, y0, x0), TILE))
                acc[sl] += pred[0]
                cnt[sl] += 1
    heat = acc / cnt
    return heat[:, :(d + OUT_STRIDE - 1) // OUT_STRIDE, :(h + OUT_STRIDE - 1) // OUT_STRIDE, :(w + OUT_STRIDE - 1) // OUT_STRIDE]


def extract_peaks(heat, exp):
    rows = []
    for i, c in enumerate(CLASSES):
        k = max(1, round(NMS_RATIO * RADIUS[c] / (VOXEL * OUT_STRIDE)))
        h = heat[i][None, None]
        keep = (h == F.max_pool3d(h, 2 * k + 1, stride=1, padding=k)) & (h > MIN_CONF)
        idx = keep[0, 0].nonzero()
        conf = h[0, 0][keep[0, 0]]
        zyx = (idx.float() * OUT_STRIDE + (OUT_STRIDE - 1) / 2) * VOXEL
        for (z, y, x), s in zip(zyx.tolist(), conf.tolist()):
            rows.append((exp, c, x, y, z, s))
    return pd.DataFrame(rows, columns=['experiment', 'particle_type', 'x', 'y', 'z', 'conf'])


def evaluate_tomogram(net, vol, gt, exp, tta, thresholds=None):
    pred = extract_peaks(predict_volume(net, vol, tta), exp)
    thresholds = thresholds or tune_thresholds(gt, pred)
    total, per_class = score(gt, apply_thresholds(pred, thresholds))
    return total, per_class, thresholds
