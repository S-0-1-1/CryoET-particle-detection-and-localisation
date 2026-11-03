import copy
import json
import math

import torch


class EMA:
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        self.n = 0
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m, alpha=1 - d)
            else:
                e.copy_(m)


def warmup_cosine(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parallel(model):
    return torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model


def load_checkpoint(path):
    if path.startswith(('http://', 'https://')):
        return torch.hub.load_state_dict_from_url(path, map_location='cpu')
    return torch.load(path, map_location='cpu')


def load_thresholds(path, ckpt):
    if path:
        with open(path) as f:
            return json.load(f)
    if ckpt.get('thresholds') is None:
        raise ValueError('the checkpoint has no thresholds, pass --thresholds')
    return ckpt['thresholds']
