import torch
import torch.nn.functional as F


def heatmap_focal_loss(logits, target, class_weights):
    logits = logits.float()
    p = torch.sigmoid(logits)
    pos = (target == 1).float()
    pos_term = -((1 - p) ** 2) * F.logsigmoid(logits) * pos
    neg_term = -(p ** 2) * F.logsigmoid(-logits) * ((1 - target) ** 4) * (1 - pos)
    per_class = (pos_term + neg_term).sum(dim=(0, 2, 3, 4))
    return (per_class * class_weights).sum() / pos.sum().clamp(min=1)
