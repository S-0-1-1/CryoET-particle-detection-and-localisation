import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from .config import CLASSES, MIN_CONF, RADIUS, WEIGHT


def class_counts(gt, pred, c):
    tp = fp = fn = 0
    for exp in gt.experiment.unique():
        ref = gt[(gt.experiment == exp) & (gt.particle_type == c)][['x', 'y', 'z']].values
        cand = pred[(pred.experiment == exp) & (pred.particle_type == c)][['x', 'y', 'z']].values
        if len(ref) == 0 or len(cand) == 0:
            fp += len(cand)
            fn += len(ref)
            continue
        matches = KDTree(cand).query_ball_tree(KDTree(ref), r=0.5 * RADIUS[c])
        hits = len({j for m in matches for j in m})
        tp += hits
        fp += len(cand) - hits
        fn += len(ref) - hits
    return tp, fp, fn


def fbeta(tp, fp, fn, beta=4):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)


def score(gt, pred):
    per_class = {c: fbeta(*class_counts(gt, pred, c)) for c in CLASSES if WEIGHT[c] > 0}
    return sum(per_class[c] * WEIGHT[c] for c in per_class) / sum(WEIGHT.values()), per_class


def tune_thresholds(gt, pred):
    thresholds = {}
    for c in CLASSES:
        pred_c = pred[pred.particle_type == c]
        best = (-1.0, MIN_CONF)
        for t in np.arange(MIN_CONF, 0.95, 0.025):
            s = fbeta(*class_counts(gt, pred_c[pred_c.conf > t], c))
            if s > best[0]:
                best = (s, float(t))
        thresholds[c] = best[1]
    return thresholds


def apply_thresholds(pred, thresholds):
    return pd.concat([pred[(pred.particle_type == c) & (pred.conf > thresholds[c])] for c in CLASSES if WEIGHT[c] > 0])
