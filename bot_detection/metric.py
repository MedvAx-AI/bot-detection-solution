"""Precision maximized over thresholds with recall >= 0.70; ties stay grouped."""
import numpy as np


def precision_at_recall(y_true, score, recall=0.70):
    """Official metric: maximize precision over grouped-score thresholds."""
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y, s = y_true[order], score[order]
    tp = np.cumsum(y)
    ends = np.r_[s[1:] != s[:-1], True]
    prec = tp[ends] / np.arange(1, len(y) + 1)[ends]
    rec = tp[ends] / y_true.sum()
    return float(prec[rec >= recall].max())
