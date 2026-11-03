# Metrics, ablations and track aggregation

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def topk(proba, y, k):
    # samples whose true class is among the k most probable

    k = min(k, proba.shape[1])
    best = np.argpartition(-proba, k - 1, axis=1)[:, :k]
    return float((best == y[:, None]).any(axis=1).mean())


def score(proba, y):
    # top-1, top-5, log loss, multiclass Brier score & macro-F1
    clipped = np.clip(proba, 1e-12, 1.0)
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y)), y] = 1.0
    return {
        "top1": topk(proba, y, 1),
        "top5": topk(proba, y, 5),
        "log_loss": float(-np.mean(np.log(clipped[np.arange(len(y)), y]))),
        "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
        "macro_f1": float(f1_score(y, proba.argmax(axis=1), average="macro", zero_division=0)),
    }


def track_proba(proba, aircraft_ids, track_length, seed=0):
    rng = np.random.default_rng(seed)
    log_p = np.log(np.clip(proba, 1e-12, 1.0))
    out, first_index = [], []
    for aircraft in pd.unique(aircraft_ids):
        rows = np.flatnonzero(aircraft_ids == aircraft)
        rng.shuffle(rows)
        for start in range(0, len(rows) - track_length + 1, track_length):
            chunk = rows[start:start + track_length]
            combined = log_p[chunk].sum(axis=0)
            combined -= combined.max()
            p = np.exp(combined)
            out.append(p / p.sum())
            first_index.append(chunk[0])
    return np.array(out).reshape(-1, proba.shape[1]), np.array(first_index, dtype=int)


def compare(results):
    return pd.DataFrame(results).sort_values(["target", "scenario", "split", "model"]).reset_index(drop=True)
