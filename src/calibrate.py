# Calibrate predicted probabilities on the validation set

import numpy as np
from scipy.optimize import minimize_scalar


def _apply(proba, temperature):
    logits = np.log(np.clip(proba, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(proba, y):
    return -np.mean(np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0)))


class TemperatureCalibrator:
    # fit T on validation probabilities
    # apply to any later probabilities

    def __init__(self):
        self.temperature = 1.0

    def fit(self, proba_val, y_val):
        result = minimize_scalar(lambda log_t: _nll(_apply(proba_val, np.exp(log_t)), y_val), bounds=(-3.0, 3.0), method="bounded")
        self.temperature = float(np.exp(result.x))
        return self

    def transform(self, proba):
        return _apply(proba, self.temperature)


def fit_calibrator(proba_val, y_val, method="temperature"):
    if method != "temperature":
        raise ValueError("Use calibration='temperature'")
    return TemperatureCalibrator().fit(proba_val, y_val)


def reliability_curve(proba, y, n_bins=10):
    # top-label reliability
    # per confidence bin, mean confidence vs observed accuracy

    confidence = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(confidence, edges) - 1, 0, n_bins - 1)
    conf_mean, acc, counts = [], [], []
    for b in range(n_bins):
        in_bin = which == b
        counts.append(int(in_bin.sum()))
        conf_mean.append(float(confidence[in_bin].mean()) if in_bin.any() else np.nan)
        acc.append(float(correct[in_bin].mean()) if in_bin.any() else np.nan)
    counts_arr = np.array(counts)
    gaps = np.nan_to_num(np.abs(np.array(conf_mean) - np.array(acc)))
    ece = float((gaps * counts_arr).sum() / max(counts_arr.sum(), 1))
    return {"bin_centres": (edges[:-1] + edges[1:]) / 2, "confidence": conf_mean, "accuracy": acc, "counts": counts, "ece": ece}
