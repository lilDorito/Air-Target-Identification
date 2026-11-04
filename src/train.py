# Train LightGBM models (3 targets x 3 scenarios) and the 2 baselines

from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from . import features

TARGET_ALIAS = {"name": "aircraft", "role": "role", "type": "type"}


# DATA HELPERS

def class_list(spec_df, target):
    return sorted(spec_df[target].dropna().unique())


def make_xy(obs, cfg, scenario, target, classes):
    X = features.encode(obs[features.scenario_columns(cfg, scenario)])
    index = {c: i for i, c in enumerate(classes)}
    y = obs[target].map(index).to_numpy()
    return X, y


def balanced_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return len(y) / (np.count_nonzero(counts) * counts[y])


# LGBM

def lgb_params(cfg, n_classes, overrides=None):
    lgb_cfg = cfg["lightgbm"]
    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "seed": cfg["seeds"]["model"],
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": lgb_cfg.get("num_threads", 0),
        "verbose": -1,
    }
    params.update(lgb_cfg.get("params", {}))
    if overrides:
        params.update(overrides)
    return params


def train_one(X_train, y_train, X_val, y_val, cfg, target, n_classes, params=None, num_boost_round=None):
    lgb_cfg = cfg["lightgbm"]
    weights = balanced_weights(y_train, n_classes) if target in lgb_cfg.get("class_weight_for", []) else None
    params = lgb_params(cfg, n_classes, params)
    dtrain = lgb.Dataset(X_train, y_train, weight=weights, params=params, free_raw_data=False)

    if num_boost_round is not None:
        return lgb.train(params, dtrain, num_boost_round=num_boost_round)

    dval = lgb.Dataset(X_val, y_val, reference=dtrain, params=params)
    return lgb.train(
        params, dtrain,
        num_boost_round=lgb_cfg["n_estimators"],
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(lgb_cfg["early_stopping_rounds"], verbose=False)],
    )


def predict_proba(booster, X):
    best = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else None
    return booster.predict(X, num_iteration=best)


def save_model(booster, out_dir, target, scenario, suffix=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{TARGET_ALIAS[target]}_{scenario}{suffix}.txt"
    booster.save_model(str(path))
    return path


def save_model_meta(out_dir, target, scenario, classes, temperature, feature_columns, suffix=""):
    """Write <alias>_<scenario><suffix>.json next to the model: class labels, calibration
    temperature and input columns, so the API can serve the model without the dataset.
    Aircraft meta contains the roster (class names) -> gitignored like aircraft models."""
    import json
    path = Path(out_dir) / f"{TARGET_ALIAS[target]}_{scenario}{suffix}.json"
    path.write_text(json.dumps({"target": target, "scenario": scenario, "classes": list(classes),
                                "temperature": float(temperature), "features": list(feature_columns)},
                               indent=1), encoding="utf-8")
    return path


# BASELINES

def prior_baseline(y_train, n_classes, n_rows):
    freq = np.bincount(y_train, minlength=n_classes).astype(float)
    freq /= freq.sum()
    return np.tile(freq, (n_rows, 1))


def _numeric(X):
    X = X.copy()
    for col in X.columns:
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].cat.codes.replace(-1, np.nan)
    return X.astype(float)


def _signed_log(Z):
    return np.sign(Z) * np.log1p(np.abs(Z))


def knn_baseline(X_train, y_train, X_test, n_classes, k=25):
    # nearest neighbour on log-scaled & standardised features

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(_signed_log),
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k, weights="distance"),
    )
    model.fit(_numeric(X_train), y_train)
    proba = np.zeros((len(X_test), n_classes))
    proba[:, model.classes_] = model.predict_proba(_numeric(X_test))
    return proba
