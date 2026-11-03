# End-to-end experiment: sample -> (tune) -> train -> calibrate -> evaluate -> filter check

import numpy as np
import pandas as pd
from . import calibrate, country_filter, evaluate, load, sampling, train


def blank_estimated(spec_df):
    mask = load.estimated_mask(spec_df)
    out = spec_df.copy()
    cols = list(mask.columns)
    out[cols] = out[cols].mask(mask[cols])
    return out


def _tuned_for(tuned_params, target, scenario):
    return (tuned_params or {}).get(f"{train.TARGET_ALIAS[target]}_{scenario}")


def run(spec_df, cfg, targets=None, scenarios=None, ablation=False, tuned_params=None, model_dir=None, splits=None, with_baselines=True):
    # train and evaluate every requested target x scenario

    targets = targets or cfg["targets"]
    scenarios = scenarios or list(cfg["scenarios"])
    spec = blank_estimated(spec_df) if ablation else spec_df
    splits = splits or sampling.make_splits(spec, cfg)
    tag = "ablation" if ablation else "full"

    rows, artefacts = [], {}
    for target in targets:
        classes = train.class_list(spec_df, target)
        n_classes = len(classes)
        for scenario in scenarios:
            xy = {name: train.make_xy(obs, cfg, scenario, target, classes) for name, obs in splits.items()}
            X_tr, y_tr = xy["train"]
            X_va, y_va = xy["val"]

            booster = train.train_one(X_tr, y_tr, X_va, y_va, cfg, target, n_classes, params=_tuned_for(tuned_params, target, scenario))
            if model_dir:
                train.save_model(booster, model_dir, target, scenario, suffix="" if not ablation else "_ablation")

            calibrator = calibrate.fit_calibrator(train.predict_proba(booster, X_va), y_va, cfg.get("calibration", "temperature"))

            def add(model, split, proba, y, extra=None):
                rows.append({"target": target, "scenario": scenario, "model": model, "split": split, "data": tag, **evaluate.score(proba, y), **(extra or {})})

            for split in ("test", "shifted_test"):
                X, y = xy[split]
                raw = train.predict_proba(booster, X)
                cal = calibrator.transform(raw)
                add("lightgbm_raw", split, raw, y)
                add("lightgbm", split, cal, y, {"temperature": calibrator.temperature, "ece": calibrate.reliability_curve(cal, y)["ece"], "best_iteration": booster.best_iteration})
                if split == "test":
                    artefacts[(target, scenario)] = {"proba": cal, "y": y, "classes": classes, "aircraft_id": splits[split]["aircraft_id"].to_numpy()}

            if with_baselines:
                X_te, y_te = xy["test"]
                add("prior", "test", train.prior_baseline(y_tr, n_classes, len(y_te)), y_te)
                add("knn", "test", train.knn_baseline(X_tr, y_tr, X_te, n_classes), y_te)

            # aircraft model only
            if target == "name":
                proba, y = artefacts[(target, scenario)]["proba"], artefacts[(target, scenario)]["y"]
                for coarse in ("role", "type"):
                    coarse_classes = train.class_list(spec_df, coarse)
                    label_of = dict(zip(spec_df["name"], spec_df[coarse]))
                    summed = country_filter.aggregate(proba, classes, label_of, coarse_classes)
                    idx = {c: i for i, c in enumerate(coarse_classes)}
                    y_coarse = np.array([idx[label_of[classes[i]]] for i in y])
                    rows.append({"target": coarse, "scenario": scenario, "model": "aircraft_summed", "split": "test", "data": tag, **evaluate.score(summed, y_coarse)})
                ids = artefacts[(target, scenario)]["aircraft_id"]
                for k in cfg.get("evaluation", {}).get("track_lengths", [1]):
                    if k == 1:
                        continue
                    tp, first = evaluate.track_proba(proba, ids, k, seed=cfg["seeds"]["test"])
                    if len(first) == 0:
                        continue
                    add(f"lightgbm_track{k}", "test", tp, y[first])

    return evaluate.compare(rows), artefacts


def filter_check(artefact, spec_df, selected, country_groups=None):
    # country filter on the aircraft model

    classes, proba, y = artefact["classes"], artefact["proba"], artefact["y"]
    country_of = dict(zip(spec_df["name"], spec_df["country"]))
    allowed = country_filter.allowed_mask(classes, country_of, selected, country_groups)
    keep = allowed[y]
    before = evaluate.score(proba[keep], y[keep])
    after = evaluate.score(country_filter.filter_probs(proba[keep], classes, country_of, selected, country_groups), y[keep])
    return pd.DataFrame([{"selection": "+".join(selected), "filter": "off", "n_classes": len(classes), **before}, {"selection": "+".join(selected), "filter": "on", "n_classes": int(allowed.sum()), **after}])


def unseen_families(spec_df, cfg, scenario, target, num_boost_round=300, params=None):
    # role/type on held-out families

    if target == "name":
        raise ValueError("The aircraft model cannot name unseen aircraft, use role or type")
    classes = train.class_list(spec_df, target)
    rows = []
    for fold, parts in sampling.family_holdout_splits(spec_df, cfg).items():
        X_tr, y_tr = train.make_xy(parts["train"], cfg, scenario, target, classes)
        X_te, y_te = train.make_xy(parts["test"], cfg, scenario, target, classes)
        booster = train.train_one(X_tr, y_tr, None, None, cfg, target, len(classes), params=params, num_boost_round=num_boost_round)
        rows.append({"fold": fold, "target": target, "scenario": scenario, **evaluate.score(train.predict_proba(booster, X_te), y_te)})
    return pd.DataFrame(rows)
