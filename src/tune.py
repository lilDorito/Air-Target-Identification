# Hyperparameter tuning with Optuna

from pathlib import Path
import lightgbm as lgb
import numpy as np
import optuna
import yaml
from . import train

def subsample_per_aircraft(obs, n_per_aircraft, seed):
    # seeded shuffle then keep n observations of every aircraft

    rng = np.random.default_rng(seed)
    selected = []

    for _, group in obs.groupby("aircraft_id", sort=False):
        indices = group.index.to_numpy()
        chosen = rng.permutation(indices)[:n_per_aircraft]
        selected.extend(chosen)

    return obs.loc[selected].reset_index(drop=True)


def search_space(trial):
    # return a dict of lgbm params suggested by the trial

    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.2, log=True
        ),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_data_in_leaf": trial.suggest_int(
            "min_data_in_leaf", 10, 100
        ),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.6, 1.0
        ),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.6, 1.0
        ),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
        "lambda_l2": trial.suggest_float(
            "lambda_l2", 0.5, 50.0, log=True
        ),
        "max_delta_step": trial.suggest_float(
            "max_delta_step", 0.3, 5.0, log=True
        ),
        "max_bin": trial.suggest_categorical(
            "max_bin", [63, 127, 255, 511]
        ),
    }


def tune(X_train, y_train, X_val, y_val, cfg, target, n_classes, study_name=None):
    # run one study: TPESampler(seed) + MedianPruner, early stopping per trial, optionally report val loss to the trial every N rounds for pruning
    # return the study

    tuning_cfg = cfg["tuning"]
    lgb_cfg = cfg["lightgbm"]

    sampler = optuna.samplers.TPESampler(seed=tuning_cfg["seed"])
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=tuning_cfg["pruning_warmup"])

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial):
        overrides = search_space(trial)
        overrides["feature_pre_filter"] = False
        overrides["metric"] = "multi_logloss"

        params = train.lgb_params(cfg, n_classes, overrides)

        weights = (
            train.balanced_weights(y_train, n_classes)
            if target in lgb_cfg.get("class_weight_for", [])
            else None
        )

        # rebuild both datasets for every trial:
        # max_bin and min_data_in_leaf affect dataset construction
        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            weight=weights,
            params=params,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_val,
            label=y_val,
            reference=dtrain,
            params=params,
            free_raw_data=False,
        )

        report_every = tuning_cfg["report_every"]

        def pruning_callback(env):
            step = env.iteration + 1

            if step % report_every != 0:
                return

            for result in env.evaluation_result_list or []:
                dataset, metric, value = result[:3]

                if dataset == "val" and metric == "multi_logloss":
                    trial.report(value, step)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

        pruning_callback.order = 20
        pruning_callback.before_iteration = False

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=lgb_cfg["n_estimators"],
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(lgb_cfg["early_stopping_rounds"], verbose=False), pruning_callback],
        )

        return booster.best_score["val"]["multi_logloss"]

    study.optimize(
        objective,
        n_trials=tuning_cfg["n_trials"],
        timeout=tuning_cfg.get("timeout_s"),
    )

    return study


def tune_all(splits, spec_df, cfg, targets=None, scenarios=None, out_path=None):
    # loop targets x scenarios, build X/y with train.make_xy, call tune(), save with save_best()

    targets = targets or cfg["targets"]
    scenarios = scenarios or list(cfg["scenarios"])

    tuning_cfg = cfg["tuning"]

    train_obs = subsample_per_aircraft(splits["train"], tuning_cfg["samples_per_aircraft_train"], tuning_cfg["seed"])
    val_obs = splits["val"]

    studies = {}

    for target in targets:
        classes = train.class_list(spec_df, target)
        n_classes = len(classes)

        for scenario in scenarios:
            X_train, y_train = train.make_xy(train_obs, cfg, scenario, target, classes)
            X_val, y_val = train.make_xy(val_obs, cfg, scenario, target, classes)

            key = f"{train.TARGET_ALIAS[target]}_{scenario}"

            print(f"Tuning {key}...")

            study = tune(
                X_train, y_train,
                X_val, y_val,
                cfg, target, n_classes,
                study_name=key,
            )

            studies[key] = study

            print(f"{key}: best validation log loss = "
                  f"{study.best_value:.4f}"
            )

    path = out_path or tuning_cfg["best_params_path"]
    save_best(studies, path)

    return studies


def save_best(studies, path):
    # write {key: best_params} to yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    best_params = {
        key: study.best_params
        for key, study in studies.items()
        if study.best_trial is not None
    }

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            best_params,
            file,
            sort_keys=False,
            allow_unicode=True,
        )


def load_best(path):
    # read the yaml back as {key: params}
    # return {} if the file does not exist yet

    path = Path(path)

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
