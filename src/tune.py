# Hyperparameter tuning with Optuna


def subsample_per_aircraft(obs, n_per_aircraft, seed):
    # seeded shuffle then keep n observations of every aircraft
    raise NotImplementedError


def search_space(trial):
    # return a dict of lgbm params suggested by the trial
    raise NotImplementedError


def tune(X_train, y_train, X_val, y_val, cfg, target, n_classes, study_name=None):
    # run one study: TPESampler(seed) + MedianPruner, early stopping per trial, optionally report val loss to the trial every N rounds for pruning
    # return the study
    raise NotImplementedError


def tune_all(splits, spec_df, cfg, targets=None, scenarios=None, out_path=None):
    # loop targets x scenarios, build X/y with train.make_xy, call tune(), save with save_best()
    raise NotImplementedError


def save_best(studies, path):
    # write {key: best_params} to yaml
    raise NotImplementedError


def load_best(path):
    # read the yaml back as {key: params}
    # return {} if the file does not exist yet
    raise NotImplementedError
