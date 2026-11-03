# Select scenario columns and encode them

import pandas as pd

ENGINE_TYPES = [
    'turbojet', 'turbofan', 'piston', 'turboprop', 'turboshaft', 'electric', 'none',
    'solid rocket', 'liquid rocket', 'ramjet'
    ]

IR_CATEGORIES = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "very high": 3
}

def scenario_columns(cfg, scenario_name):
    scenario = cfg["scenarios"][scenario_name]

    columns = []

    if "extends" in scenario:
        columns.extend(scenario_columns(cfg, scenario["extends"]))

    columns.extend(scenario.get("numeric", []))
    columns.extend(scenario.get("categorical", []))

    return columns


def encode(X):
    X = X.copy()

    if "afterburner" in X.columns:
        X["afterburner"] = X["afterburner"].map({"yes": 1, "no": 0})

    if "ir_category" in X.columns:
        X["ir_category"] = X["ir_category"].map(IR_CATEGORIES)

    if "engine_type" in X.columns:
        X["engine_type"] = pd.Categorical(X["engine_type"], categories=ENGINE_TYPES)

    return X


def drop_estimated(X, mask):
    X = X.copy()
    X[mask] = float("nan")
    return X