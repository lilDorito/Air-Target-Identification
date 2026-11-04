# Restrict predictions to aircraft operated by selected countries

import numpy as np
import pandas as pd


def _operator_set(value):
    if pd.isna(value):
        return set()
    return {part.strip() for part in str(value).split(";") if part.strip()}


def operator_list(values):
    # every state that appears in an operators column, sorted (for the country picker)

    names = set()
    for value in values:
        names.update(_operator_set(value))
    return sorted(names)


def allowed_mask(classes, operators_of, selected):
    # bool array over model classes
    # true if any selected country operates the aircraft
    # types with no current operator are never allowed once a filter is on

    wanted = set(selected)
    return np.array([bool(_operator_set(operators_of.get(c)) & wanted) for c in classes])


def filter_probs(proba, classes, operators_of, selected):
    # zero out aircraft nobody in the selection operates and renormalise each row to 1

    mask = allowed_mask(classes, operators_of, selected)
    if not mask.any():
        raise ValueError(f"No aircraft are operated by {sorted(selected)}")
    filtered = np.where(mask, proba, 0.0)
    total = filtered.sum(axis=1, keepdims=True)
    uniform = mask / mask.sum()
    return np.where(total > 0, filtered / np.where(total > 0, total, 1.0), uniform)


def aggregate(proba_aircraft, classes, label_of, label_classes):
    # sum aircraft probabilities into a coarser label (role or type)

    index = {label: j for j, label in enumerate(label_classes)}
    projection = np.zeros((len(classes), len(label_classes)))
    for i, aircraft in enumerate(classes):
        projection[i, index[label_of[aircraft]]] = 1.0
    return proba_aircraft @ projection
