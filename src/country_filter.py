# Restrict predictions to selected countries by masking and renormalising

import numpy as np
import pandas as pd


def _origin_set(country_value):
    if pd.isna(country_value):
        return set()
    return {part.strip() for part in str(country_value).split("/") if part.strip()}


def expand_selection(selected, country_groups=None):
    # expand user-selected names with configured groups

    expanded = set()
    groups = country_groups or {}
    for name in selected:
        expanded.add(name)
        expanded.update(groups.get(name, []))
    return expanded


def allowed_mask(classes, country_of, selected, country_groups=None):
    # bool array over model classes
    # true if the aircraft's origin matches the selection

    wanted = expand_selection(selected, country_groups)
    return np.array([bool(_origin_set(country_of.get(c)) & wanted) for c in classes])


def filter_probs(proba, classes, country_of, selected, country_groups=None):
    # zero out aircraft from other countries and renormalise each row to 1
    
    mask = allowed_mask(classes, country_of, selected, country_groups)
    if not mask.any():
        raise ValueError(f"No aircraft match the selection {sorted(selected)}")
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
