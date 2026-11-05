
import numpy as np
import pandas as pd

from .load import FEATURE_COLUMNS


def near_duplicate_pairs(df, top_n=20):
    categorical_cols = ["engine_type", "afterburner", "ir_category"]
    numeric_cols = [c for c in FEATURE_COLUMNS if c not in categorical_cols]

    n = len(df)

    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    ranges = numeric.max() - numeric.min()
    ranges = ranges.replace(0, np.nan)

    num = ((numeric - numeric.min()) / ranges).to_numpy(dtype=float)

    num_valid = ~np.isnan(num)
    num_filled = np.nan_to_num(num)

    num_diff = np.abs(num_filled[:, None, :] - num_filled[None, :, :])
    num_mask = num_valid[:, None, :] & num_valid[None, :, :]

    total = np.sum(num_diff * num_mask, axis=2)
    count = num_mask.sum(axis=2)

    cat = df[categorical_cols].to_numpy(dtype=object)
    cat_missing = pd.isna(cat)

    cat_diff = ((cat[:, None, :] != cat[None, :, :]) & ~cat_missing[:, None, :] & ~cat_missing[None, :, :])

    total += cat_diff.sum(axis=2)
    count += ((~cat_missing[:, None, :] & ~cat_missing[None, :, :]).sum(axis=2))

    distances = np.divide(
        total,
        count,
        out=np.full((n, n), np.nan),
        where=count > 0
    )

    distances[np.tril_indices(n)] = np.nan

    valid = np.isfinite(distances)
    if not valid.any():
        return pd.DataFrame(columns=["aircraft_a", "aircraft_b", "distance"])

    flat = np.flatnonzero(valid)
    values = distances.ravel()[flat]

    k = min(top_n, len(values))
    selected = np.argpartition(values, k - 1)[:k]
    selected = selected[np.argsort(values[selected])]

    i, j = np.unravel_index(flat[selected], distances.shape)

    return pd.DataFrame({"aircraft_a": df["name"].to_numpy()[i], "aircraft_b": df["name"].to_numpy()[j], "distance": values[selected]})
