import pandas as pd

REQUIRED_COLUMNS = {
    "name", "role", "type",
    "speed_min_kmh", "speed_cruise_kmh", "speed_max_kmh",
    "alt_min_m", "alt_max_m",
    "climb_max_ms", "turn_rate_max_dps",
    "rcs_min_m2", "rcs_max_m2",
    "engine_type", "engine_count", "afterburner", "ir_category",
    "length_m", "wingspan_m", "mtow_kg", "first_flight",
    "estimated_fields",
}

FEATURE_COLUMNS = [
    "speed_min_kmh", "speed_cruise_kmh", "speed_max_kmh",
    "alt_min_m", "alt_max_m",
    "climb_max_ms", "turn_rate_max_dps",
    "rcs_min_m2", "rcs_max_m2",
    "engine_type", "engine_count", "afterburner", "ir_category",
    "length_m", "wingspan_m", "mtow_kg", "first_flight"
]

def load_targets(path):
    targets = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(targets.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return targets


def estimated_mask(df):
    mask = pd.DataFrame(False, index = df.index, columns = FEATURE_COLUMNS)

    for idx, row in df.iterrows():
        cols = row["estimated_fields"]

        if pd.notna(cols):
            for col in cols.split("."):
                col = col.strip()
                if col not in mask.columns:
                    raise ValueError(f"Unknown estimated column: {col}")
                mask.at[idx, col] = True

    return mask