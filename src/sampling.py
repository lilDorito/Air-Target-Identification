# Turn spec rows into simulated sensor observations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def sample_observations(df, n_per_aircraft, seed, cfg, shifted=False):
    # draw n observations per aircraft with its own RNG seed
    # equal count per aircraft

    rng = np.random.default_rng(seed)
    observations = []
    sampling_cfg = cfg["sampling"]
    noise_cfg = cfg["noise"]

    def _missing(*values):
        return any(pd.isna(v) for v in values)

    def _triangular(low, mode, high):
        if _missing(low, high):
            return np.nan
        if _missing(mode):
            return rng.uniform(low, high)
        return rng.triangular(low, mode, high)

    distributions = {
        "triangular": _triangular,
        "uniform": lambda low, high: np.nan if _missing(low, high) else rng.uniform(low, high),
        "log_uniform": lambda low, high: np.nan if _missing(low, high) else np.exp(rng.uniform(np.log(low), np.log(high)))
    }

    speed_distribution = (cfg["shifted_test"]["speed"] if shifted else sampling_cfg["speed"])

    noise_multiplier = (cfg["shifted_test"]["noise_multiplier"] if shifted else 1.0)

    copy_features = [
        "engine_count", "engine_type", "afterburner", "ir_category",
        "length_m", "wingspan_m", "mtow_kg", "first_flight"
    ]

    for _, aircraft in df.iterrows():
        for _ in range(n_per_aircraft):
            obs = {}

            if speed_distribution == "triangular":
                speed = distributions["triangular"](aircraft["speed_min_kmh"], aircraft["speed_cruise_kmh"], aircraft["speed_max_kmh"])
            else:
                speed = distributions[speed_distribution](aircraft["speed_min_kmh"], aircraft["speed_max_kmh"])

            speed_noise = noise_cfg["speed_rel"] * noise_multiplier
            obs["speed"] = speed * rng.uniform(1 - speed_noise, 1 + speed_noise)

            altitude = distributions[sampling_cfg["altitude"]](aircraft["alt_min_m"],aircraft["alt_max_m"])

            altitude_noise = noise_cfg["altitude_rel"] * noise_multiplier
            obs["altitude"] = altitude * rng.uniform(1 - altitude_noise, 1 + altitude_noise)

            skew = sampling_cfg["climb_turn_skew"]
            obs["climb_rate"] = aircraft["climb_max_ms"] * rng.beta(1, skew)
            obs["turn_rate"] = aircraft["turn_rate_max_dps"] * rng.beta(1, skew)

            obs["rcs"] = distributions[sampling_cfg["rcs"]](aircraft["rcs_min_m2"], aircraft["rcs_max_m2"])

            rcs_noise = noise_cfg["rcs_db"] * noise_multiplier
            rcs_db = rng.uniform(-rcs_noise, rcs_noise)
            obs["rcs"] *= 10 ** (rcs_db / 10)

            for col in copy_features:
                obs[col] = aircraft[col]

            obs["aircraft_id"] = aircraft["id"]
            obs["family"] = aircraft["family"]
            obs["country"] = aircraft["country"]

            for target in cfg["targets"]:
                obs[target] = aircraft[target]

            observations.append(obs)

    return pd.DataFrame(observations)


def make_splits(df, cfg):
    # build train / val / test / shifted-test sets with independent seeds
    # every aircraft in every set

    splits = {}

    for split_name in ("train", "val", "test", "shifted_test"):
        splits[split_name] = sample_observations(
            df=df,
            n_per_aircraft=cfg["samples_per_aircraft"][split_name],
            seed=cfg["seeds"][split_name],
            cfg=cfg,
            shifted=(split_name == "shifted_test")
        )

    return splits


def family_holdout_splits(df, cfg):
    # GroupKFold by family for the unseen-aircraft experiment (role/type only)
    # sample after splitting
    
    splitter = StratifiedGroupKFold(
        n_splits=cfg["unseen_experiment"]["n_splits"],
        shuffle=True,
        random_state=cfg["seeds"]["family_split"]
    )

    folds = {}

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X=df, y=df["role"], groups=df["family"]), start=1):
        train_specs = df.iloc[train_idx]
        test_specs = df.iloc[test_idx]

        folds[f"fold_{fold}"] = {
            "train": sample_observations(
                df=train_specs,
                n_per_aircraft=cfg["samples_per_aircraft"]["train"],
                seed=cfg["seeds"]["train"],
                cfg=cfg
            ),
            "test": sample_observations(
                df=test_specs,
                n_per_aircraft=cfg["samples_per_aircraft"]["test"],
                seed=cfg["seeds"]["test"],
                cfg=cfg
            )
        }

    return folds