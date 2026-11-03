# Unit tests on synthetic data

import numpy as np
import pandas as pd
import pytest

from src import calibrate, config, country_filter, evaluate, features, load, sampling

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return config.load_config(ROOT / "configs/scenarios.yaml", ROOT / "configs/experiment.yaml")


@pytest.fixture(scope="module")
def spec():
    # 4 fake aircraft with every column the pipeline needs
    # one has missing RCS

    return pd.DataFrame({
        "id": [1, 2, 3, 4], "name": ["A1", "A2", "B1", "C1"], "family": ["A", "A", "B", "C"],
        "country": ["USA", "USSR", "Iran/Russia", "France"],
        "role": ["fighter", "fighter", "drone", "bomber"], "type": ["jet", "jet", "uav", "jet"],
        "speed_min_kmh": [200, 220, 100, 250], "speed_cruise_kmh": [900, 850, 150, 800],
        "speed_max_kmh": [2000, 1900, 180, 1000], "alt_min_m": [50, 50, 60, 150],
        "alt_max_m": [15000, 16000, 4000, 12000], "climb_max_ms": [250, 230, 6, 15],
        "turn_rate_max_dps": [20, 19, 30, 3], "rcs_min_m2": [1, 2, np.nan, 10], "rcs_max_m2": [4, 8, np.nan, 40],
        "engine_type": ["turbofan", "turbofan", "piston", "turbofan"], "engine_count": [1, 2, 1, 4],
        "afterburner": ["yes", "yes", "no", "no"], "ir_category": ["very high", "very high", "low", "high"],
        "length_m": [15, 17, 3.5, 40], "wingspan_m": [10, 11, 2.5, 40], "mtow_kg": [20000, 22000, 200, 150000],
        "first_flight": [1975, 1977, 2021, 1960],
        "estimated_fields": ["rcs_min_m2, rcs_max_m2", None, "alt_min_m.turn_rate_max_dps", None],
    })


def test_estimated_mask_accepts_both_separators(spec):
    mask = load.estimated_mask(spec)
    assert mask.loc[0, "rcs_min_m2"] and mask.loc[0, "rcs_max_m2"]
    assert mask.loc[2, "alt_min_m"] and mask.loc[2, "turn_rate_max_dps"]
    assert mask.values.sum() == 4


def test_sampling_equal_counts_and_nan_safe(spec, cfg):
    obs = sampling.sample_observations(spec, 7, seed=1, cfg=cfg)
    assert obs.groupby("aircraft_id").size().eq(7).all()
    assert obs.loc[obs.aircraft_id == 3, "rcs"].isna().all()
    assert obs.loc[obs.aircraft_id != 3, "rcs"].notna().all()


def test_sampling_is_reproducible_and_seeds_differ(spec, cfg):
    a = sampling.sample_observations(spec, 5, seed=1, cfg=cfg)
    b = sampling.sample_observations(spec, 5, seed=1, cfg=cfg)
    c = sampling.sample_observations(spec, 5, seed=2, cfg=cfg)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["speed"], c["speed"])


def test_encode_handles_missing_columns_and_fixed_categories():
    X = features.encode(pd.DataFrame({"speed": [1.0]}))
    assert list(X.columns) == ["speed"]
    one = features.encode(pd.DataFrame({"engine_type": ["piston"]}))
    two = features.encode(pd.DataFrame({"engine_type": ["turbofan"]}))
    assert list(one.engine_type.cat.categories) == list(two.engine_type.cat.categories)


def test_scenarios_never_use_forbidden_columns(cfg):
    for scenario in cfg["scenarios"]:
        assert not set(features.scenario_columns(cfg, scenario)) & set(cfg["never_inputs"])


def test_country_filter_renormalises_and_matches_multi_origin(spec):
    classes = list(spec["name"])
    country_of = dict(zip(spec["name"], spec["country"]))
    proba = np.full((2, 4), 0.25)
    out = country_filter.filter_probs(proba, classes, country_of, ["Russia"], {"Russia": ["USSR"]})
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    assert out[0, 0] == 0 and out[0, 3] == 0
    assert out[0, 1] > 0 and out[0, 2] > 0


def test_aggregate_sums_to_one(spec):
    classes = list(spec["name"])
    role_of = dict(zip(spec["name"], spec["role"]))
    proba = np.array([[0.4, 0.3, 0.2, 0.1]])
    summed = country_filter.aggregate(proba, classes, role_of, ["bomber", "drone", "fighter"])
    np.testing.assert_allclose(summed, [[0.1, 0.2, 0.7]])


def test_temperature_calibration_softens_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 500)
    proba = np.full((500, 5), 0.01)
    proba[np.arange(500), np.where(rng.random(500) < 0.5, y, (y + 1) % 5)] = 0.96
    cal = calibrate.fit_calibrator(proba, y)
    assert cal.temperature > 1
    assert cal.transform(proba).max(axis=1).mean() < proba.max(axis=1).mean()


def test_scores_and_tracks():
    proba = np.eye(3)[[0, 1, 2, 0]]
    y = np.array([0, 1, 2, 0])
    s = evaluate.score(proba, y)
    assert s["top1"] == 1.0 and s["macro_f1"] == 1.0
    tp, first = evaluate.track_proba(proba, np.array([1, 2, 3, 1]), 2)
    assert len(tp) == 1 and y[first[0]] == 0
    tp, first = evaluate.track_proba(proba, np.array([1, 2, 3, 1]), 5)
    assert tp.shape == (0, 3) and first.dtype.kind == "i"
