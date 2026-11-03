# Load config files

from pathlib import Path
import yaml


def load_config(scenarios_path, experiment_path):
    with open(scenarios_path, "r", encoding="utf-8") as file:
        scenarios_cfg = yaml.safe_load(file)

    with open(experiment_path, "r", encoding="utf-8") as file:
        experiment_cfg = yaml.safe_load(file)

    overlap = scenarios_cfg.keys() & experiment_cfg.keys()
    if overlap:
        raise ValueError(f"Duplicate config keys: {sorted(overlap)}")

    return {**scenarios_cfg, **experiment_cfg}