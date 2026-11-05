# Run the Optuna tuning and save the best params

# python scripts/run_tuning.py     - all targets x scenarios
# python scripts/run_tuning.py --targets role type     - skip the slow aircraft studies
# python scripts/run_tuning.py --targets name --scenarios A_radar

import argparse
import sys
import tempfile
import time
import warnings
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import optuna

from src import config, load, sampling, tune

optuna.logging.set_verbosity(optuna.logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument("--targets", nargs="+", choices=["name", "role", "type"])
parser.add_argument("--scenarios", nargs="+")
parser.add_argument("--trials", type=int, help="override tuning.n_trials")
args = parser.parse_args()

cfg = config.load_config(ROOT / "configs/scenarios.yaml", ROOT / "configs/experiment.yaml")
if args.trials:
    cfg["tuning"]["n_trials"] = args.trials
targets = args.targets or cfg["targets"]
scenarios = args.scenarios or list(cfg["scenarios"])
unknown = set(scenarios) - set(cfg["scenarios"])
if unknown:
    sys.exit(f"unknown scenarios: {sorted(unknown)}; choose from {list(cfg['scenarios'])}")

best_path = ROOT / cfg["tuning"]["best_params_path"]
df = load.load_targets(ROOT / cfg["data_path"])
splits = sampling.make_splits(df, cfg)

start = time.time()
for target in targets:
    for scenario in scenarios:
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            one = Path(tmp) / "one.yaml"
            tune.tune_all(splits, df, cfg, targets=[target], scenarios=[scenario], out_path=one)
            new = yaml.safe_load(one.read_text(encoding="utf-8")) or {}
        merged = tune.load_best(best_path) or {}
        merged.update(new)
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
        print(f"  saved {list(new) or 'nothing (no trial completed)'} in {time.time() - t0:.0f}s", flush=True)

print(f"done in {(time.time() - start) / 60:.1f} min -> {best_path.relative_to(ROOT)}")