# Train and save the model bundle the API serves

import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src import config, load, pipeline, tune

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="tiny settings for a fast test bundle")
parser.add_argument("--models", default=str(ROOT / "models"))
args = parser.parse_args()

cfg = config.load_config(ROOT / "configs/scenarios.yaml", ROOT / "configs/experiment.yaml")
if args.quick:
    cfg["samples_per_aircraft"] = {"train": 30, "val": 8, "test": 8, "shifted_test": 8}
    cfg["lightgbm"].update(n_estimators=120, early_stopping_rounds=15)

df = load.load_targets(ROOT / cfg["data_path"])
tuned = tune.load_best(ROOT / cfg["tuning"]["best_params_path"])
t = time.time()
results, _ = pipeline.run(df, cfg, tuned_params=tuned, model_dir=args.models, with_baselines=False)
out = ROOT / "results/metrics/bundle_results.csv"
out.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(out, index=False)
print(results[(results.split == "test") & (results.model == "lightgbm")][["target", "scenario", "top1", "top5"]].round(3).to_string(index=False))
print(f"bundle written to {args.models} in {time.time() - t:.0f}s")
