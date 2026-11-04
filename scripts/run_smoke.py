# Quick end-to-end check of the pipeline on the real dataset with tiny settings

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src import config, load, pipeline, sampling

cfg = config.load_config(ROOT / "configs/scenarios.yaml", ROOT / "configs/experiment.yaml")
cfg["samples_per_aircraft"] = {"train": 20, "val": 5, "test": 5, "shifted_test": 5}
cfg["lightgbm"]["n_estimators"] = 60
cfg["lightgbm"]["early_stopping_rounds"] = 10
df = load.load_targets(ROOT / cfg["data_path"])

t = time.time()
splits = sampling.make_splits(df, cfg)
print("splits:", {k: v.shape for k, v in splits.items()})

res, art = pipeline.run(df, cfg, targets=["type", "name"], scenarios=["A_radar"], splits=splits)
cols = ["target", "scenario", "model", "split", "top1", "top5", "log_loss"]
print(res[cols].round(3).to_string(index=False))
print(pipeline.filter_check(art[("name", "A_radar")], df, ["Russia"])[["filter", "n_classes", "top1", "top5"]].round(3).to_string(index=False))
abl, _ = pipeline.run(df, cfg, targets=["type"], scenarios=["A_radar"], splits=None, ablation=True, with_baselines=False)
print("ablation ran:", len(abl), "rows")
print("unseen families ran:", len(pipeline.unseen_families(df, cfg, "A_radar", "type", num_boost_round=20)), "folds")
print(f"SMOKE TEST OK in {time.time() - t:.0f}s")
