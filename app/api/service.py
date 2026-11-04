# Model loading and inference for the API

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config, country_filter, features

ROOT = Path(__file__).resolve().parents[2]
MAX_TOP_K = 10
ALIAS = {"name": "aircraft", "role": "role", "type": "type"}


def _env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class LoadedModel:
    booster: lgb.Booster
    classes: list
    temperature: float
    features: list


@dataclass
class Service:
    cfg: dict
    models: dict = field(default_factory=dict)
    roster: pd.DataFrame | None = None
    mode: str = "public"

    # loading

    @classmethod
    def load(cls, models_dir=None, data_path=None):
        cfg = config.load_config(ROOT / "configs/scenarios.yaml", ROOT / "configs/experiment.yaml")
        models_dir = Path(models_dir or os.environ.get("MODELS_DIR", ROOT / "models"))
        data_path = Path(data_path or os.environ.get("DATA_PATH", ROOT / cfg["data_path"]))
        service = cls(cfg=cfg)
        force_public = _env_flag("FORCE_PUBLIC")

        for scenario in cfg["scenarios"]:
            for target in cfg["targets"]:
                if target == "name" and force_public:
                    continue
                stem = models_dir / f"{ALIAS[target]}_{scenario}"
                if stem.with_suffix(".txt").exists() and stem.with_suffix(".json").exists():
                    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
                    service.models[(target, scenario)] = LoadedModel(booster=lgb.Booster(model_file=str(stem.with_suffix(".txt"))), classes=meta["classes"], temperature=meta["temperature"], features=meta["features"])

        has_aircraft = any(t == "name" for t, _ in service.models)
        if has_aircraft and data_path.exists() and not force_public:
            columns = ["name", "manufacturer", "callsign", "country", "role", "type"]
            service.roster = pd.read_csv(data_path, usecols=columns).set_index("name")
            service.mode = "full"
        else:
            service.models = {k: v for k, v in service.models.items() if k[0] != "name"}
        return service

    # describe

    def scenarios(self):
        out = []
        for scenario in self.cfg["scenarios"]:
            targets = [ALIAS[t] for t in self.cfg["targets"] if (t, scenario) in self.models]
            if targets:
                out.append({"id": scenario, "inputs": features.scenario_columns(self.cfg, scenario), "targets": targets})
        return out

    def countries(self):
        if self.mode != "full":
            return []
        names = set()
        for value in self.roster["country"].dropna():
            names.update(part.strip() for part in str(value).split("/"))
        names.update(self.cfg.get("country_groups", {}))
        return sorted(names)

    # inference

    def _proba(self, target, scenario, observations):
        model = self.models[(target, scenario)]
        frame = pd.DataFrame([{col: obs.get(col) for col in model.features} for obs in observations], columns=model.features)
        for col in frame.columns:
            if col not in ("engine_type", "afterburner", "ir_category"):
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        raw = model.booster.predict(features.encode(frame))
        logits = np.log(np.clip(raw, 1e-12, 1.0)) / model.temperature
        combined = logits.sum(axis=0)
        combined -= combined.max()
        p = np.exp(combined)
        return model.classes, p / p.sum()

    @staticmethod
    def _top(classes, proba, k):
        order = np.argsort(-proba)[:k]
        return [{"label": classes[i], "probability": float(proba[i])} for i in order]

    def predict(self, scenario, observations, countries=None, top_k=5):
        top_k = max(1, min(int(top_k), MAX_TOP_K))
        if not any((t, scenario) in self.models for t in self.cfg["targets"]):
            raise KeyError(f"No models loaded for scenario {scenario}")
        if countries and self.mode != "full":
            raise PermissionError("The country filter needs the private (full) version")

        result = {"scenario": scenario, "snapshots": len(observations), "mode": self.mode, "filter": sorted(countries) if countries else None, "aircraft": None}

        if self.mode == "full" and ("name", scenario) in self.models:
            classes, proba = self._proba("name", scenario, observations)
            if countries:
                country_of = self.roster["country"].to_dict()
                proba = country_filter.filter_probs(proba[None, :], classes, country_of, countries, self.cfg.get("country_groups"))[0]
            details = self.roster.loc[classes]
            result["aircraft"] = [{**row, **{k: (None if pd.isna(v) else v) for k, v in details.loc[row["label"]].items()}} for row in self._top(classes, proba, top_k)]
            if countries:
                for target in ("role", "type"):
                    labels = sorted(self.roster[target].unique())
                    summed = country_filter.aggregate(proba[None, :], classes, self.roster[target].to_dict(), labels)[0]
                    result[target] = {"source": "filtered aircraft", "top": self._top(labels, summed, top_k)}
                return result

        for target in ("role", "type"):
            if (target, scenario) in self.models:
                classes, proba = self._proba(target, scenario, observations)
                result[target] = {"source": f"{target} model", "top": self._top(classes, proba, top_k)}
        return result
