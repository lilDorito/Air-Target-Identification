# Air Target Identification from Sensor-Observable Characteristics

> **Status:** models trained and tuned, notebook complete up to error analysis, web demo deployed. Still to do: figures (notebook section 8) and conclusions.

## 1. Question

Given the few quantities a sensor could plausibly observe (speed, altitude, climb and turn rate, radar cross-section, and optionally engine and IR cues), **which exact aircraft, UAV or missile is it?** As by-products, the models also predict its **role** (26 classes) and **coarse type** (5 classes).

The outputs are probabilities (top-5). At prediction time they can be restricted to the types a chosen country **operates today**, for example Ukraine's F-16s or Russia's Geran drones, wherever they were built.

## 2. Data

- 535 air targets: crewed aircraft, helicopters, fixed-wing UAVs, missiles and guided munitions.
- Built from public sources, AI-recalled reference figures with an audit, and documented estimation rules. Every value records where it came from. Full provenance is in [`docs/DATA_COLLECTION_NOTICE.md`](docs/DATA_COLLECTION_NOTICE.md).
- Each row also lists its current **operators** (116 states), which are used only by the filter. Dissolved states such as the USSR appear only as the country of origin.

**Data availability.** The compiled table is **not distributed** here. Neither are derived observations or the aircraft-level models, which effectively encode the table. The code runs on any CSV at `data/targets.csv` with the columns listed in the schema of the data notice (`load.REQUIRED_COLUMNS` checks them).

## 3. Method

- **Observations:** each spec row is turned into simulated sensor snapshots:
  - speed is drawn from a triangular distribution with its mode at cruise speed;
  - altitude is uniform;
  - climb and turn rate are skewed towards zero;
  - RCS is log-uniform, with dB noise added.

  Every aircraft gets the same number of samples, in independently seeded train / validation / test / shifted-test sets. The shifted test set uses uniform speed and doubled noise.
- **Feature scenarios:** A (radar only), B (radar + engine and IR cues), C (everything, excluding flight profile). C is an upper bound, not a field scenario.
- **Targets:** aircraft (main), role and type, giving 3 × 3 = 9 LightGBM models. Hyperparameters were tuned with Optuna (TPE sampler, median pruning, validation log loss) on a per-aircraft subsample. They are stored in [`results/tuning/best_params.yaml`](results/tuning/best_params.yaml).
- **Baselines:** class prior, and k-nearest neighbours.
- **Calibration:** temperature scaling on the validation set.
- **Tracks:** several snapshots of one contact are combined by summing log-probabilities.
- **Operator filter:** after calibration, aircraft that no selected country operates are masked out and the rest are renormalised. Role and type are then summed from the filtered aircraft. The models never see the operators, so the filter needs no retraining.
- **Experiments:**
  - ablation: rule-estimated cells removed;
  - unseen families for role and type (whole families held out);
  - distribution shift;
  - track length.

## 4. Results

Test set, top-1 / top-5 accuracy in %, from [`results/metrics/bundle_results.csv`](results/metrics/bundle_results.csv) (tuned models).

| Target | A: radar only | B: + sensors | C: everything |
|---|---|---|---|
| Aircraft (535 classes) | 14.1 / 39.5 | 28.1 / 66.0 | 98.7 / 100 |
| Aircraft, track of 3 snapshots | 27.4 / 61.4 | 42.0 / 80.5 | 98.8 / 100 |
| Aircraft, track of 10 snapshots | 44.3 / 79.4 | 57.8 / 92.3 | 98.8 / 100 |
| Role (26), dedicated model | 42.8 / 90.5 | 55.6 / 97.7 | 98.9 / 100 |
| Role, summed from aircraft | 48.0 / 91.3 | 59.6 / 97.9 | 98.9 / 100 |
| Type (5) | 91.3 / 100 | 96.4 / 100 | 100 / 100 |

- **Chance level** for the aircraft model is 0.2% top-1. One radar snapshot is genuinely ambiguous, because many types share speed and altitude ranges. Tracks are what make identification usable: ten snapshots roughly triple top-1 accuracy on radar only.
- **Summing aircraft probabilities beats the dedicated role model** in A and B.
- **Under distribution shift** (uniform speed, double noise), aircraft top-1 accuracy on radar only drops from 14.1% to 9.5%, and type from 91.3% to 86.2%.
- **Calibration:** expected calibration error is at most 0.06 on the test set.
- **Operator filter:** selecting a country cuts the candidates from 535 to the types it operates (97 for Russia, 67 for Ukraine), which sharply improves top-5 accuracy on radar only. Final numbers are in notebook section 5 (`results/metrics/operator_filter_results.csv`).

Ablation, unseen-family, filter and error-analysis tables come from [`notebooks/main.ipynb`](notebooks/main.ipynb) and are saved to `results/metrics/`. _TODO:_ figures in `results/figures/`.

## 5. Conclusions

_TODO_

## 6. Limitations

- **Simulated observations.** Training data is sampled from published spec ranges, not recorded sensor tracks. Real radar noise, aspect-dependent RCS and manoeuvre patterns are not modelled.
- **Estimated inputs.** About 36% of filled feature cells are rule-based estimates, mostly RCS, turn rate, climb and minimum altitude. The ablation measures how much the models rely on them.
- **Near-identical variants** (e.g. Su-24M / Su-24MR) cannot be separated from kinematics. Notebook section 1.7 counts them.
- **Scenario C** is close to a lookup of the spec sheet, which is why it is an upper bound and not a result.
- **Operator lists** are a snapshot (November 2026) and change with every transfer and retirement.

## 7. Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put the dataset CSV at data/targets.csv
pytest # unit tests on synthetic data
python scripts/run_smoke.py # whole pipeline on tiny settings, about a minute
jupyter nbconvert --to notebook --execute notebooks/main.ipynb # set QUICK = True in the first cell for a fast check
```

- `scripts/run_tuning.py` reruns Optuna (hours).
- `scripts/build_bundle.py` trains the 9 models the web app serves.

All seeds and settings are in [`configs/`](configs/). Tested with Python 3.13.

## 8. Web app

A FastAPI service (`app/api`) and a React page (`app/web`).

- **Full mode** runs locally only, when the aircraft models and the CSV are present. It names the aircraft (top-5 with callsign, maker and origin) and has the operator filter.
- **Public mode** (`FORCE_PUBLIC=1`) is what is deployed (API on Render, page on Vercel). It serves only the role and type models, so no aircraft-level data leaves the machine. The API never returns the roster as a list.

```bash
cp .env.example .env # remove FORCE_PUBLIC for the full version
uvicorn app.api.main:app --reload --env-file .env
cd app/web && npm install && npm run dev # http://localhost:5173
```

## 9. AI use

See [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md).

## 10. Authors

- lilDorito
- DEFORTEX

## 11. License

Code is released under the [MIT License](LICENSE). The license covers code and documentation only; the dataset is not part of this repository and is not licensed for redistribution.
