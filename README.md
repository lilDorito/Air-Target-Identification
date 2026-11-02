# Air Target Identification from Sensor-Observable Characteristics

> **Status:** work in progress. Sections marked _TODO_ are filled in as experiments finish.

## 1. Question

Given the few quantities a sensor could plausibly observe (speed, altitude, climb and turn rate, radar cross-section, and optionally engine and IR cues), **which exact aircraft, UAV or missile is it?** As by-products, the models also predict its **role** (26 classes) and **coarse type** (5 classes).

Outputs are probabilities (top-5), and they can be restricted at prediction time to the countries an operator actually faces.

## 2. Data

- 535 air targets: crewed aircraft, helicopters, fixed-wing UAVs, missiles and guided munitions.
- Built from public sources, AI-recalled reference figures with an audit, and documented estimation rules. Full provenance is in [`docs/DATA_COLLECTION_NOTICE.md`](docs/DATA_COLLECTION_NOTICE.md).

**Data availability.** The compiled table is **not distributed** in this repository, and neither are derived observations or the aircraft-level model, which effectively encodes the table.  The code runs on any CSV with the schema described in [`data/README.md`](data/README.md).

## 3. Method

Preprocessing is described in [`docs/PREPROCESSING_NOTICE.md`](docs/PREPROCESSING_NOTICE.md).

- **Observations:** each spec row is turned into simulated sensor observations, using equal samples per aircraft and independently seeded train / validation / test / shifted-test sets.
- **Feature scenarios:** A (radar only), B (radar + other sensors), C (everything, excluding flight profile).
- **Targets:** aircraft (main), role, type. 3 × 3 = 9 LightGBM models.
- **Baselines:** class prior, and nearest neighbour on spec values.
- **Calibration** on the validation set, then the **country filter** by masking and renormalising probabilities.
- **Unseen-aircraft experiment** for role and type: hold out whole families.
- **Ablation:** with vs without rule-estimated cells.

## 4. Results

_TODO:_ tables from `results/metrics/`, figures from `results/figures/`.

## 5. Conclusions

_TODO_

## 6. Limitations

_TODO_

## 7. Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put the dataset CSV at data/targets.csv (see data/README.md)
jupyter nbconvert --to notebook --execute notebooks/main.ipynb
```

All seeds and settings are in [`configs/`](configs/). Tested with Python 3.11 (_TODO: confirm_).

## 8. AI use

See [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md).

## 9. Authors

- lilDorito
- DEFORTEX

## 10. License

Code is released under the [MIT License](LICENSE). The license covers code and documentation only; the dataset is not part of this repository and is not licensed for redistribution.
