# results/

- `metrics/`: aggregate tables (accuracy, top-5, log loss, macro-F1 per model, scenario and ablation)
- `figures/`: reliability diagrams, feature importance, role/type confusion matrices, accuracy vs scenario

**Publish aggregates only.** Per-aircraft tables and 535×535 confusion matrices reveal the data. Files matching `per_aircraft*` or `*aircraft_confusion*` are gitignored. Write anything aircraft-level under those names.
