# models/

Trained models are written here. This folder is **gitignored**, except:

- `role_*.txt` and `type_*.txt`: LightGBM role and type models (text format). These predict categories and do not reproduce the table.

**Never committed:** aircraft-level models (`aircraft_*`). A model that names 535 individual aircraft effectively encodes the dataset.
