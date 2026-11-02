# Select scenario columns and encode them

def scenario_columns(cfg, scenario):
    # resolve the column list for A / B / C from configs/scenarios.yaml
    raise NotImplementedError


def encode(X):
    # afterburner -> 0/1, ir_category -> 0..3, engine_type -> pandas category
    raise NotImplementedError


def drop_estimated(X, mask):
    # set rule-estimated cells to NaN for the ablation run
    raise NotImplementedError
