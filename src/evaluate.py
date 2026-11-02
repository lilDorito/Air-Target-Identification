# Metrics, ablations and figures, writes aggregate results only

def score(proba, y):
    # top-1, top-5, log loss, Brier, macro-F1
    raise NotImplementedError


def compare(results):
    # table across models, scenarios, baselines, ablation, shifted test, filter on/off
    raise NotImplementedError
