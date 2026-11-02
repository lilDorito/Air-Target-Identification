# Calibrate predicted probabilities on the val set

def fit_calibrator(model, X_val, y_val, method):
    # isotonic or temperature scaling
    raise NotImplementedError


def reliability_curve(proba, y):
    # predicted vs observed frequency for reliability diagrams
    raise NotImplementedError
