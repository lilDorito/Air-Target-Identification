# Train the 9 LightGBM models (3 targets x 3 scenarios) + baselines

def train_one(X, y, cfg, target):
    # fit one LightGBM model
    # class weights for role/type
    # fixed seed
    raise NotImplementedError


def baselines(X_train, y_train, X_test):
    # class-prior and nearest-neighbour
    # log + standardise inside its own pipeline
    raise NotImplementedError


def train_all(cfg):
    # loop over targets x scenarios, save models
    # aircraft models stay gitignored
    raise NotImplementedError
