# Turn spec rows into simulated sensor observations (see docs/PREPROCESSING_NOTICE.md, section 4)."""

def sample_observations(df, n_per_aircraft, seed, cfg, shifted=False):
    # draw n observations per aircraft with its own RNG seed
    # equal count per aircraft
    raise NotImplementedError


def make_splits(df, cfg):
    # build train / val / test / shifted-test sets with independent seeds
    # every aircraft in every set
    raise NotImplementedError


def family_holdout_splits(df, cfg):
    # GroupKFold by family for the unseen-aircraft experiment (role/type only)
    # sample after splitting
    raise NotImplementedError
