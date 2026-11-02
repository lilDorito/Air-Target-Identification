# Restrict predictions to selected countries by masking and renormalising

def filter_probs(proba, classes, country_of, allowed):
    # zero out aircraft from other countries, renormalise to 1
    raise NotImplementedError


def aggregate(proba_aircraft, role_of, type_of):
    # sum aircraft probs into role and type probs
    raise NotImplementedError
