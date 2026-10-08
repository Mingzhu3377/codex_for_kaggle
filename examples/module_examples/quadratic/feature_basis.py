"""A source variant adding the quadratic term, with the same function interface."""
POWERS = (0, 1, 2)


def features(x):
    return tuple(float(x) ** power for power in POWERS)
