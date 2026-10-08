"""Small original teaching module. MIT license; not a published architecture."""
POWERS = (0, 1)


def features(x):
    return tuple(float(x) ** power for power in POWERS)
