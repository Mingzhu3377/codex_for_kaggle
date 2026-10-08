"""Protected, task-specific checks. They cannot establish paper fidelity or SOTA."""
import math
import os
from pathlib import Path

from feature_basis import POWERS, features
from kaggle_harness.util import atomic_json


def main():
    probes = [-1., -.25, 0., .5, 1.]
    actual = [features(x) for x in probes]
    interface_ok = POWERS in ((0, 1), (0, 1, 2)) and all(len(fs) == len(POWERS) for fs in actual)
    order_ok = all(tuple(x**p for p in POWERS) == fs for x, fs in zip(probes, actual))
    finite_ok = all(math.isfinite(f) for fs in actual for f in fs)
    fs = features(.4)
    weights = [.2] * len(fs)
    def loss(ws):
        return (sum(w*f for w, f in zip(ws, fs))-.8)**2
    gradient_ok = True
    for j in range(len(weights)):
        plus, minus = list(weights), list(weights)
        plus[j] += 1e-6; minus[j] -= 1e-6
        numeric = (loss(plus)-loss(minus))/2e-6
        analytic = 2*(sum(w*f for w, f in zip(weights, fs))-.8)*fs[j]
        gradient_ok &= abs(numeric-analytic) < 1e-8
    checks = [{"name": name, "passed": bool(passed), "details": details} for name, passed, details in (
        ("interface", interface_ok, "Constant/linear or constant/linear/quadratic ordered feature vector."),
        ("feature_order", order_ok, "Compared actual source outputs with an independent power calculation."),
        ("finite_outputs", finite_ok, "Five finite probes in the declared input interval [-1,1]."),
        ("gradient", gradient_ok, "Squared-error weight gradient matches central finite differences."))]
    atomic_json(Path(os.environ["KH_PREFLIGHT_OUTPUT"]),
                {"schema_version": 1, "scope": "Synthetic polynomial regression only; five probes and one gradient point.",
                 "checks": checks})


if __name__ == "__main__":
    main()
