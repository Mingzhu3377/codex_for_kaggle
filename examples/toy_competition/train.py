"""Real CPU training: clean CSV -> standardize train data -> SGD/Adam -> artifact.
This deliberately small linear regression is a pipeline demonstration, not a Kaggle baseline.
"""
import csv
import json
import math
import os
from pathlib import Path
import random
import time

from kaggle_harness.recorder import Recorder
from kaggle_harness.util import atomic_json


def main():
    config = json.loads(Path(os.environ["KH_CONFIG"]).read_text())
    datasets = json.loads(os.environ["KH_DATASETS_JSON"])
    output = Path(os.environ["KH_OUTPUT_DIR"])
    points, rejected = [], 0
    with open(datasets["train"], newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                x, y = float(row["x"]), float(row["y"])
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("nonfinite")
                points.append((x, y))
            except (ValueError, TypeError, KeyError):
                rejected += 1
    if not points:
        raise ValueError("No usable training rows")
    mean = sum(x for x, _ in points) / len(points)
    std = math.sqrt(sum((x - mean) ** 2 for x, _ in points) / len(points))
    if std == 0:
        raise ValueError("Feature x has zero variance")
    points = [((x - mean) / std, y) for x, y in points]
    atomic_json(output / "cleaning.json", {"accepted": len(points), "rejected": rejected,
                                           "mean": mean, "std": std})
    resolved = dict(config)
    mode = config["failure_mode"]
    if mode == "silent_default":
        resolved["learning_rate"] = 0.12345
    rng = random.Random(config["seed"])
    w, b = rng.uniform(-0.1, 0.1), 0.
    lr = config["learning_rate"]
    mw = mb = vw = vb = 0.
    with Recorder.from_env(resolved) as recorder:
        for step in range(1, config["steps"] + 1):
            errors = [(w * x + b - y, x) for x, y in points]
            loss = sum(e * e for e, _ in errors) / len(errors)
            gw = 2 * sum(e * x for e, x in errors) / len(errors)
            gb = 2 * sum(e for e, _ in errors) / len(errors)
            if config["optimizer"] == "sgd":
                w -= lr * gw
                b -= lr * gb
            elif config["optimizer"] == "adam":
                mw, mb = 0.9 * mw + 0.1 * gw, 0.9 * mb + 0.1 * gb
                vw, vb = 0.999 * vw + 0.001 * gw * gw, 0.999 * vb + 0.001 * gb * gb
                w -= lr * (mw / (1 - 0.9 ** step)) / (math.sqrt(vw / (1 - 0.999 ** step)) + 1e-8)
                b -= lr * (mb / (1 - 0.9 ** step)) / (math.sqrt(vb / (1 - 0.999 ** step)) + 1e-8)
            else:
                raise ValueError("optimizer must be sgd or adam")
            recorder.log(step=step, epoch=step, train_loss=loss, learning_rate=lr)
            print(f"step={step} train_loss={loss:.7f} lr={lr}", flush=True)
            if mode == "timeout" and step == 1:
                time.sleep(60)
            if mode == "crash" and step == 2:
                raise RuntimeError("Intentional test failure after two recorded steps")
            if mode == "short" and step == 2:
                break
        atomic_json(output / "model.json", {"weight": w, "bias": b, "mean": mean, "std": std,
                                            "optimizer": config["optimizer"], "seed": config["seed"]})
        recorder.finish(stop_reason="smoke_complete" if os.environ["KH_PURPOSE"] == "smoke_test"
                        else "budget_complete")


if __name__ == "__main__":
    main()
