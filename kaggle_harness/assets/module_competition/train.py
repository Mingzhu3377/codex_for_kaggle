"""Actual CPU training using the explicitly mounted feature module."""
import csv
import json
import os
from pathlib import Path
import random

from feature_basis import POWERS, features
from kaggle_harness.recorder import Recorder
from kaggle_harness.util import atomic_json


def main():
    config = json.loads(Path(os.environ["KH_CONFIG"]).read_text())
    data = json.loads(os.environ["KH_DATASETS_JSON"])
    with open(data["train"], encoding="utf-8", newline="") as stream:
        points = [(features(float(r["x"])), float(r["y"])) for r in csv.DictReader(stream)]
    rng = random.Random(config["seed"])
    weights = [rng.uniform(-.1, .1) for _ in POWERS]
    lr = config["learning_rate"]
    if config["optimizer"] != "sgd":
        raise ValueError("This fixed-budget example implements SGD only")
    with Recorder.from_env(config) as recorder:
        for step in range(1, config["steps"] + 1):
            errors = [(sum(w*f for w, f in zip(weights, fs))-y, fs) for fs, y in points]
            loss = sum(e*e for e, _ in errors) / len(errors)
            gradients = [2*sum(e*fs[j] for e, fs in errors)/len(errors) for j in range(len(weights))]
            weights = [w-lr*g for w, g in zip(weights, gradients)]
            recorder.log(step=step, epoch=step, train_loss=loss, learning_rate=lr)
            if step == 1 or step % 100 == 0:
                print(f"step={step} loss={loss:.8f}", flush=True)
        atomic_json(Path(os.environ["KH_OUTPUT_DIR"]) / "model.json",
                    {"powers": list(POWERS), "weights": weights, "seed": config["seed"]})
        recorder.finish(stop_reason="budget_complete")


if __name__ == "__main__":
    main()
