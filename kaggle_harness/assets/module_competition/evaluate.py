"""Protected evaluator computes predictions independently from the model artifact."""
import csv
import json
import math
import os
from pathlib import Path

from kaggle_harness.util import atomic_json


def main():
    config = json.loads(Path(os.environ["KH_CONFIG"]).read_text())
    data = json.loads(os.environ["KH_DATASETS_JSON"])
    model = json.loads((Path(os.environ["KH_OUTPUT_DIR"]) / "model.json").read_text())
    powers, weights = model["powers"], model["weights"]
    if powers not in ([0, 1], [0, 1, 2]) or len(weights) != len(powers):
        raise ValueError("Invalid feature basis artifact")
    errors = {fid: [] for fid in json.loads(os.environ["KH_FOLD_IDS"])}
    with open(data["validation"], encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            x, y = float(row["x"]), float(row["y"])
            prediction = sum(w*x**p for w, p in zip(weights, powers))
            error = (prediction-y)**2
            if not math.isfinite(error):
                raise ValueError("Nonfinite prediction")
            errors[row["fold"]].append(error)
    if any(not es for es in errors.values()):
        raise ValueError("Empty validation fold")
    all_errors = [e for es in errors.values() for e in es]
    result = {"schema_version": 1, "metric": "mse", "value": sum(all_errors)/len(all_errors),
              "folds": [{"id": fid, "value": sum(es)/len(es)} for fid, es in errors.items()],
              "seed": config["seed"], "protocol_hash": os.environ["KH_PROTOCOL_HASH"]}
    atomic_json(Path(os.environ["KH_EVAL_OUTPUT"]), result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
