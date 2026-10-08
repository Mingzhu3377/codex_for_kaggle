"""Protected evaluator: fixed held-out rows, fixed fold IDs, no model-side score trust."""
import csv
import json
import math
import os
from pathlib import Path

from kaggle_harness.util import atomic_json


def main():
    config = json.loads(Path(os.environ["KH_CONFIG"]).read_text())
    datasets = json.loads(os.environ["KH_DATASETS_JSON"])
    model = json.loads((Path(os.environ["KH_OUTPUT_DIR"]) / "model.json").read_text())
    fold_ids = json.loads(os.environ["KH_FOLD_IDS"])
    errors = {fid: [] for fid in fold_ids}
    with open(datasets["validation"], newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            x, y = float(row["x"]), float(row["y"])
            prediction = model["weight"] * ((x - model["mean"]) / model["std"]) + model["bias"]
            error = (prediction - y) ** 2
            if not math.isfinite(error):
                raise ValueError("Nonfinite prediction")
            errors[row["fold"]].append(error)
    if any(not values for values in errors.values()):
        raise ValueError("Empty validation fold")
    folds = [{"id": fid, "value": sum(values) / len(values)} for fid, values in errors.items()]
    all_errors = [v for values in errors.values() for v in values]
    result = {"schema_version": 1, "metric": "mse", "value": sum(all_errors) / len(all_errors),
              "folds": folds, "seed": config["seed"], "protocol_hash": os.environ["KH_PROTOCOL_HASH"]}
    atomic_json(Path(os.environ["KH_EVAL_OUTPUT"]), result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
