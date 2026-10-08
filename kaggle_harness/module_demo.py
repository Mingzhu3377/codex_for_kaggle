"""Executable example of source variants, actual use, cases and cross-task reuse."""
from __future__ import annotations

import csv
from pathlib import Path
import random
import shutil

from . import integrity, module_usage
from .codex import example_proposal
from .contracts import DEFAULT_POLICY
from .engine import Harness
from .modules import ModuleLibrary, card_template
from .report import write_report
from .util import HarnessError, atomic_json


def _card(name, parents, width):
    card = card_template()
    card.update(family="polynomial-basis", name=name, parents=parents,
                mechanism=f"Compute ordered powers of one scalar, from power 0 to {width-1}.",
                author="harness-example", tags=["tabular", "feature-expansion"])
    card["interface"] = {"inputs": "One finite scalar in [-1,1].", "outputs": f"A tuple of {width} features.",
                         "constraints": ["Teaching example for a scalar regression task."],
                         "invariants": ["First feature is 1; second feature preserves the original scalar."]}
    card["usage"] = {"insertion_points": ["Before the linear readout."], "initialization": "No learnable parameters.",
                     "adaptation_notes": "The child adds a quadratic term; the function name and scalar input remain unchanged."}
    card["provenance"] = {"references": [{"source": "Original repository teaching example", "locator": "examples/module_examples",
                                          "revision": "Exact source SHA-256 is frozen in the module record."}],
                          "license": "MIT", "contributors": ["codex_for_kaggle example authors"]}
    return card


def _task(root, assets, *, name, quadratic):
    workspace = root / "workspace"
    shutil.copytree(assets / "module_competition", workspace)
    data = workspace / "data"
    data.mkdir()
    rng = random.Random(2026)
    for split, count in (("train", 120), ("validation", 60)):
        with (data / (split + ".csv")).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(["x", "y", "fold"])
            for i in range(count):
                x = rng.uniform(-1., 1.)
                y = 1+2*x+quadratic*x*x+rng.uniform(-.01, .01)
                writer.writerow([x, y, str(i % 2)])
    atomic_json(workspace / "config.json", {"optimizer": "sgd", "learning_rate": .08, "steps": 800, "seed": 42})
    policy = dict(DEFAULT_POLICY, competition_id=name, metric={"name": "mse", "direction": "minimize", "min_improvement": 1e-6},
                  validation_id="fixed-synthetic-holdout-v1", fold_ids=["0", "1"],
                  preflight_command=["{python}", "preflight.py"], protected_files=["evaluate.py", "preflight.py"],
                  editable_globs=["train.py", "feature_basis.py"], datasets={"train": "data/train.csv", "validation": "data/validation.csv"},
                  max_run_seconds=20., total_wall_seconds=120.)
    return Harness.initialize(workspace, root / "state", policy)


def _run(h, module_id, hypothesis):
    p = example_proposal(h)
    p.update(timeout_seconds=20., hypothesis=hypothesis, expected_observation="Independent held-out MSE at an identical 800-step training budget.",
             modules=[{"module_id": module_id, "files": [{"source": "feature_basis.py", "target": "feature_basis.py"}],
                       "adaptation": "Mount the actual source and call features(x) for every training row.", "mode": "copy"}])
    run_id = h.register(p)
    run = h.run(run_id)
    if run["status"] != "completed":
        raise HarnessError(f"Module demo run failed: {run['error']}")
    return run


def run_module_demo(destination: Path) -> dict:
    root = destination.resolve()
    if root.exists():
        raise HarnessError("Module demo requires a NEW destination")
    assets = Path(__file__).resolve().parent / "assets"
    root.mkdir(parents=True)
    results = {"demonstration": "Real CPU polynomial regression; fixed synthetic tasks, not Kaggle/SOTA evidence."}
    with ModuleLibrary(root / "library", create=True) as lib:
        linear = lib.add(_card("Linear basis", [], 2), assets / "module_examples" / "linear")["id"]
        quadratic = lib.add(_card("Quadratic basis", [linear], 3), assets / "module_examples" / "quadratic")["id"]
        with _task(root / "task_a", assets, name="module-demo-quadratic", quadratic=3.) as h:
            module_usage.attach(h, lib.root)
            baseline = _run(h, linear, "A linear basis underfits a quadratic response.")
            h.promote(baseline["id"], "Completed initial source-backed baseline")
            candidate = _run(h, quadratic, "Adding the frozen quadratic feature reduces approximation error.")
            comparison = h.compare(baseline["id"], candidate["id"])
            h.promote(candidate["id"], "Lower same-protocol holdout MSE after actual module use")
            lib.add_case(h, quadratic, {"outcome": "positive", "conditions": "Quadratic synthetic response; scalar input; fixed two-fold holdout; seed 42; SGD .08; 800 steps.",
                         "reason": "Lower held-out error after adding the quadratic term; this single-seed result has narrow task scope.",
                         "failure_layer": None, "author": "demo-reviewer"}, run_id=candidate["id"], baseline_id=baseline["id"])
            results["task_a"] = {"baseline": baseline["id"], "candidate": candidate["id"], "comparison": comparison,
                                 "baseline_mse": baseline["result"]["value"], "candidate_mse": candidate["result"]["value"],
                                 "integrity": integrity.check(h)}
            write_report(h, root / "task_a" / "report.html")
        results["variant_diff"] = lib.diff(linear, quadratic)
        results["export"] = lib.export(quadratic, root / "bundle")
        with ModuleLibrary(root / "imported_library", create=True) as imported:
            results["import"] = imported.import_bundle(root / "bundle")
            with _task(root / "task_b", assets, name="module-demo-linear", quadratic=0.) as h:
                module_usage.attach(h, imported.root)
                reused = _run(h, quadratic, "Reuse the same frozen quadratic module on a second task without assuming universal benefit.")
                imported.add_case(h, quadratic, {"outcome": "undecided", "conditions": "Linear response on a different dataset; no same-task linear baseline was run.",
                                  "reason": "Execution and score are recorded; no module advantage is inferred without a comparator.",
                                  "failure_layer": None, "author": "demo-reviewer"}, run_id=reused["id"])
                results["task_b"] = {"run": reused["id"], "mse": reused["result"]["value"], "integrity": integrity.check(h)}
                write_report(h, root / "task_b" / "report.html")
            results["imported_library_check"] = imported.check()
        results["module_ids"] = {"linear": linear, "quadratic": quadratic}
        results["library_check"] = lib.check()
        results["source_uses"] = lib.uses(quadratic)
    if not all(results[k]["ok"] for k in ("library_check", "imported_library_check")) or not all(results[t]["integrity"]["ok"] for t in ("task_a", "task_b")):
        raise HarnessError("Module demo integrity check failed")
    atomic_json(root / "module_demo_results.json", results)
    return results
