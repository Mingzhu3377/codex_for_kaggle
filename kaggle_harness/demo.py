from __future__ import annotations

import json
from pathlib import Path
import shutil

from .codex import example_proposal
from .engine import Harness
from .report import write_report
from .util import HarnessError, atomic_json, load_json


def run_demo(destination: Path, examples: Path | None = None) -> dict:
    root = destination.resolve()
    if root.exists():
        raise HarnessError("Demo requires a NEW destination; it will never erase prior experiments")
    if examples is None:
        local = Path(__file__).resolve().parent.parent / "examples"
        examples = local if (local / "toy_competition").is_dir() else Path(__file__).resolve().parent / "assets"
    if not (examples / "toy_competition").is_dir():
        raise HarnessError("Run demo from the unpacked source bundle, or pass --examples PATH")
    root.mkdir(parents=True)
    shutil.copytree(examples / "toy_competition", root / "workspace")
    result = {"demonstration": "real CPU linear regression on synthetic data; not a competition score", "runs": {}}
    with Harness.initialize(root / "workspace", root / "state", load_json(examples / "toy_policy.json")) as h:
        p = example_proposal(h)
        p["timeout_seconds"] = 10.
        p["hypothesis"] = "A two-step run is sufficient only to check the pipeline."
        p["expected_observation"] = "All files appear, but champion promotion must be rejected."
        p["purpose"], p["config"]["steps"] = "smoke_test", 2
        rid = h.register(p)
        result["runs"]["smoke"] = {"id": rid, "status": h.run(rid)["status"]}
        try:
            h.promote(rid, "This promotion must be rejected")
            raise AssertionError("Smoke promotion unexpectedly succeeded")
        except HarnessError:
            result["smoke_promotion_blocked"] = True
        p = example_proposal(h)
        p["timeout_seconds"] = 10.
        p["hypothesis"] = "Establish a complete reproducible baseline at learning rate 0.002."
        p["expected_observation"] = "Forty logged steps, fixed holdout evaluation, saved model."
        rid = h.register(p)
        run = h.run(rid)
        if run["status"] != "completed":
            raise HarnessError(f"Demo baseline failed: {run['error']}")
        h.promote(rid, "First complete baseline")
        result["runs"]["baseline"] = {"id": rid, "status": run["status"], "mse": run["result"]["value"]}
        p = example_proposal(h)
        p["timeout_seconds"] = 10.
        p["hypothesis"] = "Increasing SGD learning rate improves under-converged regression."
        p["expected_observation"] = "Faster decreasing loss and lower holdout MSE at equal steps."
        p["config"]["learning_rate"] = .06
        for d in p["decisions"]:
            if d["key"] == "learning_rate":
                d.update(origin="deliberate", reason="Test whether the baseline is under-converged.")
        rid = h.register(p)
        run = h.run(rid)
        if run["status"] != "completed":
            raise HarnessError(f"Demo challenger failed: {run['error']}")
        comparison = h.compare(result["runs"]["baseline"]["id"], rid)
        h.promote(rid, "Fixed-protocol improvement above the configured threshold")
        result["runs"]["challenger"] = {"id": rid, "status": run["status"], "mse": run["result"]["value"]}
        result["comparison"] = comparison
        h.store.note(rid, "At the same training budget, the larger learning rate reduced under-convergence in this toy task.",
                     ["optimizer", "learning_rate"], "demo-reviewer")
        for mode in ("crash", "short", "timeout"):
            p = example_proposal(h)
            p["config"]["failure_mode"] = mode
            p["timeout_seconds"] = .8 if mode == "timeout" else 10.
            p["hypothesis"] = f"Fault injection: {mode} must retain records and preserve champion."
            p["expected_observation"] = "Failure status recorded; no automatic champion overwrite."
            rid = h.register(p)
            run = h.run(rid)
            result["runs"][mode] = {"id": rid, "status": run["status"]}
        result["champion"] = h.store.champion()["run_id"]
        result["champion_survived_failures"] = result["champion"] == result["runs"]["challenger"]["id"]
        result["integrity"] = h.verify(result["champion"])
        atomic_json(root / "context.json", h.context())
        write_report(h, root / "report.html")
        result["paths"] = {"workspace": str(root / "workspace"), "store": str(root / "state"),
                           "report": str(root / "report.html")}
    atomic_json(root / "demo_results.json", result)
    return result
