from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from .util import HarnessError, dumps, finite_number, integer, safe_relative

ORIGINS = {"inherited_default", "deliberate", "validated"}
TERMINAL = {"completed", "failed", "incomplete", "timed_out", "cancelled", "interrupted"}

DEFAULT_POLICY = {
    "schema_version": 1,
    "competition_id": "my-competition",
    "metric": {"name": "score", "direction": "maximize", "min_improvement": 0.0},
    "validation_id": "fixed-validation-v1",
    "fold_ids": ["0"],
    "train_command": ["{python}", "train.py"],
    "evaluate_command": ["{python}", "evaluate.py"],
    "protected_files": ["evaluate.py"],
    "editable_globs": ["train.py", "models/*.py", "features/*.py"],
    "snapshot_exclude": ["outputs/*", "checkpoints/*", "*.ipynb"],
    "snapshot_max_bytes": 100_000_000,
    "datasets": {},
    "required_artifacts": ["model.json"],
    "decision_keys": ["optimizer", "learning_rate", "steps", "seed"],
    "training_budget_key": "steps",
    "min_formal_steps": 20,
    "allow_early_stopping": False,
    "max_run_seconds": 3600.0,
    "total_wall_seconds": 14400.0,
    "max_concurrent_runs": 1,
    "allocated_gpus": 0,
    "pass_env": [],
}


def validate_policy(policy: dict) -> dict:
    if not isinstance(policy, dict):
        raise HarnessError("Policy must be an object")
    missing = set(DEFAULT_POLICY) - set(policy)
    unknown = set(policy) - set(DEFAULT_POLICY)
    if missing or unknown:
        raise HarnessError(f"Policy keys: missing={sorted(missing)}, unknown={sorted(unknown)}")
    if policy["schema_version"] != 1:
        raise HarnessError("Unsupported policy schema_version")
    for key in ("competition_id", "validation_id", "training_budget_key"):
        if not isinstance(policy[key], str) or not policy[key].strip():
            raise HarnessError(f"{key} must be a nonempty string")
    metric = policy["metric"]
    if not isinstance(metric, dict) or set(metric) != {"name", "direction", "min_improvement"}:
        raise HarnessError("metric needs name, direction, min_improvement")
    if not isinstance(metric["name"], str) or not metric["name"]:
        raise HarnessError("metric.name must be nonempty")
    if metric["direction"] not in ("maximize", "minimize"):
        raise HarnessError("metric.direction must be maximize or minimize")
    finite_number(metric["min_improvement"], "min_improvement", minimum=0)
    for key in ("fold_ids", "protected_files", "editable_globs", "snapshot_exclude",
                "required_artifacts", "decision_keys", "pass_env"):
        if not isinstance(policy[key], list) or any(not isinstance(x, str) or not x for x in policy[key]):
            raise HarnessError(f"{key} must be a list of nonempty strings")
        if len(set(policy[key])) != len(policy[key]):
            raise HarnessError(f"{key} contains duplicate entries")
    for key in ("fold_ids", "protected_files", "required_artifacts"):
        if not policy[key]:
            raise HarnessError(f"{key} cannot be empty")
    for key in ("protected_files", "required_artifacts"):
        for path in policy[key]:
            safe_relative(path)
    for key in ("train_command", "evaluate_command"):
        command = policy[key]
        if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
            raise HarnessError(f"{key} must be a nonempty argv list, not a shell command")
    if not isinstance(policy["datasets"], dict) or any(not isinstance(k, str) or not k or
            not isinstance(v, str) or not v for k, v in policy["datasets"].items()):
        raise HarnessError("datasets must map names to paths")
    integer(policy["min_formal_steps"], "min_formal_steps", 1)
    integer(policy["snapshot_max_bytes"], "snapshot_max_bytes", 1)
    integer(policy["max_concurrent_runs"], "max_concurrent_runs", 1)
    integer(policy["allocated_gpus"], "allocated_gpus", 0)
    for key in ("max_run_seconds", "total_wall_seconds"):
        finite_number(policy[key], key, minimum=0.1)
    if not isinstance(policy["allow_early_stopping"], bool):
        raise HarnessError("allow_early_stopping must be boolean")
    dumps(policy)
    return policy


def validate_proposal(p: dict, policy: dict) -> dict:
    required = {"schema_version", "parent_id", "source", "purpose", "hypothesis",
                "expected_observation", "config", "decisions", "edits", "timeout_seconds"}
    if not isinstance(p, dict) or set(p) != required:
        actual = set(p) if isinstance(p, dict) else set()
        raise HarnessError(f"Proposal keys: missing={sorted(required - actual)}, "
                           f"unknown={sorted(actual - required)}")
    if p["schema_version"] != 1:
        raise HarnessError("Unsupported proposal schema_version")
    if p["parent_id"] is not None and (not isinstance(p["parent_id"], str) or not p["parent_id"]):
        raise HarnessError("parent_id must be a run ID or null")
    if p["source"] not in ("workspace", "parent"):
        raise HarnessError("source must be workspace or parent")
    if p["source"] == "parent" and p["parent_id"] is None:
        raise HarnessError("source=parent requires parent_id")
    if p["purpose"] not in ("smoke_test", "experiment"):
        raise HarnessError("purpose must be smoke_test or experiment")
    for key in ("hypothesis", "expected_observation"):
        if not isinstance(p[key], str) or not p[key].strip() or len(p[key]) > 5000:
            raise HarnessError(f"{key} must contain 1..5000 characters")
    timeout = finite_number(p["timeout_seconds"], "timeout_seconds", minimum=0.1)
    if timeout > policy["max_run_seconds"]:
        raise HarnessError("Requested timeout exceeds policy.max_run_seconds")
    if not isinstance(p["config"], dict):
        raise HarnessError("config must be a JSON object")
    config = p["config"]
    for key in policy["decision_keys"]:
        if key not in config:
            raise HarnessError(f"Missing explicit configuration: {key}")
    budget_key = policy["training_budget_key"]
    integer(config.get(budget_key), budget_key, 1)
    integer(config.get("seed"), "seed", 0)
    if p["purpose"] == "experiment" and config[budget_key] < policy["min_formal_steps"]:
        raise HarnessError("Formal experiment budget is below min_formal_steps; use smoke_test")
    if not isinstance(p["decisions"], list):
        raise HarnessError("decisions must be a list")
    covered = set()
    for d in p["decisions"]:
        if not isinstance(d, dict) or set(d) != {"key", "origin", "reason", "evidence_ids"}:
            raise HarnessError("Each decision needs key, origin, reason, evidence_ids")
        if d["key"] not in config or d["key"] in covered:
            raise HarnessError(f"Unknown/duplicate decision key: {d['key']}")
        covered.add(d["key"])
        if d["origin"] not in ORIGINS or not isinstance(d["reason"], str) or not d["reason"].strip():
            raise HarnessError("Decision needs a valid origin and nonempty reason")
        if not isinstance(d["evidence_ids"], list) or any(not isinstance(x, str) or not x for x in d["evidence_ids"]):
            raise HarnessError("evidence_ids must be a list of run IDs")
        if d["origin"] == "validated" and not d["evidence_ids"]:
            raise HarnessError("Validated decisions must cite experimental evidence IDs")
    if set(policy["decision_keys"]) - covered:
        raise HarnessError("Every decision_keys entry needs an explicit decision record")
    if not isinstance(p["edits"], list):
        raise HarnessError("edits must be a list")
    paths = set()
    total = 0
    for edit in p["edits"]:
        if not isinstance(edit, dict) or set(edit) != {"path", "content"}:
            raise HarnessError("Each edit needs path and content")
        rel = safe_relative(edit["path"]).as_posix()
        if rel in paths or rel in policy["protected_files"]:
            raise HarnessError(f"Duplicate/protected edit: {rel}")
        if not any(fnmatch.fnmatch(rel, g) for g in policy["editable_globs"]):
            raise HarnessError(f"Edit not permitted by editable_globs: {rel}")
        if not isinstance(edit["content"], str):
            raise HarnessError("Edit content must be text")
        total += len(edit["content"].encode("utf-8"))
        if total > 2_000_000:
            raise HarnessError("Code edits exceed 2 MB; perform a reviewed workspace change instead")
        paths.add(rel)
    try:
        dumps(p)
    except (ValueError, TypeError) as exc:
        raise HarnessError(f"Proposal is not finite JSON: {exc}") from exc
    return p


def diff_configs(parent: dict, child: dict, prefix: str = "") -> list[dict]:
    changes = []
    for key in sorted(set(parent) | set(child)):
        path = f"{prefix}.{key}" if prefix else key
        a, b = parent.get(key), child.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            changes.extend(diff_configs(a, b, path))
        elif key not in parent or key not in child or a != b:
            changes.append({"key": path, "before": a, "after": b,
                            "before_present": key in parent, "after_present": key in child})
    return changes
