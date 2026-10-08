"""Ledger/archive and structured-report checks; no training or network is performed."""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import validate_proposal
from .research import tree
from .util import HarnessError, digest, file_hash, finite_number, load_json


def _verify_result(h, run):
    finite_number(run["result"]["value"], "stored metric")
    saved = load_json(h.store.run_dir(run["id"]) / "evaluation.json")
    for key in ("schema_version", "metric", "value", "folds", "seed", "protocol_hash"):
        if saved.get(key) != run["result"].get(key):
            raise HarnessError(f"Evaluation and ledger disagree on {key}")


def report_data(h) -> dict:
    from .module_usage import bindings
    rows = []
    for run in h.store.rows(limit=1000000):
        rows.append({"id": run["id"], "parent_id": run["parent_id"], "status": run["status"],
                     "purpose": run["purpose"], "value": run["result"]["value"] if run["result"] else None,
                     "protocol_hash": run["protocol_hash"],
                     "artifacts": run["result"].get("sealed_files", {}) if run["result"] else {},
                     "modules": [{"module_id": b["module_id"], "content_hash": b["content_hash"],
                                  "binding_hash": b["binding_hash"]} for b in bindings(h, run["id"])]})
    return {"schema_version": 1, "competition_id": h.policy["competition_id"],
            "metric": h.policy["metric"], "champion": tree(h)["champion"], "runs": rows,
            "scientific_significance": "not_assessed",
            "kaggle_jobs": [{"id": r["id"], "ref": r["ref"], "status": r["status"]}
                            for r in h.store.db.execute("SELECT id,ref,status FROM kaggle_jobs ORDER BY at,id")]}


def audit_report(h, document: dict) -> dict:
    if not isinstance(document, dict) or document.get("schema_version") != 1 or not isinstance(document.get("runs"), list):
        raise HarnessError("Report data needs schema_version=1 and a runs list")
    errors = []
    if document.get("scientific_significance") != "not_assessed":
        errors.append("This report cannot claim an assessed significance test")
    if document.get("competition_id") != h.policy["competition_id"] or document.get("metric") != h.policy["metric"]:
        errors.append("Competition/metric differs from the ledger")
    if document.get("champion") != tree(h)["champion"]:
        errors.append("Champion differs from the ledger")
    seen = set()
    required = {"id", "parent_id", "status", "purpose", "value", "protocol_hash", "artifacts"}
    for row in document["runs"]:
        if not isinstance(row, dict) or not required.issubset(row) or set(row)-required-{"modules"}:
            errors.append("Malformed report row")
            continue
        rid = row["id"]
        if not isinstance(rid, str) or rid in seen:
            errors.append("Duplicate/non-string report run ID")
            continue
        seen.add(rid)
        try:
            if row["value"] is not None:
                finite_number(row["value"], "reported metric")
            run = h.store.get(rid)
            expected = {"id": run["id"], "parent_id": run["parent_id"], "status": run["status"],
                        "purpose": run["purpose"], "value": run["result"]["value"] if run["result"] else None,
                        "protocol_hash": run["protocol_hash"],
                        "artifacts": run["result"].get("sealed_files", {}) if run["result"] else {}}
            from .module_usage import bindings
            refs = bindings(h, rid)
            if "modules" in row or refs:
                expected["modules"] = [{"module_id": b["module_id"], "content_hash": b["content_hash"],
                                        "binding_hash": b["binding_hash"]} for b in refs]
            if row != expected:
                errors.append(f"{rid}: report numbers/status/artifacts differ from the ledger")
            if run["snapshot_hash"]:
                h.verify(rid)
            if run["status"] == "completed":
                _verify_result(h, run)
        except (HarnessError, OSError, ValueError) as exc:
            errors.append(f"{rid}: {exc}")
    all_ids = {r[0] for r in h.store.db.execute("SELECT id FROM runs")}
    if seen != all_ids:
        errors.append(f"Report coverage mismatch: missing={sorted(all_ids-seen)}, extra={sorted(seen-all_ids)}")
    expected_jobs = [{"id": r["id"], "ref": r["ref"], "status": r["status"]}
                     for r in h.store.db.execute("SELECT id,ref,status FROM kaggle_jobs ORDER BY at,id")]
    if document.get("kaggle_jobs") != expected_jobs:
        errors.append("Kaggle job coverage/status differs from the ledger")
    return {"ok": not errors, "runs_checked": len(seen), "errors": errors,
            "scope": "Structured numbers, coverage and artifacts; narrative scientific claims still need review"}


def check(h, *, deep: bool = True) -> dict:
    errors, warnings = [], []
    sqlite_result = h.store.db.execute("PRAGMA integrity_check").fetchall()
    if [r[0] for r in sqlite_result] != ["ok"]:
        errors.extend(str(r[0]) for r in sqlite_result)
    errors.extend(str(tuple(r)) for r in h.store.db.execute("PRAGMA foreign_key_check"))
    runs = h.store.rows(limit=1000000)
    for run in runs:
        rid = run["id"]
        try:
            validate_proposal(run["proposal"], h.policy)
            if run["snapshot_hash"] and deep:
                h.verify(rid)
            if run["status"] == "completed":
                if not run["result"] or not run["snapshot_hash"]:
                    raise HarnessError("Completed run lacks result/snapshot")
                if deep:
                    _verify_result(h, run)
            elif run["result"] is not None:
                raise HarnessError("A non-completed run has a result")
            if run["status"] in {"ready", "running", "completed"} and not run["snapshot_hash"]:
                raise HarnessError("Active/completed run has no frozen snapshot")
            if run["status"] in {"preparing", "running"}:
                warnings.append(f"{rid}: currently active; archive is not final")
        except (HarnessError, OSError, ValueError, KeyError) as exc:
            errors.append(f"{rid}: {exc}")
    graph = tree(h)
    nodes = {r["id"]: r for r in graph["nodes"]}
    if deep:
        for node in nodes.values():
            for item in node.get("evidence_files", []):
                try:
                    if file_hash(Path(item["path"])) != item["sha256"]:
                        raise HarnessError("Referenced research file changed")
                except (HarnessError, OSError, ValueError, KeyError) as exc:
                    errors.append(f"{node['id']}: {exc}")
    visiting, done = set(), set()

    def visit(nid):
        if nid in done:
            return
        if nid in visiting:
            raise HarnessError(f"Research graph cycle at {nid}")
        if nid not in nodes:
            raise HarnessError(f"Missing graph parent {nid}")
        visiting.add(nid)
        for parent in nodes[nid].get("parents", []):
            visit(parent)
        visiting.remove(nid)
        done.add(nid)

    try:
        for nid in nodes:
            visit(nid)
    except (HarnessError, RecursionError) as exc:
        errors.append(str(exc))
    champion = h.store.champion()
    if champion:
        run = champion["run"]
        if run["status"] != "completed" or run["purpose"] != "experiment":
            errors.append("Champion is not a completed formal experiment")
    for row in h.store.db.execute("SELECT * FROM kaggle_jobs"):
        try:
            record = json.loads(row["record_json"])
            if record.get("snapshot_hash") and deep:
                root = h.store.root / "kaggle_jobs" / row["id"]
                manifest = load_json(root / "source_manifest.json")
                from .util import tree_manifest
                if digest(manifest) != record["snapshot_hash"] or tree_manifest(root / "source") != manifest:
                    raise HarnessError("Remote notebook snapshot differs from its manifest")
            if row["status"] in {"launching", "unknown"}:
                warnings.append(f"{row['id']}: uncertain remote outcome; inspect before another launch")
        except (HarnessError, OSError, ValueError) as exc:
            errors.append(f"{row['id']}: {exc}")
    return {"ok": not errors, "runs_checked": len(runs), "nodes_checked": len(nodes),
            "revision": graph["revision"], "deep": deep, "errors": errors, "warnings": warnings}
