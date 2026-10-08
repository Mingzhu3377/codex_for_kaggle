"""Research views over the existing ledger; experiment metrics have one source of truth."""
from __future__ import annotations

from collections import Counter
import json
import math
import re
import uuid

from .contracts import TERMINAL
from .util import HarnessError, dumps, file_hash, finite_number, integer, now
from pathlib import Path

FAILURE_LAYERS = {"data", "representation", "optimization", "objective", "execution",
                  "evaluation", "transfer", "resource", "other"}
OPERATORS = {"baseline", "improve", "ablation", "repair"}
DEFAULT_WEIGHTS = {"quality": 1.0, "progress": 0.5, "novelty": 0.35}
REPLAY_LIMIT = ("Recorded proposals only. Unrecorded branches and counterfactual model decisions "
                "are unavailable; this replay is not evidence of future competition gains.")


def _text(value, name: str, maximum: int = 1200) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= maximum:
        raise HarnessError(f"{name} must be 1..{maximum} characters")
    return value.strip()


def _sources(value) -> list[str]:
    if not isinstance(value, list) or not value:
        raise HarnessError("sources must contain a URL, evidence reference, or 'local-only'")
    return [_text(v, "source", 2000) for v in value]


def annotate(h, run_id: str, record: dict, revision: int) -> dict:
    """Append an interpretation, without changing the run's facts or promoting it."""
    required = {"family", "operator", "verdict", "failure_layer", "reason", "sources", "author"}
    if not isinstance(record, dict) or set(record) != required:
        raise HarnessError(f"Annotation needs exactly {sorted(required)}")
    family = record["family"]
    if not isinstance(family, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}", family):
        raise HarnessError("family must be a short method slug")
    if (not isinstance(record["operator"], str) or record["operator"] not in OPERATORS
            or not isinstance(record["verdict"], str) or record["verdict"] not in {"keep", "revert", "undecided"}):
        raise HarnessError("Invalid operator or verdict")
    layer = record["failure_layer"]
    if layer is not None and (not isinstance(layer, str) or layer not in FAILURE_LAYERS):
        raise HarnessError("Invalid failure_layer")
    if record["verdict"] == "revert" and layer is None:
        raise HarnessError("A reverted interpretation needs a failure_layer")
    clean = dict(record, reason=_text(record["reason"], "reason"),
                 sources=_sources(record["sources"]), author=_text(record["author"], "author", 120))
    with h.store.transaction():
        h.store.require_revision(revision)
        run = h.store.get(run_id)
        if record["verdict"] == "keep" and (run["status"] != "completed" or run["purpose"] != "experiment"):
            raise HarnessError("Only a completed formal experiment can be marked keep")
        cur = h.store.db.execute(
            "INSERT INTO run_annotations(run_id,at,record_json) VALUES(?,?,?)", (run_id, now(), dumps(clean)))
        h.store.event("run_annotated", {"annotation": int(cur.lastrowid), **clean}, run_id)
        return {"run_id": run_id, "annotation": int(cur.lastrowid), "revision": h.store.revision()}


def add_node(h, record: dict, revision: int) -> dict:
    required = {"kind", "title", "summary", "parents", "sources", "author"}
    if not isinstance(record, dict) or not required.issubset(record) or set(record)-required-{"evidence_files"}:
        raise HarnessError(f"Research node needs {sorted(required)}, with optional evidence_files")
    if not isinstance(record["kind"], str) or record["kind"] not in {"research", "hypothesis", "review"}:
        raise HarnessError("kind must be research, hypothesis or review")
    parents = record["parents"]
    if not isinstance(parents, list) or any(not isinstance(p, str) or not p for p in parents):
        raise HarnessError("parents must be node IDs")
    if len(set(parents)) != len(parents):
        raise HarnessError("Duplicate parent")
    clean = dict(record, title=_text(record["title"], "title", 200),
                 summary=_text(record["summary"], "summary", 5000),
                 sources=_sources(record["sources"]), author=_text(record["author"], "author", 120))
    if "evidence_files" in record:
        evidence = record["evidence_files"]
        if not isinstance(evidence, list) or len(evidence) > 32:
            raise HarnessError("evidence_files must be a list of at most 32 file references")
        for item in evidence:
            if (not isinstance(item, dict) or set(item) != {"path", "sha256"}
                    or not isinstance(item["path"], str) or not isinstance(item["sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
                raise HarnessError("Evidence file needs a path and SHA-256")
            if file_hash(Path(item["path"])) != item["sha256"]:
                raise HarnessError("Research evidence changed before registration")
    node_id = "R-" + uuid.uuid4().hex[:16]
    with h.store.transaction():
        h.store.require_revision(revision)
        known = {r[0] for r in h.store.db.execute("SELECT id FROM runs")}
        known.update(r[0] for r in h.store.db.execute("SELECT id FROM research_nodes"))
        known.update(r[0] for r in h.store.db.execute("SELECT id FROM kaggle_jobs"))
        if set(parents) - known:
            raise HarnessError(f"Unknown parents: {sorted(set(parents) - known)}")
        # New IDs can refer only to already existing, immutable nodes, so cycles cannot be added.
        h.store.db.execute("INSERT INTO research_nodes VALUES(?,?,?)", (node_id, now(), dumps(clean)))
        h.store.event("research_added", {"id": node_id, **clean})
        return {"id": node_id, "revision": h.store.revision()}


def _snapshot(h) -> dict:
    with h.store.transaction():
        runs = [h.store.decode(r) for r in h.store.db.execute("SELECT * FROM runs ORDER BY created_at,id")]
        annotations = {}
        for row in h.store.db.execute("SELECT * FROM run_annotations ORDER BY seq"):
            annotations[row["run_id"]] = json.loads(row["record_json"])
        studies = [{"id": r["id"], "at": r["at"], **json.loads(r["record_json"])}
                   for r in h.store.db.execute("SELECT * FROM research_nodes ORDER BY at,id")]
        visits = Counter(r[0] for r in h.store.db.execute("SELECT run_id FROM branch_visits"))
        champion = h.store.champion()
        jobs = [{"id": r["id"], "kind": "kaggle_job", "parents": [], "ref": r["ref"],
                 "account": r["account"], "status": r["status"], "metric": None,
                 "scientific_result": "not_imported"}
                for r in h.store.db.execute("SELECT * FROM kaggle_jobs ORDER BY at,id")]
        revision = h.store.revision()
    return {"runs": runs, "annotations": annotations, "studies": studies, "visits": visits,
            "champion": champion["run_id"] if champion else None, "revision": revision, "jobs": jobs}


def tree(h) -> dict:
    s = _snapshot(h)
    nodes = []
    for r in s["runs"]:
        nodes.append({"id": r["id"], "kind": "experiment", "parents": [r["parent_id"]] if r["parent_id"] else [],
                      "purpose": r["purpose"], "status": r["status"], "hypothesis": r["proposal"]["hypothesis"],
                      "expected_observation": r["proposal"]["expected_observation"],
                      "metric": r["result"]["value"] if r["result"] else None,
                      "protocol_hash": r["protocol_hash"], "wall_seconds": r["elapsed_seconds"],
                      "annotation": s["annotations"].get(r["id"]), "visits": s["visits"][r["id"]]})
    nodes.extend(s["studies"])
    nodes.extend(s["jobs"])
    return {"competition_id": h.policy["competition_id"], "revision": s["revision"],
            "champion": s["champion"], "nodes": nodes,
            "budget": h.store.budget(h.policy),
            "interpretation_policy": "Annotations are interpretations; metrics come exclusively from the run ledger."}


def compact(h) -> dict:
    """A bounded model view, with no raw source bodies, failed configs or training logs."""
    studies = []
    for row in h.store.db.execute("SELECT * FROM research_nodes ORDER BY at DESC,id DESC LIMIT 5"):
        node = json.loads(row["record_json"])
        studies.append({"id": row["id"], "kind": node["kind"], "title": node["title"],
                        "summary": node["summary"][:1200], "parents": node["parents"],
                        "sources": node["sources"][:3]})
    annotations = []
    for row in h.store.db.execute("SELECT * FROM run_annotations ORDER BY seq DESC LIMIT 8"):
        node = json.loads(row["record_json"])
        annotations.append({"run_id": row["run_id"], "family": node["family"], "verdict": node["verdict"],
                            "failure_layer": node["failure_layer"], "reason": node["reason"][:600]})
    return {"revision": h.store.revision(), "recent_research": studies, "recent_interpretations": annotations,
            "scope": "Bounded documentary context, not automatic Skill or cross-domain trick retrieval"}


def _cohort(s: dict, reference: str | None) -> tuple[str | None, str | None]:
    complete = [r for r in s["runs"] if r["status"] == "completed" and r["purpose"] == "experiment"]
    ref = reference or s["champion"] or (complete[0]["id"] if complete else None)
    if ref is None:
        return None, None
    run = next((r for r in complete if r["id"] == ref), None)
    if not run:
        raise HarnessError("Reference must be a completed formal run")
    return ref, run["protocol_hash"]


def _rank(runs: list[dict], annotations: dict, visits: Counter, direction: str, weights: dict) -> list[dict]:
    signed = 1 if direction == "maximize" else -1
    candidates = [r for r in runs if r["status"] == "completed" and r["purpose"] == "experiment"
                  and annotations.get(r["id"], {}).get("verdict") != "revert"]
    by_id = {r["id"]: r for r in runs}
    qualities = {r["id"]: signed * finite_number(r["result"]["value"], "metric") for r in candidates}
    progress = {}
    for r in candidates:
        parent = by_id.get(r["parent_id"])
        progress[r["id"]] = (signed * (r["result"]["value"] - parent["result"]["value"])
                             if parent and parent["status"] == "completed" and parent["purpose"] == "experiment"
                             and parent["protocol_hash"] == r["protocol_hash"] else 0.0)
    families = Counter(annotations.get(r["id"], {}).get("family") for r in runs)
    lo, hi = min(qualities.values(), default=0), max(qualities.values(), default=0)
    scale = max((abs(x) for x in progress.values()), default=0) or 1.0
    out = []
    for r in candidates:
        rid = r["id"]
        family = annotations.get(rid, {}).get("family")
        quality = (qualities[rid] - lo) / (hi - lo) if hi > lo else 1.0
        novelty = 1.0 / families[family] if family else 0.0
        improvement = progress[rid] / scale
        cooling = 0.5 ** (visits[rid] / 2.0)
        utility = (weights["quality"] * quality + weights["progress"] * improvement
                   + weights["novelty"] * novelty) * cooling
        out.append({"id": rid, "value": r["result"]["value"], "family": family,
                    "quality": quality, "progress": progress[rid], "progress_normalized": improvement,
                    "novelty": novelty, "visits": visits[rid], "cooling": cooling, "utility": utility,
                    "elapsed_seconds": r["elapsed_seconds"]})
    return sorted(out, key=lambda r: (-r["utility"], r["id"]))


def select(h, reference: str | None = None, *, record_visit: bool = False,
           weights: dict | None = None) -> dict:
    w = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        if not isinstance(weights, dict) or set(weights) - set(w):
            raise HarnessError("Unknown branch weights")
        for k, v in weights.items():
            w[k] = finite_number(v, k, minimum=0)
    s = _snapshot(h)
    ref, protocol = _cohort(s, reference)
    cohort = [r for r in s["runs"] if protocol and r["protocol_hash"] == protocol]
    eligible, rejected = [], []
    for r in cohort:
        if r["status"] == "completed" and r["purpose"] == "experiment":
            try:
                h.verify(r["id"])
            except HarnessError as exc:
                rejected.append({"id": r["id"], "reason": str(exc)})
                continue
        eligible.append(r)
    ranked = _rank(eligible, s["annotations"], s["visits"], h.policy["metric"]["direction"], w)
    chosen = ranked[0]["id"] if ranked else None
    if chosen and record_visit:
        with h.store.transaction():
            h.store.require_revision(s["revision"])
            h.store.db.execute("INSERT INTO branch_visits(run_id,at,reason) VALUES(?,?,?)",
                               (chosen, now(), "quality + progress + rarity, with visit cooling"))
            h.store.event("branch_selected", {"id": chosen, "weights": w}, chosen)
    return {"selected": chosen, "reference": ref, "protocol_hash": protocol,
            "revision": h.store.revision(), "ranked": ranked, "rejected": rejected,
            "weights": w, "family_count": len({r["family"] for r in ranked if r["family"]}),
            "selection_is": "A research heuristic, not a statistical significance or promotion decision"}


def replay(h, *, budget_seconds: float, steps: int = 100, reference: str | None = None) -> dict:
    """Compare policies by revealing stored children; never rank on future child results."""
    budget = finite_number(budget_seconds, "budget_seconds", minimum=0.1)
    integer(steps, "steps", 1)
    if steps > 10000:
        raise HarnessError("Replay steps must be <= 10000")
    s = _snapshot(h)
    ref, protocol = _cohort(s, reference)
    history = [r for r in s["runs"] if protocol and r["protocol_hash"] == protocol
               and r["purpose"] == "experiment" and r["status"] in TERMINAL and r["snapshot_hash"]]
    ids = {r["id"] for r in history}
    outcomes = []
    signed = 1 if h.policy["metric"]["direction"] == "maximize" else -1
    for policy in ("chronological", "greedy", "balanced"):
        observed, visits, order = {}, Counter(), []
        reserved = actual = 0.0
        for _ in range(steps):
            available = [r for r in history if r["id"] not in observed
                         and (r["parent_id"] not in ids or r["parent_id"] in observed)
                         and reserved + r["timeout_seconds"] <= budget]
            if not available:
                break
            if policy == "chronological":
                chosen = available[0]
            else:
                # Root has a neutral score. Scores of children remain hidden until selected.
                ranks = _rank(list(observed.values()), s["annotations"], visits,
                              h.policy["metric"]["direction"], DEFAULT_WEIGHTS)
                priorities = {r["id"]: r["utility"] for r in ranks}
                if policy == "greedy":
                    priorities = {r["id"]: signed * r["result"]["value"] for r in observed.values()
                                  if r["status"] == "completed"}
                root_priority = 0.85 if policy == "balanced" else -math.inf
                chosen = max(available, key=lambda r: priorities.get(r["parent_id"], root_priority))
            observed[chosen["id"]] = chosen
            visits[chosen["parent_id"]] += 1
            order.append(chosen["id"])
            reserved += chosen["timeout_seconds"]
            actual += chosen["elapsed_seconds"]
        successful = [r for r in observed.values() if r["status"] == "completed"]
        best = max(successful, key=lambda r: signed * r["result"]["value"], default=None)
        outcomes.append({"policy": policy, "order": order, "runs": len(order),
                         "reserved_seconds": reserved, "actual_seconds": actual,
                         "best_run": best["id"] if best else None, "best_value": best["result"]["value"] if best else None})
    return {"reference": ref, "policies": outcomes, "limitation": REPLAY_LIMIT,
            "budget_basis": "Declared reservations known before execution; actual times are reported separately"}
