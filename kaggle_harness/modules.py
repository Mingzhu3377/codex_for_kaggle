"""Source-backed, manually selected module versions and conditional experience.

The library is shared across competitions. Experiment facts remain in their run
stores; library cases are immutable interpretations with frozen provenance.
Registering or importing a module never imports/executes its Python code.
"""
from __future__ import annotations

import ast
from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import re
import sqlite3
import uuid

from .contracts import TERMINAL
from .util import (HarnessError, atomic_json, copy_source, digest, dumps, file_hash,
                   finite_number, is_within, load_json, now, safe_relative, tree_manifest)

MAX_SOURCE_BYTES = 8_000_000
MAX_BUNDLE_BYTES = 64_000_000
CARD_KEYS = {"schema_version", "family", "name", "parents", "mechanism", "interface",
             "usage", "provenance", "limitations", "tags", "author"}
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS modules(
 id TEXT PRIMARY KEY, family TEXT NOT NULL, at TEXT NOT NULL,
 record_json TEXT NOT NULL, content_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS module_edges(
 child TEXT NOT NULL REFERENCES modules(id), parent TEXT NOT NULL REFERENCES modules(id),
 PRIMARY KEY(child,parent)
);
CREATE TABLE IF NOT EXISTS module_cases(
 id TEXT PRIMARY KEY, module_id TEXT NOT NULL REFERENCES modules(id), at TEXT NOT NULL,
 record_json TEXT NOT NULL, content_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS module_uses(
 id TEXT PRIMARY KEY,module_id TEXT NOT NULL REFERENCES modules(id),at TEXT NOT NULL,
 store_root TEXT NOT NULL,run_id TEXT NOT NULL,binding_hash TEXT NOT NULL,
 UNIQUE(module_id,store_root,run_id)
);
CREATE TABLE IF NOT EXISTS events(
 seq INTEGER PRIMARY KEY AUTOINCREMENT,at TEXT NOT NULL,kind TEXT NOT NULL,payload_json TEXT NOT NULL
);
"""
for _table in ("modules", "module_edges", "module_cases", "module_uses", "events"):
    for _action in ("UPDATE", "DELETE"):
        SCHEMA += (f"CREATE TRIGGER IF NOT EXISTS {_table}_no_{_action.lower()} "
                   f"BEFORE {_action} ON {_table} BEGIN SELECT RAISE(ABORT, "
                   "'module history is immutable'); END;\n")


def _text(value, name, limit=3000):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= limit:
        raise HarnessError(f"{name} must contain 1..{limit} characters")
    return value.strip()


def _texts(value, name, *, maximum=32, required=False):
    if not isinstance(value, list) or len(value) > maximum or (required and not value):
        raise HarnessError(f"{name} must be a list of at most {maximum} strings")
    return [_text(v, name) for v in value]


def _id(value, prefix="M"):
    if not isinstance(value, str) or not re.fullmatch(prefix + r"-[0-9a-f]{16}", value):
        raise HarnessError(f"Invalid {prefix} identifier")
    return value


def _credential_check_text(text):
    # Match actual token forms, not words such as 'api_key' in a tutorial.
    known = [v for k, v in os.environ.items()
             if k.upper() in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "KAGGLE_API_TOKEN",
                              "KAGGLE_KEY", "GH_TOKEN", "GITHUB_TOKEN"} and len(v) >= 8]
    if any(v in text for v in known) or re.search(
            r"KGAT_[A-Za-z0-9._-]{8,}|(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}|"
            r"sk-(?:proj-)?[A-Za-z0-9_-]{24,}", text):
        raise HarnessError("Credential found in module material; remove it before registration")


def _credential_check_file(path):
    with path.open("rb") as stream:
        tail = b""
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            _credential_check_text((tail + chunk).decode("utf-8", errors="replace"))
            tail = chunk[-8192:]


def validate_card(card):
    if not isinstance(card, dict) or set(card) != CARD_KEYS or isinstance(card["schema_version"], bool) or card["schema_version"] != 1:
        raise HarnessError(f"Module card needs exactly {sorted(CARD_KEYS)} and schema_version=1")
    if not isinstance(card["family"], str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}", card["family"]):
        raise HarnessError("Module family must be a short slug")
    for key, limit in (("name", 200), ("mechanism", 5000), ("author", 120)):
        _text(card[key], key, limit)
    if not isinstance(card["parents"], list) or len(card["parents"]) > 16:
        raise HarnessError("Module parents must be a list of at most 16 version IDs")
    if len(set(_id(v) for v in card["parents"])) != len(card["parents"]):
        raise HarnessError("Duplicate module parent")
    for key in ("limitations", "tags"):
        _texts(card[key], key)
    interface = card["interface"]
    if not isinstance(interface, dict) or set(interface) != {"inputs", "outputs", "constraints", "invariants"}:
        raise HarnessError("Module interface needs inputs, outputs, constraints, invariants")
    for key in ("inputs", "outputs"):
        _text(interface[key], key)
    for key in ("constraints", "invariants"):
        _texts(interface[key], key)
    usage = card["usage"]
    if not isinstance(usage, dict) or set(usage) != {"insertion_points", "initialization", "adaptation_notes"}:
        raise HarnessError("Module usage needs insertion_points, initialization, adaptation_notes")
    _texts(usage["insertion_points"], "insertion_points")
    for key in ("initialization", "adaptation_notes"):
        _text(usage[key], key)
    provenance = card["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {"references", "license", "contributors"}:
        raise HarnessError("Module provenance needs references, license, contributors")
    refs = provenance["references"]
    if not isinstance(refs, list) or not 1 <= len(refs) <= 16:
        raise HarnessError("Module needs 1..16 explicit source references")
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"source", "locator", "revision"}:
            raise HarnessError("Each reference needs source, locator, revision")
        for key in ref:
            _text(ref[key], key)
    _text(provenance["license"], "license")
    _texts(provenance["contributors"], "contributors", required=True)
    if len(dumps(card)) > 12000:
        raise HarnessError("Module card exceeds 12,000 characters; keep long material in its source folder")
    _credential_check_text(dumps(card))
    return card


def card_template():
    return {"schema_version": 1, "family": "replace-with-method-family", "name": "Replace with a version name",
            "parents": [], "mechanism": "Describe what the actual source computes.",
            "interface": {"inputs": "Input shape and meaning", "outputs": "Output shape and meaning",
                          "constraints": [], "invariants": []},
            "usage": {"insertion_points": [], "initialization": "Describe the actual initialization",
                      "adaptation_notes": "Record adaptations; an unchanged reference is also valid."},
            "provenance": {"references": [{"source": "local-only", "locator": "file/class/section",
                                             "revision": "Record a commit or source SHA-256"}],
                           "license": "Record the source license or explicitly unknown",
                           "contributors": ["Record actual contributors or explicitly unknown"]},
            "limitations": ["No task-level effectiveness is established by registration."],
            "tags": [], "author": "human"}


def _static_checks(source):
    count = 0
    for rel in tree_manifest(source):
        path = source / rel
        if path.suffix != ".py":
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=path.name)
        except (SyntaxError, UnicodeError, RecursionError) as exc:
            raise HarnessError(f"Python syntax check failed in {path.name}: {type(exc).__name__}") from exc
        count += 1
    return {"python_files": count, "syntax": "passed" if count else "not_applicable",
            "runtime": "not_tested", "paper_fidelity": "not_assessed", "task_effectiveness": "not_assessed"}


def _record_hash(record):
    return digest({k: v for k, v in record.items() if k != "content_hash"})


class ModuleLibrary:
    def __init__(self, root: Path, *, create=False):
        self.root = root.resolve()
        if create:
            self.root.mkdir(parents=True, exist_ok=False)
            (self.root / "versions").mkdir()
            (self.root / "cases").mkdir()
        elif not (self.root / "library.sqlite3").is_file():
            raise HarnessError(f"Not a module library: {self.root}")
        self.db = sqlite3.connect(self.root / "library.sqlite3", timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA busy_timeout=30000")
        if not create:
            try:
                row = self.db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
                if row is None or row[0] != "1":
                    raise HarnessError("Unsupported module library schema")
            except (sqlite3.Error, HarnessError) as exc:
                self.close()
                raise HarnessError("Not a supported module library database") from exc
        try:
            self.db.executescript(SCHEMA)
            if create:
                self.db.execute("INSERT INTO meta VALUES('schema_version','1')")
        except sqlite3.Error as exc:
            self.close()
            raise HarnessError("Module library database could not be initialized") from exc

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        else:
            self.db.execute("COMMIT")

    def event(self, kind, payload):
        self.db.execute("INSERT INTO events(at,kind,payload_json) VALUES(?,?,?)", (now(), kind, dumps(payload)))

    def get(self, module_id):
        _id(module_id)
        row = self.db.execute("SELECT * FROM modules WHERE id=?", (module_id,)).fetchone()
        if row is None:
            raise HarnessError(f"Unknown module version: {module_id}")
        record = json.loads(row["record_json"])
        if record.get("id") != row["id"] or record.get("at") != row["at"] or record.get("content_hash") != row["content_hash"] or record.get("card", {}).get("family") != row["family"]:
            raise HarnessError("Module indexed fields differ from its record")
        return record

    def source(self, module_id):
        self.get(module_id)
        return self.root / "versions" / module_id / "source"

    def _validate_record(self, record, source):
        required = {"schema_version", "id", "at", "card", "files", "static_checks", "content_hash"}
        if not isinstance(record, dict) or set(record) != required or isinstance(record["schema_version"], bool) or record["schema_version"] != 1:
            raise HarnessError("Malformed stored module record")
        _id(record["id"])
        _text(record["at"], "module timestamp")
        validate_card(record["card"])
        if record["content_hash"] != _record_hash(record):
            raise HarnessError("Module record hash differs")
        if not isinstance(record["files"], dict) or not record["files"]:
            raise HarnessError("Module has no source files")
        for rel, checksum in record["files"].items():
            safe_relative(rel)
            if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
                raise HarnessError("Invalid module source hash")
        if sum((source / safe_relative(p)).stat().st_size for p in record["files"]) > MAX_SOURCE_BYTES:
            raise HarnessError("Module source exceeds size limit")
        if tree_manifest(source) != record["files"]:
            raise HarnessError(f"Module source changed: {record['id']}")
        if _static_checks(source) != record["static_checks"]:
            raise HarnessError("Module static-check record differs")
        for rel in record["files"]:
            _credential_check_file(source / rel)
        return record

    def verify(self, module_id):
        record = self.get(module_id)
        disk = load_json(self.root / "versions" / module_id / "module.json")
        if disk != record:
            raise HarnessError(f"Stored module card changed: {module_id}")
        self._validate_record(record, self.source(module_id))
        parents = [r[0] for r in self.db.execute("SELECT parent FROM module_edges WHERE child=?", (module_id,))]
        if set(parents) != set(record["card"]["parents"]):
            raise HarnessError("Module parent graph differs from its card")
        return record

    def add(self, card, folder):
        validate_card(card)
        folder = folder.resolve()
        if is_within(folder, self.root) or is_within(self.root, folder):
            raise HarnessError("Import folder and module library must be disjoint")
        for parent in card["parents"]:
            self.verify(parent)
        module_id = "M-" + uuid.uuid4().hex[:16]
        version = self.root / "versions" / module_id
        version.mkdir()
        try:
            files = copy_source(folder, version / "source", [], [], MAX_SOURCE_BYTES,
                                validate_file=_credential_check_file)
            if not files:
                raise HarnessError("No module source files remain after exclusions")
            record = {"schema_version": 1, "id": module_id, "at": now(), "card": card,
                      "files": files, "static_checks": _static_checks(version / "source")}
            record["content_hash"] = _record_hash(record)
            atomic_json(version / "module.json", record)
            with self.transaction():
                self.db.execute("INSERT INTO modules VALUES(?,?,?,?,?)",
                                (module_id, card["family"], record["at"], dumps(record), record["content_hash"]))
                for parent in card["parents"]:
                    self.db.execute("INSERT INTO module_edges VALUES(?,?)", (module_id, parent))
                self.event("module_added", {"id": module_id, "content_hash": record["content_hash"]})
            return record
        except BaseException:
            # Keep the failed attempt for inspection; it never becomes a registered version.
            self.event("module_import_failed", {"attempt_id": module_id})
            raise

    def list(self, *, family=None, tag=None, limit=100):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10000:
            raise HarnessError("Module list limit must be in 1..10000")
        out = []
        for row in self.db.execute("SELECT record_json FROM modules ORDER BY at DESC,id DESC"):
            r = json.loads(row[0]); c = r["card"]
            if (family is None or c["family"] == family) and (tag is None or tag in c["tags"]):
                out.append({"id": r["id"], "name": c["name"], "family": c["family"], "parents": c["parents"],
                            "tags": c["tags"], "content_hash": r["content_hash"], "checks": r["static_checks"]})
            if len(out) >= limit:
                break
        return out

    def diff(self, parent_id, candidate_id):
        a, b = self.verify(parent_id), self.verify(candidate_id)
        x, y = a["files"], b["files"]
        return {"parent": parent_id, "candidate": candidate_id,
                "added": sorted(set(y)-set(x)), "removed": sorted(set(x)-set(y)),
                "modified": sorted(k for k in set(x)&set(y) if x[k] != y[k]),
                "card_changes": [k for k in sorted(CARD_KEYS) if a["card"][k] != b["card"][k]]}

    def cases(self, module_id):
        self.get(module_id)
        out = []
        for row in self.db.execute("SELECT * FROM module_cases WHERE module_id=? ORDER BY at,id", (module_id,)):
            case = self._case_record(row)
            case["live_evidence"] = live_case_evidence(case)
            out.append(case)
        return out

    def _case_record(self, row):
        case = json.loads(row["record_json"])
        self._validate_case(case)
        if any(case[k] != row[k] for k in ("id", "module_id", "at", "content_hash")):
            raise HarnessError("Module case indexed fields differ from its record")
        if load_json(self.root / "cases" / (row["id"] + ".json")) != case:
            raise HarnessError("Module case archive differs")
        if case["evidence"]["binding"].get("content_hash") != self.get(case["module_id"])["content_hash"]:
            raise HarnessError("Module case references another source version")
        return case

    def record_use(self, store_root, run_id, binding):
        module = self.verify(binding["module_id"])
        if module["content_hash"] != binding["content_hash"]:
            raise HarnessError("Usage index version differs from frozen module")
        with self.transaction():
            row = self.db.execute("SELECT binding_hash FROM module_uses WHERE module_id=? AND store_root=? AND run_id=?",
                                  (binding["module_id"], str(store_root), run_id)).fetchone()
            if row is not None:
                if row[0] != binding["binding_hash"]:
                    raise HarnessError("Module usage already points to another binding")
                return
            self.db.execute("INSERT INTO module_uses VALUES(?,?,?,?,?,?)",
                            ("MU-" + uuid.uuid4().hex[:16], binding["module_id"], now(),
                             str(store_root), run_id, binding["binding_hash"]))
            self.event("module_used", {"module_id": binding["module_id"], "run_id": run_id})

    def uses(self, module_id):
        self.get(module_id)
        rows = []
        for row in self.db.execute("SELECT * FROM module_uses WHERE module_id=? ORDER BY at,id", (module_id,)):
            use = dict(row)
            database = Path(use["store_root"]) / "ledger.sqlite3"
            try:
                if not database.is_file():
                    raise OSError("Original run store missing")
                with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                    db.row_factory = sqlite3.Row
                    run = db.execute("SELECT status,purpose,result_json,protocol_hash FROM runs WHERE id=?", (use["run_id"],)).fetchone()
                    bound = db.execute("SELECT record_json FROM module_bindings WHERE run_id=? AND module_id=?",
                                       (use["run_id"], module_id)).fetchone()
                    if run is None or bound is None or json.loads(bound[0])["binding_hash"] != use["binding_hash"]:
                        use["live_evidence"] = {"status": "changed"}
                    else:
                        result = json.loads(run["result_json"]) if run["result_json"] else None
                        use["live_evidence"] = {"status": "metadata_matches", "run_status": run["status"],
                            "purpose": run["purpose"], "value": result["value"] if result else None,
                            "protocol_hash": run["protocol_hash"]}
            except (sqlite3.Error, OSError, ValueError, KeyError):
                use["live_evidence"] = {"status": "unavailable"}
            use["scope"] = "Source binding and live run metadata; execution/causality is not inferred"
            rows.append(use)
        return rows

    def add_case(self, h, module_id, interpretation, *, run_id, baseline_id=None):
        self.verify(module_id)
        keys = {"outcome", "conditions", "reason", "failure_layer", "author"}
        if not isinstance(interpretation, dict) or set(interpretation) != keys:
            raise HarnessError(f"Module case interpretation needs exactly {sorted(keys)}")
        if not isinstance(interpretation["outcome"], str) or interpretation["outcome"] not in {"positive", "negative", "mixed", "neutral", "undecided"}:
            raise HarnessError("Invalid module case outcome")
        from .research import FAILURE_LAYERS
        if interpretation["failure_layer"] is not None and (not isinstance(interpretation["failure_layer"], str) or interpretation["failure_layer"] not in FAILURE_LAYERS):
            raise HarnessError("Invalid module case failure layer")
        for key in ("conditions", "reason", "author"):
            _text(interpretation[key], key, 120 if key == "author" else 5000)
        run = h.store.get(run_id)
        if run["status"] not in TERMINAL:
            raise HarnessError("Module cases need a terminal experiment")
        if interpretation["outcome"] != "undecided" and (run["status"] != "completed" or run["purpose"] != "experiment"):
            raise HarnessError("Smoke/failed executions can only record an undecided method outcome")
        h.verify(run_id)
        row = h.store.db.execute("SELECT record_json FROM module_bindings WHERE run_id=? AND module_id=?",
                                 (run_id, module_id)).fetchone()
        if row is None:
            raise HarnessError("This run has no frozen binding to this module version")
        binding = json.loads(row[0])
        if binding["library_root"] != str(self.root):
            raise HarnessError("Module case library differs from the run's bound library")
        evidence = experiment_evidence(h, run, binding)
        if baseline_id:
            evidence["baseline"] = h.compare(baseline_id, run_id)
        _credential_check_text(dumps(evidence))
        case = {"schema_version": 1, "id": "MC-" + uuid.uuid4().hex[:16], "at": now(),
                "module_id": module_id, "interpretation": interpretation, "evidence": evidence,
                "scientific_causality": "human_interpretation_not_automatically_established"}
        case["content_hash"] = _record_hash(case)
        self._validate_case(case)
        atomic_json(self.root / "cases" / (case["id"] + ".json"), case)
        with self.transaction():
            self.db.execute("INSERT INTO module_cases VALUES(?,?,?,?,?)",
                            (case["id"], module_id, case["at"], dumps(case), case["content_hash"]))
            self.event("case_added", {"id": case["id"], "module_id": module_id, "run_id": run_id})
        return case

    def _validate_case(self, case):
        keys = {"schema_version", "id", "at", "module_id", "interpretation", "evidence", "scientific_causality", "content_hash"}
        if not isinstance(case, dict) or set(case) != keys or isinstance(case.get("schema_version"), bool) or case.get("schema_version") != 1:
            raise HarnessError("Malformed module case")
        _id(case.get("id"), "MC"); _id(case.get("module_id"))
        _text(case["at"], "case timestamp")
        if case.get("content_hash") != _record_hash(case):
            raise HarnessError("Module case hash differs")
        if not isinstance(case.get("evidence"), dict) or not isinstance(case.get("interpretation"), dict):
            raise HarnessError("Module case lacks evidence/interpretation")
        interpretation = case["interpretation"]
        if set(interpretation) != {"outcome", "conditions", "reason", "failure_layer", "author"}:
            raise HarnessError("Malformed module case interpretation")
        outcome = interpretation["outcome"]
        if not isinstance(outcome, str) or outcome not in {"positive", "negative", "mixed", "neutral", "undecided"}:
            raise HarnessError("Invalid case outcome")
        from .research import FAILURE_LAYERS
        layer = interpretation["failure_layer"]
        if layer is not None and (not isinstance(layer, str) or layer not in FAILURE_LAYERS):
            raise HarnessError("Invalid case failure layer")
        for key in ("conditions", "reason", "author"):
            _text(interpretation[key], key, 120 if key == "author" else 5000)
        e = case["evidence"]
        needed = {"store_root", "run_id", "competition_id", "status", "purpose", "snapshot_hash", "protocol_hash", "config", "result", "binding"}
        if not needed.issubset(e) or set(e)-needed-{"baseline"}:
            raise HarnessError("Malformed case provenance")
        for key in ("store_root", "run_id", "competition_id", "snapshot_hash", "protocol_hash"):
            _text(e[key], key)
        for key in ("snapshot_hash", "protocol_hash"):
            if not re.fullmatch(r"[0-9a-f]{64}", e[key]):
                raise HarnessError("Invalid case experiment hash")
        if not isinstance(e["status"], str) or e["status"] not in TERMINAL or not isinstance(e["purpose"], str) or e["purpose"] not in {"experiment", "smoke_test"} or not isinstance(e["config"], dict):
            raise HarnessError("Invalid case experiment state")
        if outcome != "undecided" and (e["status"] != "completed" or e["purpose"] != "experiment"):
            raise HarnessError("Non-formal evidence cannot establish a method outcome")
        b = e["binding"]
        if not isinstance(b, dict) or b.get("module_id") != case["module_id"] or b.get("binding_hash") != digest({k:v for k,v in b.items() if k != "binding_hash"}):
            raise HarnessError("Case binding provenance differs")
        if e["status"] == "completed":
            if not isinstance(e["result"], dict):
                raise HarnessError("Completed case lacks an evaluator result")
            finite_number(e["result"].get("value"), "case metric")
            if e["result"].get("protocol_hash") != e["protocol_hash"] or e["result"].get("seed") != e["config"].get("seed"):
                raise HarnessError("Case result differs from its protocol/configuration")
        elif e["result"] is not None:
            raise HarnessError("Unsuccessful case must not have a metric result")
        if case["scientific_causality"] != "human_interpretation_not_automatically_established":
            raise HarnessError("Module case cannot certify causal effectiveness")
        _credential_check_text(dumps(case))

    def check(self):
        errors, warnings = [], []
        if [r[0] for r in self.db.execute("PRAGMA integrity_check")] != ["ok"]:
            errors.append("Module SQLite integrity check failed")
        errors += [str(tuple(r)) for r in self.db.execute("PRAGMA foreign_key_check")]
        ids = [r[0] for r in self.db.execute("SELECT id FROM modules")]
        # Kahn's algorithm also detects cycles if a database was modified outside this API.
        parents = {mid: set() for mid in ids}
        children = {mid: set() for mid in ids}
        for child, parent in self.db.execute("SELECT child,parent FROM module_edges"):
            if child in parents and parent in children:
                parents[child].add(parent); children[parent].add(child)
        ready = [mid for mid in ids if not parents[mid]]
        visited = 0
        while ready:
            parent = ready.pop(); visited += 1
            for child in children[parent]:
                parents[child].discard(parent)
                if not parents[child]:
                    ready.append(child)
        if visited != len(ids):
            errors.append("Module ancestry contains a cycle")
        for mid in ids:
            try:
                self.verify(mid)
            except (HarnessError, OSError, ValueError) as exc:
                errors.append(f"{mid}: {exc}")
        for row in self.db.execute("SELECT * FROM module_cases"):
            try:
                case = self._case_record(row)
                live = live_case_evidence(case)
                if live["status"] == "changed":
                    errors.append(f"{row['id']}: live experiment evidence changed")
                elif live["status"] == "unavailable":
                    warnings.append(f"{row['id']}: original run store unavailable; frozen provenance retained")
            except (HarnessError, OSError, ValueError) as exc:
                errors.append(f"{row['id']}: {exc}")
        return {"ok": not errors, "versions_checked": len(ids), "errors": errors, "warnings": warnings}

    def export(self, module_id, destination):
        destination = destination.resolve()
        if destination.exists() or is_within(destination, self.root) or is_within(self.root, destination):
            raise HarnessError("Module export needs a new directory disjoint from its library")
        ordered, seen, visiting = [], set(), set()
        def visit(mid):
            if mid in seen:
                return
            if mid in visiting or len(seen)+len(visiting) >= 1000:
                raise HarnessError("Module ancestry cycle or export limit")
            visiting.add(mid); r = self.verify(mid)
            for parent in r["card"]["parents"]:
                visit(parent)
            visiting.remove(mid); seen.add(mid); ordered.append(mid)
        visit(module_id)
        destination.mkdir(parents=True)
        cases = []
        for mid in ordered:
            record = self.verify(mid)
            target = destination / "versions" / mid
            target.mkdir(parents=True)
            files = copy_source(self.source(mid), target / "source", [], [], MAX_SOURCE_BYTES,
                                validate_file=_credential_check_file)
            if files != record["files"]:
                raise HarnessError("Exported module source differs")
            atomic_json(target / "module.json", record)
            for row in self.db.execute("SELECT * FROM module_cases WHERE module_id=?", (mid,)):
                case = self._case_record(row)
                atomic_json(destination / "cases" / (case["id"] + ".json"), case)
                cases.append(case["id"])
        files = tree_manifest(destination)
        if sum((destination / rel).stat().st_size for rel in files) > MAX_BUNDLE_BYTES:
            raise HarnessError("Module export exceeds bundle limit; partial directory retained")
        bundle = {"schema_version": 1, "root_module": module_id, "versions": ordered, "cases": cases, "files": files}
        atomic_json(destination / "bundle.json", bundle)
        return {"path": str(destination), "root_module": module_id, "versions": len(ordered), "cases": len(cases)}

    def import_bundle(self, folder):
        folder = folder.resolve()
        if is_within(folder, self.root) or is_within(self.root, folder):
            raise HarnessError("Module bundle and library must be disjoint")
        bundle = load_json(folder / "bundle.json")
        if not isinstance(bundle, dict) or set(bundle) != {"schema_version", "root_module", "versions", "cases", "files"} or isinstance(bundle["schema_version"], bool) or bundle["schema_version"] != 1:
            raise HarnessError("Invalid module bundle")
        if not isinstance(bundle["versions"], list) or not 1 <= len(bundle["versions"]) <= 1000:
            raise HarnessError("Invalid module bundle version count")
        if not isinstance(bundle["cases"], list) or len(bundle["cases"]) > 10000:
            raise HarnessError("Invalid module bundle case count")
        if not isinstance(bundle["files"], dict):
            raise HarnessError("Bundle files must be a manifest")
        expected = dict(bundle["files"])
        for rel, checksum in expected.items():
            safe_relative(rel)
            if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
                raise HarnessError("Invalid bundle file hash")
        if "bundle.json" in expected:
            raise HarnessError("Bundle manifest cannot contain itself")
        if sum((folder / rel).stat().st_size for rel in expected) > MAX_BUNDLE_BYTES:
            raise HarnessError("Bundle source exceeds size limit")
        actual = tree_manifest(folder); actual.pop("bundle.json", None)
        if actual != expected or sum((folder / p).stat().st_size for p in actual) > MAX_BUNDLE_BYTES:
            raise HarnessError("Bundle source differs from manifest or exceeds size limit")
        known = {r[0] for r in self.db.execute("SELECT id FROM modules")}
        records, cases, versions = [], [], {}
        for mid in bundle["versions"]:
            _id(mid)
            r = load_json(folder / "versions" / mid / "module.json")
            self._validate_record(r, folder / "versions" / mid / "source")
            if r["id"] != mid or set(r["card"]["parents"]) - known:
                raise HarnessError("Bundle IDs/ancestry are inconsistent or not topologically ordered")
            if mid in known:
                if self.verify(mid) != r:
                    raise HarnessError("Existing module ID differs from imported content")
            else:
                records.append(r)
            known.add(mid)
            versions[mid] = r
        if bundle["root_module"] not in bundle["versions"] or len(set(bundle["versions"])) != len(bundle["versions"]):
            raise HarnessError("Bundle root/duplicate versions are invalid")
        for cid in bundle["cases"]:
            _id(cid, "MC")
        if len(set(bundle["cases"])) != len(bundle["cases"]):
            raise HarnessError("Duplicate bundle case")
        for cid in bundle["cases"]:
            _id(cid, "MC"); c = load_json(folder / "cases" / (cid + ".json")); self._validate_case(c)
            if c["id"] != cid or c["module_id"] not in bundle["versions"]:
                raise HarnessError("Bundle case references another module")
            if c["evidence"]["binding"].get("content_hash") != versions[c["module_id"]]["content_hash"]:
                raise HarnessError("Bundle case binding differs from its module source version")
            row = self.db.execute("SELECT * FROM module_cases WHERE id=?", (cid,)).fetchone()
            if row is not None:
                if self._case_record(row) != c:
                    raise HarnessError("Existing case ID differs from imported content")
            else:
                cases.append(c)
        wanted = {f"versions/{mid}/module.json" for mid in bundle["versions"]}
        for mid in bundle["versions"]:
            record = load_json(folder / "versions" / mid / "module.json")
            wanted.update(f"versions/{mid}/source/{rel}" for rel in record["files"])
        wanted.update(f"cases/{cid}.json" for cid in bundle["cases"])
        if set(expected) != wanted:
            raise HarnessError("Bundle contains material outside its listed versions/cases")
        # Copy validated bytes before the atomic database commit. Failed copies remain inspectable.
        for r in records:
            dest = self.root / "versions" / r["id"]
            if dest.exists():
                raise HarnessError("Unregistered import directory exists; inspect it before retrying")
            dest.mkdir()
            copy_source(folder / "versions" / r["id"] / "source", dest / "source", [], [], MAX_SOURCE_BYTES,
                        validate_file=_credential_check_file)
            atomic_json(dest / "module.json", r)
            self._validate_record(r, dest / "source")
        for c in cases:
            path = self.root / "cases" / (c["id"] + ".json")
            if path.exists():
                raise HarnessError("Unregistered case archive exists; inspect it before retrying")
            atomic_json(path, c)
        with self.transaction():
            for r in records:
                self.db.execute("INSERT INTO modules VALUES(?,?,?,?,?)",
                                (r["id"], r["card"]["family"], r["at"], dumps(r), r["content_hash"]))
                for parent in r["card"]["parents"]:
                    self.db.execute("INSERT INTO module_edges VALUES(?,?)", (r["id"], parent))
            for c in cases:
                self.db.execute("INSERT INTO module_cases VALUES(?,?,?,?,?)",
                                (c["id"], c["module_id"], c["at"], dumps(c), c["content_hash"]))
            self.event("bundle_imported", {"root": bundle["root_module"], "versions_added": len(records), "cases_added": len(cases)})
        return {"root_module": bundle["root_module"], "versions_added": len(records), "cases_added": len(cases)}


def experiment_evidence(h, run, binding):
    return {"store_root": str(h.store.root), "run_id": run["id"], "competition_id": h.policy["competition_id"],
            "status": run["status"], "purpose": run["purpose"], "snapshot_hash": run["snapshot_hash"],
            "protocol_hash": run["protocol_hash"], "config": run["proposal"]["config"],
            "result": run["result"], "binding": binding}


def live_case_evidence(case):
    """Read-only metadata comparison; creation verifies full run files via Harness.verify."""
    e = case["evidence"]; dbpath = Path(e["store_root"]) / "ledger.sqlite3"
    if not dbpath.is_file():
        return {"status": "unavailable", "scope": "original experiment metadata"}
    try:
        with closing(sqlite3.connect(dbpath.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM runs WHERE id=?", (e["run_id"],)).fetchone()
            bound = db.execute("SELECT record_json FROM module_bindings WHERE run_id=? AND module_id=?",
                               (e["run_id"], case["module_id"])).fetchone()
            if row is None or bound is None:
                return {"status": "changed", "scope": "original experiment metadata"}
            value = dict(row)
            actual = {"status": value["status"], "purpose": value["purpose"], "snapshot_hash": value["snapshot_hash"],
                      "protocol_hash": value["protocol_hash"], "config": json.loads(value["proposal_json"])["config"],
                      "result": json.loads(value["result_json"]) if value["result_json"] else None,
                      "binding": json.loads(bound[0])}
            matches = all(actual[k] == e[k] for k in actual)
            return {"status": "metadata_matches" if matches else "changed",
                    "scope": "original experiment metadata; archives are not rehashed by this lookup"}
    except (sqlite3.Error, OSError, ValueError, KeyError):
        return {"status": "unavailable", "scope": "original experiment metadata"}
