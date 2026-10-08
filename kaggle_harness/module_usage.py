"""Explicit module selection, frozen experiment bindings, and source reuse."""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path
import shutil
import sqlite3

from .modules import ModuleLibrary, _credential_check_file, _id
from .util import (HarnessError, atomic_json, copy_source, digest, file_hash, is_within,
                   load_json, safe_relative, tree_manifest)


def library_root(h):
    row = h.store.db.execute("SELECT value FROM meta WHERE key='module_library'").fetchone()
    return Path(json.loads(row[0])) if row else None


def attach(h, root):
    root = root.resolve()
    if is_within(root, h.workspace) or is_within(h.workspace, root):
        raise HarnessError("Shared module library and competition workspace must be disjoint")
    with ModuleLibrary(root) as lib:
        checked = lib.check()
        if not checked["ok"]:
            raise HarnessError("Module library integrity check failed")
    with h.store.transaction():
        old = library_root(h)
        if old is not None and old != root:
            raise HarnessError("This store already has another library; use a new store for another library")
        if old is None:
            h.store.set_meta("module_library", str(root))
            h.store.event("module_library_attached", {"library": str(root)})
    return {"library": str(root), "attached": True}


def validate_refs(h, refs, proposal=None):
    if not isinstance(refs, list) or len(refs) > 16:
        raise HarnessError("modules must be a list of at most 16 explicit references")
    if not refs:
        return refs
    ids, targets = set(), set()
    for ref in refs:
        required = {"module_id", "files", "adaptation"}
        if not isinstance(ref, dict) or not required.issubset(ref) or set(ref)-required-{"mode"}:
            raise HarnessError("Each module reference needs module_id, files, adaptation; optional mode")
        mid = _id(ref["module_id"])
        if mid in ids:
            raise HarnessError("Duplicate module version in proposal")
        ids.add(mid)
        if not isinstance(ref["adaptation"], str) or not 1 <= len(ref["adaptation"].strip()) <= 3000:
            raise HarnessError("Describe module integration/adaptation explicitly")
        mode = ref.get("mode", "copy")
        if mode == "inherit":
            if not proposal or proposal["source"] != "parent":
                raise HarnessError("Inherited module requires source=parent")
            parent = proposal["parent_id"]
            parent_binding = next((b for b in bindings(h, parent) if b["module_id"] == mid), None)
            if parent_binding is None:
                raise HarnessError("Selected parent has no binding to this module")
            if ref["files"] != [{k: m[k] for k in ("source", "target")} for m in parent_binding["files"]]:
                raise HarnessError("Inherited module mapping must match parent; use copy for another mapping")
            record = load_json(h.store.run_dir(parent) / "modules" / mid / "module.json")
        elif mode == "copy":
            root = library_root(h)
            if root is None:
                raise HarnessError("Attach a module library before copying module versions")
            with ModuleLibrary(root) as lib:
                record = lib.verify(mid)
        else:
            raise HarnessError("Module mode must be copy or inherit")
        if not isinstance(ref["files"], list) or not 1 <= len(ref["files"]) <= 64:
            raise HarnessError("Each module reference needs 1..64 explicit file mappings")
        for mapping in ref["files"]:
            if not isinstance(mapping, dict) or set(mapping) != {"source", "target"}:
                raise HarnessError("Module file mapping needs source and target")
            source = safe_relative(mapping["source"]).as_posix()
            target = safe_relative(mapping["target"]).as_posix()
            if source not in record["files"]:
                raise HarnessError("Module file is absent from its frozen source")
            if target in targets or target in h.policy["protected_files"] or not any(
                    fnmatch.fnmatch(target, g) for g in h.policy["editable_globs"]):
                raise HarnessError("Duplicate/protected/unpermitted module target")
            targets.add(target)
    return refs


def inherited_refs(h, parent_id):
    return [{"module_id": b["module_id"], "files": [{k: m[k] for k in ("source", "target")} for m in b["files"]],
             "adaptation": "Inherited frozen adaptation. " + b["adaptation"][:2800], "mode": "inherit"}
            for b in bindings(h, parent_id)]


def freeze_before_edits(h, run_id, refs):
    if not refs:
        return []
    rd = h.store.run_dir(run_id)
    pending_bindings = []
    for ref in refs:
        mid = ref["module_id"]
        mode = ref.get("mode", "copy")
        parent_id = h.store.get(run_id)["parent_id"] if mode == "inherit" else None
        archived = rd / "modules" / mid
        archived.mkdir(parents=True)
        if parent_id:
            original = h.store.run_dir(parent_id) / "modules" / mid
            record = load_json(original / "module.json")
            source_root = original / "source"
            root = Path(next(b for b in bindings(h, parent_id) if b["module_id"] == mid)["library_root"])
        else:
            root = library_root(h)
            with ModuleLibrary(root) as lib:
                record = lib.verify(mid)
                source_root = lib.source(mid)
        manifest = copy_source(source_root, archived / "source", [], [], h.policy["snapshot_max_bytes"],
                               validate_file=_credential_check_file)
        if manifest != record["files"]:
            raise HarnessError("Module source changed during experiment freezing")
        atomic_json(archived / "module.json", record)
        copied = []
        for mapping in ref["files"]:
            source = mapping["source"]; target = mapping["target"]
            destination = rd / "source" / safe_relative(target)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not parent_id:
                shutil.copyfile(archived / "source" / safe_relative(source), destination)
            copied.append({"source": source, "target": target, "original_sha256": record["files"][source],
                           "input_sha256": file_hash(destination)})
        pending_bindings.append({"module_id": mid, "library_root": str(root), "name": record["card"]["name"],
                         "family": record["card"]["family"], "content_hash": record["content_hash"],
                         "adaptation": ref["adaptation"], "mode": mode,
                         "inherited_from": parent_id, "files": copied})
    return pending_bindings


def finish_bindings(h, run_id, bindings):
    if not bindings:
        return
    rd = h.store.run_dir(run_id)
    for binding in bindings:
        for mapping in binding["files"]:
            final = file_hash(rd / "source" / safe_relative(mapping["target"]))
            mapping.update(final_sha256=final, adapted=final != mapping["original_sha256"],
                           changed_after_copy=final != mapping["input_sha256"])
        binding["binding_hash"] = digest(binding)
    atomic_json(rd / "module_bindings.json", bindings)
    with h.store.transaction():
        for binding in bindings:
            h.store.db.execute("INSERT INTO module_bindings VALUES(?,?,?)",
                               (run_id, binding["module_id"], json.dumps(binding, ensure_ascii=False, allow_nan=False)))
        h.store.event("modules_bound", {"modules": [b["module_id"] for b in bindings]}, run_id)
    for binding in bindings:
        try:
            with ModuleLibrary(Path(binding["library_root"])) as lib:
                lib.record_use(h.store.root, run_id, binding)
        except (HarnessError, OSError, ValueError, sqlite3.Error) as exc:
            # A reference archive is sufficient for the experiment; the shared index is advisory.
            h.store.event("module_index_unavailable", {"module_id": binding["module_id"],
                           "reason": type(exc).__name__}, run_id)


def bindings(h, run_id):
    out = []
    for row in h.store.db.execute("SELECT module_id,record_json FROM module_bindings WHERE run_id=? ORDER BY module_id", (run_id,)):
        value = json.loads(row["record_json"])
        if value.get("module_id") != row["module_id"]:
            raise HarnessError("Module binding index differs from its record")
        out.append(value)
    return out


def verify_bindings(h, run_id):
    values = bindings(h, run_id)
    refs = h.store.get(run_id)["proposal"].get("modules", [])
    if {b["module_id"] for b in values} != {r["module_id"] for r in refs}:
        raise HarnessError("Module binding coverage differs from registered proposal")
    if not values:
        return 0
    rd = h.store.run_dir(run_id)
    if sorted(load_json(rd / "module_bindings.json"), key=lambda b: b["module_id"]) != values:
        raise HarnessError("Module binding archive differs from ledger")
    for binding in values:
        expected = binding["binding_hash"]
        if digest({k: v for k, v in binding.items() if k != "binding_hash"}) != expected:
            raise HarnessError("Module binding hash differs")
        archived = rd / "modules" / _id(binding["module_id"])
        record = load_json(archived / "module.json")
        if record["content_hash"] != binding["content_hash"] or digest(
                {k: v for k, v in record.items() if k != "content_hash"}) != record["content_hash"]:
            raise HarnessError("Frozen module card differs from binding")
        if tree_manifest(archived / "source") != record["files"]:
            raise HarnessError("Frozen reference module source changed")
        ref = next(r for r in refs if r["module_id"] == binding["module_id"])
        mode = ref.get("mode", "copy")
        parent = h.store.get(run_id)["parent_id"] if mode == "inherit" else None
        if (binding["mode"] != mode or binding["inherited_from"] != parent or
                binding["name"] != record["card"]["name"] or binding["family"] != record["card"]["family"]):
            raise HarnessError("Module binding version/mode differs from proposal/archive")
        if binding["adaptation"] != ref["adaptation"] or [
                {k: m[k] for k in ("source", "target")} for m in binding["files"]] != ref["files"]:
            raise HarnessError("Module binding mapping differs from proposal")
        for m in binding["files"]:
            if record["files"].get(m["source"]) != m["original_sha256"] or file_hash(
                    rd / "source" / safe_relative(m["target"])) != m["final_sha256"]:
                raise HarnessError("Module source/adaptation hash differs")
            before = file_hash(h.store.run_dir(parent) / "source" / safe_relative(m["target"])) if parent else m["original_sha256"]
            if (m["input_sha256"] != before or m["adapted"] != (m["final_sha256"] != m["original_sha256"]) or
                    m["changed_after_copy"] != (m["final_sha256"] != before)):
                raise HarnessError("Module adaptation history differs from frozen sources")
    return len(values)


def selected_context(h, module_ids):
    if not isinstance(module_ids, list) or len(module_ids) > 16:
        raise HarnessError("Select at most 16 distinct module versions")
    for mid in module_ids:
        _id(mid)
    if len(set(module_ids)) != len(module_ids):
        raise HarnessError("Duplicate selected module version")
    if not module_ids:
        return []
    root = library_root(h)
    if root is None:
        raise HarnessError("No module library attached")
    out = []
    with ModuleLibrary(root) as lib:
        for mid in module_ids:
            record = lib.verify(mid)
            # Explicitly selected module history only; no whole-bank discovery/retrieval.
            cases = lib.cases(mid)[-8:]
            compact = [{"id": c["id"], "interpretation": c["interpretation"],
                        "competition_id": c["evidence"]["competition_id"],
                        "run_status": c["evidence"]["status"], "purpose": c["evidence"]["purpose"],
                        "live_evidence": c["live_evidence"]} for c in cases]
            # Long accumulated history must not make one selected implementation unusable.
            while len(json.dumps(compact, ensure_ascii=False)) > 16000:
                compact.pop(0)
            out.append({"id": mid, "content_hash": record["content_hash"], "card": record["card"],
                        "files": record["files"], "checks": record["static_checks"],
                        "cases": compact, "recent_cases_omitted_for_budget": len(cases)-len(compact)})
    if len(json.dumps(out, ensure_ascii=False)) > 64000:
        raise HarnessError("Selected module cards/cases exceed context budget; select fewer versions")
    return out


def copy_selected_sources(h, module_ids, destination):
    root = library_root(h)
    copied = []
    if not module_ids:
        return copied
    with ModuleLibrary(root) as lib:
        total = 0
        for mid in module_ids:
            record = lib.verify(mid)
            total += sum((lib.source(mid) / rel).stat().st_size for rel in record["files"])
            if total > 16_000_000:
                raise HarnessError("Selected module context sources exceed 16 MB; select fewer modules")
            target = destination / mid
            manifest = copy_source(lib.source(mid), target, [], [], 16_000_000)
            if manifest != record["files"]:
                raise HarnessError("Selected module changed during context copying")
            copied.append({"id": mid, "path": str(target.resolve()), "content_hash": record["content_hash"]})
    return copied
