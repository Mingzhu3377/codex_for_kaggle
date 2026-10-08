from __future__ import annotations

import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

from .contracts import (DEFAULT_POLICY, TERMINAL, diff_configs, validate_policy,
                        validate_proposal)
from .store import Store
from .util import (HarnessError, atomic_json, copy_source, dataset_digest, dataset_manifest,
                   digest, dumps, file_hash, finite_number, git_info, integer, is_within,
                   load_json, now, owner_info, process_alive, process_token, safe_relative,
                   terminate_process, tree_manifest)


class RunCancelled(Exception):
    pass


class RunTimedOut(Exception):
    pass


class RunIncomplete(Exception):
    pass


class Harness:
    def __init__(self, store: Path):
        self.store = Store(store)
        self.policy = self.store.get_meta("policy")
        self.workspace = Path(self.store.get_meta("workspace"))

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @classmethod
    def initialize(cls, workspace: Path, root: Path, policy: dict) -> "Harness":
        workspace, root = workspace.resolve(), root.resolve()
        validate_policy(policy)
        if not workspace.is_dir():
            raise HarnessError(f"Workspace does not exist: {workspace}")
        if is_within(root, workspace) or is_within(workspace, root):
            raise HarnessError("Store and code workspace must be disjoint directories")
        if root.exists():
            raise HarnessError(f"Store path already exists; choose a NEW directory: {root}")
        p = json.loads(dumps(policy))
        for name, raw in p["datasets"].items():
            path = Path(raw)
            if not path.is_absolute():
                path = workspace / path
            if path.is_symlink():
                raise HarnessError(f"Dataset symlink not allowed: {path}")
            p["datasets"][name] = str(path.resolve())
        protected = {name: file_hash(workspace / safe_relative(name)) for name in p["protected_files"]}
        data = dataset_manifest(p["datasets"])
        with Store(root, create=True) as store:
            store.set_meta("policy", p)
            store.set_meta("workspace", str(workspace))
            store.set_meta("protected_hashes", protected)
            store.set_meta("initial_dataset_digest", dataset_digest(data))
            store.event("store_initialized", {"workspace": str(workspace), "policy_hash": digest(p)})
        atomic_json(root / "policy.reference.json", p)
        return cls(root)

    def _protocol_hash(self, data: dict, seed: int) -> str:
        return digest({"competition_id": self.policy["competition_id"],
                       "validation_id": self.policy["validation_id"],
                       "metric": self.policy["metric"]["name"],
                       "direction": self.policy["metric"]["direction"],
                       "fold_ids": self.policy["fold_ids"],
                       "protected_files": self.store.get_meta("protected_hashes"),
                       "evaluate_command": self.policy["evaluate_command"],
                       "datasets": dataset_digest(data), "seed": seed})

    def validate(self, proposal: dict) -> dict:
        p = validate_proposal(proposal, self.policy)
        if p["parent_id"] is not None:
            parent = self.store.get(p["parent_id"])
            if parent["status"] not in TERMINAL:
                raise HarnessError("Parent must be a terminal experiment")
            if not parent["snapshot_hash"]:
                raise HarnessError("Parent has no usable source snapshot")
        for decision in p["decisions"]:
            for run_id in decision["evidence_ids"]:
                evidence = self.store.get(run_id)
                if decision["origin"] == "validated" and (
                    evidence["status"] != "completed" or evidence["purpose"] != "experiment"
                ):
                    raise HarnessError("Validated decisions must reference completed formal experiments")
        return p

    def register(self, proposal: dict) -> str:
        p = self.validate(proposal)
        run_id = "E-" + uuid.uuid4().hex[:16]
        owner = owner_info()
        timestamp = now()
        with self.store.transaction():
            self.store.db.execute("""INSERT INTO runs
              (id,parent_id,purpose,status,created_at,updated_at,proposal_json,timeout_seconds,
               owner_pid,owner_token,host,heartbeat) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, p["parent_id"], p["purpose"], "preparing", timestamp, timestamp,
                 dumps(p), p["timeout_seconds"], owner["owner_pid"], owner["owner_token"],
                 owner["host"], time.time()))
            self.store.event("registered", {"hypothesis": p["hypothesis"]}, run_id)
        rd = self.store.run_dir(run_id)
        try:
            rd.mkdir()
            atomic_json(rd / "proposal.json", p)
            atomic_json(rd / "requested_config.json", p["config"])
            data = dataset_manifest(self.policy["datasets"])
            if dataset_digest(data) != self.store.get_meta("initial_dataset_digest"):
                raise HarnessError("Dataset differs from the initialized version. Create a new store/protocol.")
            if p["source"] == "parent":
                self.verify(p["parent_id"], include_artifacts=False)
                source = self.store.run_dir(p["parent_id"]) / "source"
            else:
                source = self.workspace
            excluded = [Path(x) for x in self.policy["datasets"].values()]
            copy_source(source, rd / "source", excluded, self.policy["snapshot_exclude"],
                        self.policy["snapshot_max_bytes"])
            for edit in p["edits"]:
                target = rd / "source" / safe_relative(edit["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(edit["content"], encoding="utf-8")
            source_manifest = tree_manifest(rd / "source")
            for rel, expected in self.store.get_meta("protected_hashes").items():
                if source_manifest.get(rel) != expected:
                    raise HarnessError(f"Protected evaluator/protocol file changed: {rel}")
            atomic_json(rd / "source_manifest.json", source_manifest)
            atomic_json(rd / "datasets.json", data)
            atomic_json(rd / "git.json", git_info(self.workspace))
            parent_config, parent_manifest = {}, {}
            if p["parent_id"]:
                parent_config = self.store.get(p["parent_id"])["proposal"]["config"]
                parent_manifest = load_json(self.store.run_dir(p["parent_id"]) / "source_manifest.json")
            code_diff = {"added": sorted(set(source_manifest) - set(parent_manifest)),
                         "removed": sorted(set(parent_manifest) - set(source_manifest)),
                         "modified": sorted(k for k in set(parent_manifest) & set(source_manifest)
                                            if parent_manifest[k] != source_manifest[k])}
            atomic_json(rd / "diff.json", {"config": diff_configs(parent_config, p["config"]),
                                           "code": code_diff})
            snapshot_hash = digest({"source": source_manifest, "config": p["config"], "datasets": data})
            protocol_hash = self._protocol_hash(data, p["config"]["seed"])
            with self.store.transaction():
                self.store.db.execute("""UPDATE runs SET status='ready', updated_at=?, snapshot_hash=?,
                  protocol_hash=?, owner_pid=NULL,owner_token=NULL,heartbeat=NULL WHERE id=? AND status='preparing'""",
                    (now(), snapshot_hash, protocol_hash, run_id))
                self.store.event("snapshot_ready", {"snapshot_hash": snapshot_hash,
                                                     "protocol_hash": protocol_hash}, run_id)
        except BaseException as exc:
            self._finish(run_id, "failed", None, f"Snapshot failed: {type(exc).__name__}: {exc}", 0)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise HarnessError(f"{run_id} was recorded but snapshot failed: {exc}") from exc
        return run_id

    def verify(self, run_id: str, include_artifacts: bool = True) -> dict:
        run = self.store.get(run_id)
        rd = self.store.run_dir(run_id)
        source = tree_manifest(rd / "source")
        expected = load_json(rd / "source_manifest.json")
        data = load_json(rd / "datasets.json")
        config = load_json(rd / "requested_config.json")
        if config != run["proposal"]["config"] or source != expected:
            raise HarnessError(f"Source/config integrity check failed: {run_id}")
        actual = digest({"source": source, "config": config, "datasets": data})
        if actual != run["snapshot_hash"]:
            raise HarnessError(f"Snapshot manifest integrity check failed: {run_id}")
        checked = 0
        if include_artifacts and run["status"] == "completed":
            result = run["result"]
            expected_artifacts = result.get("sealed_files", {})
            if not expected_artifacts:
                raise HarnessError("Completed run has no artifact seals")
            for rel, expected_hash in expected_artifacts.items():
                if file_hash(rd / safe_relative(rel)) != expected_hash:
                    raise HarnessError(f"Artifact integrity check failed: {rel}")
                checked += 1
        return {"run_id": run_id, "source_verified": True, "sealed_files_checked": checked}

    def _claim(self, run_id: str) -> None:
        owner = owner_info()
        with self.store.transaction():
            run = self.store.get(run_id)
            if run["status"] != "ready":
                raise HarnessError(f"Cannot start run in state {run['status']}; create a child run instead")
            budget = self.store.budget(self.policy)
            if budget["active_runs"] >= self.policy["max_concurrent_runs"]:
                raise HarnessError("Concurrency limit reached; inspect active runs or recover stale runs")
            if run["timeout_seconds"] > budget["remaining_seconds"]:
                raise HarnessError("Insufficient total wall-time budget for this run's reservation")
            self.store.db.execute("""UPDATE runs SET status='running',updated_at=?,started_at=?,
              heartbeat=?,owner_pid=?,owner_token=?,host=? WHERE id=? AND status='ready'""",
                (now(), time.time(), time.time(), owner["owner_pid"], owner["owner_token"], owner["host"], run_id))
            self.store.event("started", {"reserved_seconds": run["timeout_seconds"], **owner}, run_id)

    def _heartbeat(self, run_id: str, proc: subprocess.Popen | None = None) -> None:
        self.store.db.execute("UPDATE runs SET heartbeat=?,child_pid=?,child_token=? WHERE id=? AND status='running'",
            (time.time(), proc.pid if proc else None, process_token(proc.pid) if proc else None, run_id))

    def _finish(self, run_id: str, status: str, result: dict | None, error: str | None, elapsed: float) -> None:
        with self.store.transaction():
            run = self.store.get(run_id)
            if run["status"] in TERMINAL:
                return
            self.store.db.execute("""UPDATE runs SET status=?,updated_at=?,ended_at=?,elapsed_seconds=?,
                result_json=?,error=?,child_pid=NULL,child_token=NULL WHERE id=?""",
                (status, now(), time.time(), max(0, elapsed), dumps(result) if result else None, error, run_id))
            self.store.event("finished", {"status": status, "error": error, "elapsed_seconds": elapsed}, run_id)
        rd = self.store.run_dir(run_id)
        if rd.exists():
            atomic_json(rd / "outcome.json", {"status": status, "result": result, "error": error,
                                              "elapsed_seconds": elapsed, "at": now()})

    def _environment(self, rd: Path, protocol_hash: str) -> dict[str, str]:
        # Explicit allowlist; never forward Codex/Kaggle/cloud API keys by default.
        allowed = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "LANG", "LC_ALL",
                   "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES",
                   "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VIRTUAL_ENV", "CONDA_PREFIX"}
        allowed.update(self.policy["pass_env"])
        env = {key: value for key, value in os.environ.items() if key in allowed}
        home = rd / "runtime_home"
        tmp = rd / "tmp"
        home.mkdir(exist_ok=True)
        tmp.mkdir(exist_ok=True)
        env.update({"HOME": str(home), "USERPROFILE": str(home), "TMPDIR": str(tmp),
                    "TMP": str(tmp), "TEMP": str(tmp), "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
                    "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
                    "KH_RUN_ID": rd.name, "KH_CONFIG": str(rd / "requested_config.json"),
                    "KH_OUTPUT_DIR": str(rd / "output"), "KH_EVAL_OUTPUT": str(rd / "evaluation.json"),
                    "KH_DATASETS_JSON": dumps(self.policy["datasets"]),
                    "KH_PROTOCOL_HASH": protocol_hash, "KH_FOLD_IDS": dumps(self.policy["fold_ids"])})
        return env

    def _stage(self, run_id: str, name: str, command: list[str], cwd: Path,
               env: dict[str, str], deadline: float) -> int:
        rd = self.store.run_dir(run_id)
        argv = [part.replace("{python}", sys.executable) for part in command]
        self.store.event("stage_started", {"stage": name, "argv": argv}, run_id)
        next_observation = 0.0
        popen_options = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        with (rd / f"{name}.stdout.log").open("xb", buffering=0) as stdout, \
             (rd / f"{name}.stderr.log").open("xb", buffering=0) as stderr:
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                    stdout=stdout, stderr=stderr, shell=False, **popen_options)
            self._heartbeat(run_id, proc)
            try:
                while proc.poll() is None:
                    if time.monotonic() >= deadline:
                        raise RunTimedOut()
                    self._heartbeat(run_id, proc)
                    if time.monotonic() >= next_observation:
                        self._observe(run_id)
                        next_observation = time.monotonic() + 2.0
                    time.sleep(0.1)
                code = proc.returncode
            finally:
                # Also clean descendants left behind by a nominally finished parent.
                terminate_process(proc, grace=0.25)
                self._heartbeat(run_id)
                os.fsync(stdout.fileno())
                os.fsync(stderr.fileno())
        self.store.event("stage_finished", {"stage": name, "exit_code": code}, run_id)
        return code

    def _observe(self, run_id: str) -> None:
        from .monitor import observe
        try:
            observe(self, run_id)
        except (HarnessError, OSError, ValueError) as exc:
            # Monitoring is advisory; an observation failure must not become a fake training failure.
            self.store.event("monitor_unavailable", {"error": str(exc)}, run_id)

    def _training_checks(self, rd: Path, proposal: dict) -> dict:
        output = rd / "output"
        resolved = load_json(output / "resolved_config.json")
        if resolved != proposal["config"]:
            raise RunIncomplete("Runtime resolved config differs from the registered config")
        summary = load_json(output / "training_summary.json")
        if summary.get("finished") is not True or summary.get("schema_version") != 1:
            raise RunIncomplete("No valid training completion record")
        count = integer(summary.get("completed_steps"), "completed_steps", 1)
        with (output / "curves.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            last, n = 0, 0
            for row in reader:
                step = int(row["step"])
                if step <= last:
                    raise RunIncomplete("Curve steps are not strictly increasing")
                finite_number(float(row["train_loss"]), "train_loss")
                finite_number(float(row["learning_rate"]), "learning_rate", minimum=0)
                if row.get("validation_metric"):
                    finite_number(float(row["validation_metric"]), "validation_metric")
                last, n = step, n + 1
        if n == 0 or last != count or summary.get("curve_rows") != n:
            raise RunIncomplete("Training summary and curves disagree or curves are empty")
        requested = proposal["config"][self.policy["training_budget_key"]]
        reason = summary.get("stop_reason")
        if count > requested:
            raise RunIncomplete("Training exceeded its registered step budget")
        if proposal["purpose"] == "experiment":
            if count < self.policy["min_formal_steps"]:
                raise RunIncomplete("Formal experiment stopped before min_formal_steps")
            if reason == "budget_complete":
                if count != requested:
                    raise RunIncomplete("budget_complete does not match the requested step budget")
            elif reason == "early_stopping":
                if not self.policy["allow_early_stopping"] or not summary.get("details", "").strip():
                    raise RunIncomplete("Early stopping is not enabled or lacks an explanation")
            else:
                raise RunIncomplete("A smoke-test completion is not a formal experiment")
        elif reason not in ("smoke_complete", "budget_complete") or count != requested:
            raise RunIncomplete("Smoke test must finish its explicit budget")
        for rel in self.policy["required_artifacts"]:
            file_hash(output / safe_relative(rel))
        return summary

    def _evaluation_checks(self, rd: Path, run: dict) -> dict:
        result = load_json(rd / "evaluation.json")
        if not isinstance(result, dict) or result.get("schema_version") != 1:
            raise RunIncomplete("Evaluation must be a schema_version=1 object")
        if result.get("metric") != self.policy["metric"]["name"]:
            raise RunIncomplete("Evaluator returned the wrong metric")
        if result.get("protocol_hash") != run["protocol_hash"]:
            raise RunIncomplete("Evaluator protocol mismatch")
        if result.get("seed") != run["proposal"]["config"]["seed"]:
            raise RunIncomplete("Evaluation seed mismatch")
        finite_number(result.get("value"), "evaluation.value")
        folds = result.get("folds")
        if not isinstance(folds, list):
            raise RunIncomplete("Missing fold results")
        ids = []
        for fold in folds:
            if not isinstance(fold, dict) or not isinstance(fold.get("id"), str):
                raise RunIncomplete("Invalid fold result")
            ids.append(fold["id"])
            finite_number(fold.get("value"), "fold.value")
        if len(ids) != len(set(ids)) or set(ids) != set(self.policy["fold_ids"]):
            raise RunIncomplete("Missing, unexpected or duplicate folds")
        return result

    def run(self, run_id: str) -> dict:
        # Before claim: tampering does not spend budget; a registered record already exists.
        self.verify(run_id, include_artifacts=False)
        self._claim(run_id)
        started = time.monotonic()
        run = self.store.get(run_id)
        rd = self.store.run_dir(run_id)
        deadline = started + run["timeout_seconds"]
        old_handlers = {}
        if threading.current_thread() is threading.main_thread():
            def cancelled(signum, frame):
                raise RunCancelled(f"Signal {signum}")
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_handlers[sig] = signal.signal(sig, cancelled)
        status, result, error = "failed", None, None
        try:
            current_data = dataset_manifest(self.policy["datasets"])
            if current_data != load_json(rd / "datasets.json"):
                raise RunIncomplete("Dataset changed after registration")
            (rd / "output").mkdir()
            shutil.copytree(rd / "source", rd / "train_work")
            shutil.copytree(rd / "source", rd / "eval_work")
            packages = sorted({f"{d.metadata.get('Name', 'unknown')}=={d.version}"
                               for d in importlib.metadata.distributions()})
            atomic_json(rd / "environment.json", {"python": sys.version, "executable": sys.executable,
                "platform": platform.platform(), "packages": packages,
                "allocated_gpus": self.policy["allocated_gpus"], "command_env_keys": sorted(self._environment(rd, run["protocol_hash"]))})
            env = self._environment(rd, run["protocol_hash"])
            env["KH_PURPOSE"] = run["purpose"]
            train_code = self._stage(run_id, "train", self.policy["train_command"], rd / "train_work", env, deadline)
            if train_code != 0:
                raise RuntimeError(f"Training exited with code {train_code}; inspect train.stderr.log")
            summary = self._training_checks(rd, run["proposal"])
            # Evaluate in a clean copy; training-side code writes do not become evaluator edits.
            expected_source = load_json(rd / "source_manifest.json")
            if tree_manifest(rd / "eval_work") != expected_source:
                raise RunIncomplete("Evaluation workspace changed before evaluation")
            eval_code = self._stage(run_id, "evaluate", self.policy["evaluate_command"], rd / "eval_work", env, deadline)
            if eval_code != 0:
                raise RuntimeError(f"Evaluation exited with code {eval_code}; inspect evaluate.stderr.log")
            if time.monotonic() >= deadline:
                raise RunTimedOut()
            result = self._evaluation_checks(rd, run)
            if dataset_manifest(self.policy["datasets"]) != current_data:
                raise RunIncomplete("Dataset changed during the experiment")
            self.verify(run_id, include_artifacts=False)
            if tree_manifest(rd / "eval_work") != expected_source:
                raise RunIncomplete("Evaluation changed its own source files")
            seals = {"evaluation.json": file_hash(rd / "evaluation.json")}
            # Keep arbitrary checkpoints, curves and auxiliary files, not just the winning model.
            for rel, checksum in tree_manifest(rd / "output").items():
                seals["output/" + rel] = checksum
            for name in ("train.stdout.log", "train.stderr.log", "evaluate.stdout.log", "evaluate.stderr.log"):
                seals[name] = file_hash(rd / name)
            result["training"] = summary
            result["sealed_files"] = seals
            result["allocated_gpu_seconds_estimate"] = (time.monotonic() - started) * self.policy["allocated_gpus"]
            status = "completed"
        except RunTimedOut:
            status, result, error = "timed_out", None, "Run exceeded its wall-clock reservation"
        except (RunCancelled, KeyboardInterrupt) as exc:
            status, result, error = "cancelled", None, str(exc) or "Interrupted by user"
        except (RunIncomplete, HarnessError, ValueError, KeyError, OSError) as exc:
            status, result, error = "incomplete", None, f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            status, result, error = "failed", None, f"{type(exc).__name__}: {exc}"
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        self._finish(run_id, status, result, error, time.monotonic() - started)
        self._observe(run_id)
        return self.store.get(run_id)

    def compare(self, left_id: str, right_id: str) -> dict:
        left, right = self.store.get(left_id), self.store.get(right_id)
        self.verify(left_id)
        self.verify(right_id)
        if any(r["status"] != "completed" for r in (left, right)):
            raise HarnessError("Only completed runs can be numerically compared")
        if any(r["purpose"] != "experiment" for r in (left, right)):
            raise HarnessError("Smoke tests are not formal experimental comparisons")
        if left["protocol_hash"] != right["protocol_hash"]:
            raise HarnessError("Incomparable validation/data/metric/seed protocols")
        a, b = left["result"]["value"], right["result"]["value"]
        gain = (b - a) if self.policy["metric"]["direction"] == "maximize" else (a - b)
        return {"parent": left_id, "candidate": right_id, "parent_value": a,
                "candidate_value": b, "improvement": gain,
                "passes_threshold": gain > self.policy["metric"]["min_improvement"],
                "config_diff": diff_configs(left["proposal"]["config"], right["proposal"]["config"]),
                "statistical_significance": "not_assessed"}

    def promote(self, run_id: str, reason: str) -> dict:
        if not reason.strip():
            raise HarnessError("Promotion requires an audit reason")
        self.verify(run_id)
        with self.store.transaction():
            run = self.store.get(run_id)
            if run["status"] != "completed" or run["purpose"] != "experiment":
                raise HarnessError("Only a complete formal experiment can become champion")
            current = self.store.champion()
            if current and current["run_id"] == run_id:
                raise HarnessError("Run is already the champion")
            comparison = self.compare(current["run_id"], run_id) if current else None
            if comparison and not comparison["passes_threshold"]:
                raise HarnessError("Candidate does not exceed the configured improvement threshold")
            old = current["run_id"] if current else None
            self.store.db.execute("""INSERT INTO champion(singleton,run_id,promoted_at,reason)
              VALUES(1,?,?,?) ON CONFLICT(singleton) DO UPDATE SET run_id=excluded.run_id,
              promoted_at=excluded.promoted_at,reason=excluded.reason""", (run_id, now(), reason))
            self.store.event("champion_promoted", {"previous": old, "reason": reason,
                                                   "comparison": comparison}, run_id)
        return {"champion": run_id, "previous": old, "comparison": comparison}

    def recover(self, stale_seconds: float = 30) -> list[dict]:
        """Recover ledger state, not model weights. Never silently rerun interrupted work."""
        finite_number(stale_seconds, "stale_seconds", minimum=0)
        recovered = []
        rows = self.store.db.execute("SELECT id FROM runs WHERE status IN ('running','preparing')").fetchall()
        for row in rows:
            run = self.store.get(row["id"])
            if run["host"] != socket.gethostname():
                continue
            if time.time() - (run["heartbeat"] or time.time()) < stale_seconds:
                continue
            if process_alive(run["owner_pid"], run["owner_token"]):
                continue
            child_pid = run["child_pid"]
            if process_alive(child_pid, run["child_token"]):
                if os.name != "posix" or run["child_token"] is None:
                    recovered.append({"run_id": run["id"], "action": "manual_child_cleanup_required"})
                    continue
                # Linux start token was verified above. No blind killing of reused PIDs.
                try:
                    os.killpg(child_pid, signal.SIGTERM)
                    time.sleep(0.1)
                    if process_alive(child_pid, run["child_token"]):
                        os.killpg(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elapsed = max(0, time.time() - run["started_at"]) if run["started_at"] else 0
            # Conservatively charge the whole reservation when the runner disappeared.
            charged = max(elapsed, run["timeout_seconds"]) if run["status"] == "running" else 0
            self._finish(run["id"], "interrupted", None,
                         "Controller disappeared; existing logs retained. Resume as a new child run.", charged)
            recovered.append({"run_id": run["id"], "action": "marked_interrupted",
                              "charged_seconds": charged})
        return recovered

    def context(self, tags: list[str] | None = None, evidence_ids: list[str] | None = None) -> dict:
        """No raw training logs, failed configs or archive paths in default model context."""
        tags, evidence_ids = tags or [], evidence_ids or []
        champion = self.store.champion()
        brief_champion = None
        if champion:
            run = champion["run_id"]
            self.verify(run)
            record = champion["run"]
            brief_champion = {"id": run, "config": record["proposal"]["config"],
                               "metric_value": record["result"]["value"],
                               "protocol_hash": record["protocol_hash"]}
        counts = {row[0]: row[1] for row in self.store.db.execute("SELECT status,COUNT(*) FROM runs GROUP BY status")}
        notes = []
        for row in self.store.db.execute("SELECT * FROM notes ORDER BY id DESC LIMIT 100"):
            note = dict(row)
            note["tags"] = json.loads(note.pop("tags_json"))
            if not tags or set(tags) & set(note["tags"]):
                notes.append(note)
            if len(notes) >= 12:
                break
        explicit = []
        for rid in evidence_ids[:10]:
            run = self.store.get(rid)
            explicit.append({"id": rid, "status": run["status"], "proposal": run["proposal"],
                             "result": run["result"], "error": run["error"]})
        from .research import compact
        return {"competition_id": self.policy["competition_id"], "champion": brief_champion,
                "budget": self.store.budget(self.policy), "status_counts": counts,
                "active_notes": notes, "explicitly_requested_evidence": explicit,
                "research_brief": compact(self),
                "archive_policy": "Raw logs and failed experiment details excluded by default"}

    def export(self, run_id: str, destination: Path) -> dict:
        self.verify(run_id)
        destination = destination.resolve()
        if destination.exists():
            raise HarnessError("Export refuses to overwrite an existing path")
        if is_within(destination, self.store.root) or is_within(destination, self.workspace):
            raise HarnessError("Export to a new directory outside the store and original workspace")
        rd = self.store.run_dir(run_id)
        destination.mkdir(parents=True)
        shutil.copytree(rd / "source", destination / "source")
        for name in ("requested_config.json", "proposal.json", "datasets.json", "source_manifest.json",
                     "environment.json", "evaluation.json", "outcome.json", "diff.json"):
            if (rd / name).is_file():
                shutil.copy2(rd / name, destination / name)
        if (rd / "output").exists():
            shutil.copytree(rd / "output", destination / "output")
        return {"run_id": run_id, "destination": str(destination),
                "data_copied": False, "note": "Dataset manifests included; original data bytes are external"}
