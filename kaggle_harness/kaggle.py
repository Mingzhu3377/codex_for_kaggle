"""Kaggle CLI adapter. Accounts store config-directory references, never token copies."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import urlsplit
import uuid
from concurrent.futures import ThreadPoolExecutor

from .util import (HarnessError, atomic_json, copy_source, digest, dumps, file_hash,
                   finite_number, integer, is_within, load_json, now, redact,
                   safe_relative, terminate_process, tree_manifest)

CREDENTIAL_ENV = ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")
HELP_COMMANDS = {
    "quota": ["quota"],
    "competitions": ["competitions", "list"],
    "files": ["competitions", "files"],
    "pages": ["competitions", "pages"],
    "topics": ["competitions", "topics", "list"],
    "kernels": ["kernels", "list"],
    "status": ["kernels", "status"],
    "logs": ["kernels", "logs"],
    "push": ["kernels", "push"],
    "pull": ["kernels", "pull"],
}
JOB_ACTIVE = {"launching", "queued", "running", "unknown"}


def reference(value: str, kernel: bool = False) -> str:
    if not isinstance(value, str):
        raise HarnessError("Kaggle reference must be a string")
    if value.startswith("https://"):
        u = urlsplit(value)
        if u.hostname not in {"www.kaggle.com", "kaggle.com"}:
            raise HarnessError("Expected a Kaggle URL")
        prefix = "/code/" if kernel else "/competitions/"
        if not u.path.startswith(prefix):
            raise HarnessError(f"Expected a {prefix} URL")
        value = u.path[len(prefix):].rstrip("/")
    slug = r"[A-Za-z0-9][A-Za-z0-9_-]*"
    pattern = slug + "/" + slug + r"(?:/[0-9]+)?" if kernel else slug
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise HarnessError("Invalid Kaggle competition/kernel reference")
    return value


def accounts(h) -> dict:
    rows = [dict(r) for r in h.store.db.execute("SELECT name,config_dir,at FROM kaggle_accounts ORDER BY name")]
    row = h.store.db.execute("SELECT value FROM meta WHERE key='kaggle_default_account'").fetchone()
    return {"accounts": rows, "default": json.loads(row[0]) if row else None,
            "credentials": "References to existing Kaggle config directories; tokens are not copied"}


def add_account(h, name: str, config_dir: Path, *, make_default: bool = False) -> dict:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}", name):
        raise HarnessError("Account name must be a short alias")
    folder = config_dir.expanduser().resolve()
    if not folder.is_dir():
        raise HarnessError("Kaggle config directory must already exist")
    if is_within(folder, h.workspace) or is_within(folder, Path(__file__).parent):
        raise HarnessError("Keep Kaggle credentials outside competition/package source")
    with h.store.transaction():
        if h.store.db.execute("SELECT 1 FROM kaggle_accounts WHERE name=?", (name,)).fetchone():
            raise HarnessError("Account alias exists; choose a new alias")
        h.store.db.execute("INSERT INTO kaggle_accounts VALUES(?,?,?)", (name, str(folder), now()))
        if make_default:
            h.store.set_meta("kaggle_default_account", name)
        h.store.event("kaggle_account_added", {"name": name, "config_dir": str(folder)})
    return accounts(h)


def account_directory(h, name: str | None) -> tuple[str, Path | None]:
    selected = name or accounts(h)["default"]
    if not selected:
        return "environment", None
    row = h.store.db.execute("SELECT config_dir FROM kaggle_accounts WHERE name=?", (selected,)).fetchone()
    if row is None:
        raise HarnessError(f"Unknown account alias: {selected}")
    return selected, Path(row[0])


class KaggleCLI:
    def __init__(self, binary="kaggle", *, config_dir: Path | None = None,
                 account: str = "environment", timeout: float = 30):
        self.prefix = [binary] if isinstance(binary, str) else list(binary)
        if not self.prefix or any(not isinstance(x, str) or not x for x in self.prefix):
            raise HarnessError("Kaggle binary must be an executable or argv prefix")
        resolved = shutil.which(self.prefix[0])
        if not resolved:
            raise HarnessError(f"Kaggle CLI not found: {self.prefix[0]}")
        self.prefix[0] = resolved
        self.config_dir = config_dir.expanduser().resolve() if config_dir else None
        if self.config_dir and not self.config_dir.is_dir():
            raise HarnessError("Account config directory is unavailable")
        self.account = account
        self.timeout = finite_number(timeout, "CLI timeout", minimum=0.1)
        if self.timeout > 300:
            raise HarnessError("CLI timeout must be <=300 seconds")
        self._help = {}

    def environment(self) -> dict:
        env = os.environ.copy()
        if self.config_dir:
            for key in CREDENTIAL_ENV:
                env.pop(key, None)
            env["KAGGLE_CONFIG_DIR"] = str(self.config_dir)
        return env

    def _secrets(self) -> list[str]:
        env = self.environment()
        values = [env[k] for k in ("KAGGLE_API_TOKEN", "KAGGLE_KEY") if env.get(k)]
        folder = self.config_dir or Path(env.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle")))
        for name in ("kaggle.json", "access_token", "credentials.json"):
            path = folder / name
            if not path.is_file() or path.stat().st_size > 65536:
                continue
            try:
                text = path.read_text(encoding="utf-8").strip()
                if name == "access_token":
                    values.append(text)
                else:
                    doc = json.loads(text)
                    values.extend(str(doc[k]) for k in ("key", "token", "api_token", "access_token") if doc.get(k))
            except (OSError, ValueError):
                continue
        return values

    def invoke(self, args: list[str]) -> dict:
        options = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        started = time.monotonic()
        try:
            proc = subprocess.Popen(self.prefix + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    stdin=subprocess.DEVNULL, env=self.environment(), shell=False, **options)
            try:
                out, err = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                raise HarnessError(f"Kaggle CLI timed out after {self.timeout:g}s; remote outcome may be unknown") from exc
            finally:
                terminate_process(proc, grace=0.25)
        except OSError as exc:
            raise HarnessError(f"Cannot launch Kaggle CLI: {exc}") from exc
        secrets = self._secrets()
        stdout = redact(out.decode("utf-8", errors="replace"), secrets)
        stderr = redact(err.decode("utf-8", errors="replace"), secrets)
        if len(stdout) > 2_000_000:
            raise HarnessError("Kaggle CLI response exceeds 2 MB; request a smaller page/log")
        data = None
        notices = ""
        if stdout.strip():
            try:
                data = json.loads(stdout)
            except ValueError:
                # Kaggle 2.2.x can prepend pagination/warning lines even with --format json.
                decoder = json.JSONDecoder()
                for start in re.finditer(r"(?m)^[ \t]*(?=[{\[])", stdout):
                    try:
                        candidate = stdout[start.start():].lstrip()
                        data, end = decoder.raw_decode(candidate)
                        notices = stdout[:start.start()] + candidate[end:]
                        break
                    except ValueError:
                        continue
        return {"ok": proc.returncode == 0, "returncode": proc.returncode,
                "command": [redact(a, secrets) for a in args], "account": self.account, "elapsed_seconds": time.monotonic() - started,
                "data": data, "text": stdout if data is None else None,
                "notices": notices.strip()[:65536], "stderr": stderr[:65536]}

    def help(self, operation: str) -> str:
        if operation not in self._help:
            result = self.invoke(HELP_COMMANDS[operation] + ["--help"])
            if not result["ok"]:
                raise HarnessError(f"Kaggle CLI does not support {operation}: {result['stderr']}")
            self._help[operation] = (result["text"] or "") + result["stderr"]
        return self._help[operation]

    def require(self, operation: str, *flags: str):
        text = self.help(operation)
        missing = [f for f in flags if f not in text]
        if missing:
            raise HarnessError(f"Kaggle {operation} is missing required flags: {missing}")

    def doctor(self) -> dict:
        version = self.invoke(["--version"])
        capabilities, errors = {}, {}
        for name in HELP_COMMANDS:
            try:
                text = self.help(name)
                capabilities[name] = {"available": True, "json": "--format" in text}
            except HarnessError as exc:
                capabilities[name] = {"available": False}
                errors[name] = str(exc)
        return {"available": version["ok"], "version": (version["text"] or "").strip(),
                "executable": self.prefix[0], "capabilities": capabilities, "errors": errors,
                "authentication": "not_checked; a successful live quota/query verifies API access",
                "account": self.account}

    def query(self, operation: str, *, competition: str | None = None,
              ref: str | None = None, search: str | None = None, limit: int = 10) -> dict:
        integer(limit, "limit", 1)
        if limit > 200:
            raise HarnessError("limit must be <=200")
        if operation not in HELP_COMMANDS or operation in {"push", "pull"}:
            raise HarnessError("Unknown read-only Kaggle operation")
        args = list(HELP_COMMANDS[operation])
        if operation in {"files", "pages", "topics"}:
            if not competition:
                raise HarnessError("competition is required")
            slug = reference(competition)
            if operation == "topics":
                args = ["competitions", "topics", "list", slug]
            elif operation == "pages":
                # 2.2.4 duplicates an optional positional in the parent and list parser.
                # The supported competition_opt parameter avoids that ambiguity; root defaults to list.
                args = ["competitions", "pages", "--competition", slug]
            else:
                args += [slug]
        if operation in {"status", "logs"}:
            if not ref:
                raise HarnessError("kernel ref is required")
            args += [reference(ref, kernel=True)]
        if operation in {"competitions", "kernels"}:
            self.require(operation, "--page-size")
            args += ["--page-size", str(limit)]
            if search:
                self.require(operation, "--search")
                args += ["--search", search]
            if competition and operation == "kernels":
                self.require(operation, "--competition")
                args += ["--competition", reference(competition)]
        if operation in {"files", "topics"}:
            self.require(operation, "--page-size")
            args += ["--page-size", str(limit)]
        if operation == "pages":
            self.require(operation, "--content")
            args += ["--content"]
        if operation not in {"status", "logs"}:
            self.require(operation, "--format")
            args += ["--format", "json"]
        result = self.invoke(args)
        if operation in {"competitions", "kernels", "files", "topics"} and isinstance(result["data"], list):
            result["received_items"] = len(result["data"])
            result["truncated"] = len(result["data"]) > limit
            result["data"] = result["data"][:limit]
        return result


def pull(cli: KaggleCLI, ref: str, destination: Path) -> dict:
    cli.require("pull", "--path", "--metadata")
    ref = reference(ref, kernel=True)
    destination = destination.resolve()
    if destination.exists():
        raise HarnessError("Notebook pull requires a new output directory")
    destination.mkdir(parents=True)
    result = cli.invoke(["kernels", "pull", ref, "--path", str(destination), "--metadata"])
    manifest = tree_manifest(destination)
    atomic_json(destination / "download-manifest.json",
                {"ref": ref, "fetched_at": now(), "account": cli.account, "files": manifest, "response": result})
    return {"ok": result["ok"], "ref": ref, "output": str(destination), "files": manifest,
            "executed": False, "response": result}


def collect(cli: KaggleCLI, competition: str, destination: Path, *, limit: int = 10) -> dict:
    """Four bounded independent source queries; downloaded notebooks are a separate action."""
    slug = reference(competition)
    integer(limit, "limit", 1)
    if limit > 200:
        raise HarnessError("limit must be <=200")
    destination = destination.resolve()
    if destination.exists():
        raise HarnessError("Research collection requires a new output directory")
    destination.mkdir(parents=True)
    operations = ("pages", "files", "topics", "kernels")

    def query_one(operation):
        # Each worker owns its capability cache and process; SQLite is never shared across threads.
        worker = KaggleCLI(cli.prefix, config_dir=cli.config_dir, account=cli.account, timeout=cli.timeout)
        try:
            return worker.query(operation, competition=slug, limit=limit)
        except (HarnessError, OSError, ValueError) as exc:
            return {"ok": False, "error": redact(str(exc)), "operation": operation}

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = dict(zip(operations, pool.map(query_one, operations)))
    for operation, response in results.items():
        atomic_json(destination / f"{operation}.json", response)
    files = {f"{op}.json": file_hash(destination / f"{op}.json") for op in operations}
    summary = {"competition": slug, "fetched_at": now(), "account": cli.account,
               "ok": all(r["ok"] for r in results.values()),
               "coverage": {op: {"ok": r["ok"], "items": len(r["data"]) if isinstance(r.get("data"), list) else None}
                            for op, r in results.items()},
               "files": files, "output": str(destination),
               "limitations": "First bounded pages of indexes; notebook bodies and competition data are not downloaded or executed."}
    atomic_json(destination / "manifest.json", summary)
    return summary


def _job(h, job_id: str) -> dict:
    row = h.store.db.execute("SELECT * FROM kaggle_jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        raise HarnessError(f"Unknown Kaggle job: {job_id}")
    return {**dict(row), "record": json.loads(row["record_json"])}


def launch(h, cli: KaggleCLI, folder: Path, *, timeout_seconds: int = 3600,
           accelerator: str | None = None, execute: bool = False) -> dict:
    integer(timeout_seconds, "timeout_seconds", 1)
    if timeout_seconds > 43200:
        raise HarnessError("Kaggle notebook runtime must be <=43200 seconds")
    if accelerator not in {None, "none", "gpu", "tpu", "T4", "P100"}:
        raise HarnessError("Unsupported accelerator; use gpu/tpu or a documented device")
    cli.require("push", "--path", "--timeout")
    if accelerator:
        cli.require("push", "--accelerator")
    folder = folder.resolve()
    metadata = load_json(folder / "kernel-metadata.json")
    if not isinstance(metadata, dict):
        raise HarnessError("kernel-metadata.json must be an object")
    ref = reference(metadata.get("id", ""), kernel=True)
    code_file = safe_relative(metadata.get("code_file", ""))
    file_hash(folder / code_file)
    if metadata.get("is_private") is not True:
        raise HarnessError("This launch entry requires explicit is_private=true in kernel-metadata.json")
    args = ["kernels", "push", "--path", str(folder), "--timeout", str(timeout_seconds)]
    if accelerator:
        args += ["--accelerator", accelerator]
    if not execute:
        return {"dry_run": True, "ref": ref, "account": cli.account, "command": args,
                "message": "No notebook uploaded or started; add --execute for an actual private notebook run"}
    if is_within(h.store.root, folder) or is_within(folder, h.store.root):
        raise HarnessError("Notebook source and experiment store must be separate")
    job_id = "K-" + uuid.uuid4().hex[:16]
    record = {"timeout_seconds": timeout_seconds, "accelerator": accelerator, "source_folder": str(folder),
              "scientific_result": "not_imported; a remote complete status is not a formal local experiment"}
    with h.store.transaction():
        existing = h.store.db.execute(
            "SELECT id FROM kaggle_jobs WHERE ref=? AND status IN ('launching','queued','running','unknown')",
            (ref,)).fetchone()
        if existing:
            raise HarnessError(f"An active/uncertain job already exists for this ref: {existing[0]}")
        h.store.db.execute("INSERT INTO kaggle_jobs VALUES(?,?,?,?,?,?,?)",
                           (job_id, now(), now(), ref, cli.account, "launching", dumps(record)))
        h.store.event("kaggle_launch_registered", {"job_id": job_id, "ref": ref, "account": cli.account})
    # File system and SQL are not one transaction. A failed snapshot keeps its registered identity.
    try:
        destination = h.store.root / "kaggle_jobs" / job_id / "source"
        secrets = [s.encode("utf-8") for s in cli._secrets() if len(s) >= 8]

        def validate_upload_file(path):
            overlap = max([256] + [len(s) for s in secrets])
            previous = b""
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    body = previous + block
                    if any(s in body for s in secrets) or re.search(rb"KGAT_[A-Za-z0-9._-]+", body):
                        raise HarnessError(f"Credential-like content in upload source: {path.relative_to(folder)}")
                    previous = body[-overlap:]

        manifest = copy_source(folder, destination, [], [], h.policy["snapshot_max_bytes"],
                               validate_file=validate_upload_file)
        if code_file.as_posix() not in manifest or "kernel-metadata.json" not in manifest:
            raise HarnessError("Notebook metadata/code was excluded from the upload snapshot")
        atomic_json(destination.parent / "source_manifest.json", manifest)
        record["snapshot_hash"] = digest(manifest)
        args[3] = str(destination)
        result = cli.invoke(args)
        record["launch_response"] = result
        version = None
        if isinstance(result["data"], dict):
            version = result["data"].get("versionNumber")
        match = re.search(r"Kernel version\s+([0-9]+)\s+successfully pushed", result["text"] or "", re.I)
        if match:
            version = int(match[1])
        if isinstance(version, int) and not isinstance(version, bool) and version > 0:
            record["version_ref"] = ref.split("/")[0] + "/" + ref.split("/")[1] + "/" + str(version)
        status = "queued" if result["ok"] and record.get("version_ref") else "unknown"
    except (HarnessError, OSError) as exc:
        # An invoke timeout can mean the upload succeeded. Never retry such a ref automatically.
        status = "unknown" if "snapshot_hash" in record else "failed"
        record["error"] = redact(str(exc))
    with h.store.transaction():
        h.store.db.execute("UPDATE kaggle_jobs SET updated_at=?,status=?,record_json=? WHERE id=?",
                           (now(), status, dumps(record), job_id))
        h.store.event("kaggle_launch_finished", {"job_id": job_id, "status": status})
    return {"job_id": job_id, "ref": ref, "status": status, "record": record}


def poll(h, cli: KaggleCLI, job_id: str) -> dict:
    job = _job(h, job_id)
    if job["account"] != cli.account:
        raise HarnessError("Poll with the same account alias used to launch this job")
    record = job["record"]
    source = h.store.root / "kaggle_jobs" / job_id / "source"
    if "snapshot_hash" in record and digest(tree_manifest(source)) != record["snapshot_hash"]:
        raise HarnessError("Kaggle upload snapshot changed")
    result = cli.query("status", ref=record.get("version_ref", job["ref"]))
    text = ((result["text"] or "") + result["stderr"]).lower()
    # The CLI prints e.g. 'has status "complete"'. Never infer completion from log silence.
    match = re.search(r'has status\s*["\x27]?(complete|running|queued|error|cancelled|canceled)\b', text)
    status = match[1] if result["ok"] and match and record.get("version_ref") else "unknown"
    if status == "canceled":
        status = "cancelled"
    record["status_response"] = result
    record["last_poll"] = now()
    with h.store.transaction():
        h.store.db.execute("UPDATE kaggle_jobs SET updated_at=?,status=?,record_json=? WHERE id=?",
                           (now(), status, dumps(record), job_id))
        h.store.event("kaggle_job_polled", {"job_id": job_id, "status": status})
    return {"job_id": job_id, "ref": job["ref"], "status": status, "scientific_result": "not_imported",
            "response": result}


def bind_version(h, cli: KaggleCLI, job_id: str, version: int, reason: str) -> dict:
    """Explicitly reconcile an uncertain launch after the user identifies its remote version."""
    integer(version, "version", 1)
    if not isinstance(reason, str) or not reason.strip():
        raise HarnessError("A reason explaining how the remote version was identified is required")
    job = _job(h, job_id)
    if job["account"] != cli.account:
        raise HarnessError("Use the original account alias")
    record = job["record"]
    if record.get("version_ref") or not record.get("snapshot_hash"):
        raise HarnessError("Only an unpinned, snapshotted uncertain launch can be reconciled")
    if job["status"] != "unknown":
        raise HarnessError("This job is not an uncertain launch")
    ref = "/".join(job["ref"].split("/")[:2]) + "/" + str(version)
    result = cli.query("status", ref=ref)
    if not result["ok"]:
        raise HarnessError("The supplied remote version could not be read")
    record["version_ref"] = ref
    record["manual_version_binding"] = {"reason": reason.strip(), "at": now(),
                                        "scope": "Human-identified version; remote code equality is not certified"}
    with h.store.transaction():
        h.store.db.execute("UPDATE kaggle_jobs SET updated_at=?,record_json=? WHERE id=?",
                           (now(), dumps(record), job_id))
        h.store.event("kaggle_job_reconciled", {"job_id": job_id, **record["manual_version_binding"]})
    return poll(h, cli, job_id)
