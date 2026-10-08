from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any


class HarnessError(Exception):
    """An actionable validation or operational error."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def load_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant: {value}")
    def unique_pairs(pairs: list) -> dict:
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"Duplicate JSON key: {key}")
            out[key] = value
        return out
    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          parse_constant=reject_constant, object_pairs_hook=unique_pairs)
    except (OSError, ValueError) as exc:
        raise HarnessError(f"Cannot read JSON {path}: {exc}") from exc


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def digest(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise HarnessError(f"Expected regular file, not symlink: {path}")
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise HarnessError(f"File changed during hashing: {path}")
    return h.hexdigest()


def finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise HarnessError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise HarnessError(f"{name} must be >= {minimum}")
    return float(value)


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HarnessError(f"{name} must be an integer >= {minimum}")
    return value


def safe_relative(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise HarnessError(f"Invalid portable relative path: {value!r}")
    p = Path(value)
    if p.is_absolute() or any(x in ("..", ".") for x in value.split("/")):
        raise HarnessError(f"Unsafe relative path: {value!r}")
    if any(not x for x in value.split("/")):
        raise HarnessError(f"Invalid relative path: {value!r}")
    return p


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
                 "node_modules", ".codex", ".agents", ".harness", "dist", "build", ".idea"}
SECRET_PATTERNS = (".env", ".env.*", "*.pem", "*.key", "auth.json", "kaggle.json",
                   "credentials.json", "*secret*", "*.sqlite*", "*.db", "*.pyc")


def copy_source(source: Path, dest: Path, excluded: list[Path], extra_excludes: list[str],
                max_bytes: int) -> dict[str, str]:
    """Copy real bytes, including uncommitted changes; never follow links."""
    if not source.is_dir() or source.is_symlink():
        raise HarnessError(f"Not a regular source directory: {source}")
    dest.mkdir(parents=True, exist_ok=False)
    total = 0
    manifest = {}
    for base, dirs, files in os.walk(source, followlinks=False):
        relbase = Path(base).relative_to(source)
        kept = []
        for name in sorted(dirs):
            p = Path(base) / name
            rel = (relbase / name).as_posix()
            if name in EXCLUDED_DIRS or any(fnmatch.fnmatch(rel, g) for g in extra_excludes):
                continue
            if any(p.resolve() == e.resolve() for e in excluded):
                continue
            if p.is_symlink():
                raise HarnessError(f"Source symlink rejected: {p}")
            kept.append(name)
        dirs[:] = kept
        for name in sorted(files):
            p = Path(base) / name
            rel = (relbase / name).as_posix()
            if any(fnmatch.fnmatch(name, g) for g in SECRET_PATTERNS):
                continue
            if any(fnmatch.fnmatch(rel, g) for g in extra_excludes):
                continue
            if any(p.resolve() == e.resolve() for e in excluded):
                continue
            if p.is_symlink() or not p.is_file():
                raise HarnessError(f"Source symlink/non-regular file rejected: {p}")
            total += p.stat().st_size
            if total > max_bytes:
                raise HarnessError("Source snapshot too large. Register data under policy.datasets; "
                                   "exclude checkpoints with policy.snapshot_exclude.")
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            manifest[rel] = file_hash(target)
            if file_hash(p) != manifest[rel]:
                raise HarnessError(f"Source changed while copied: {p}; retry from a stable workspace")
    return manifest


def tree_manifest(root: Path) -> dict[str, str]:
    result = {}
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            if (Path(base) / name).is_symlink():
                raise HarnessError(f"Directory symlink rejected: {Path(base) / name}")
        for name in sorted(files):
            p = Path(base) / name
            result[p.relative_to(root).as_posix()] = file_hash(p)
    return result


def dataset_manifest(datasets: dict[str, str]) -> dict:
    result = {}
    for name, raw in sorted(datasets.items()):
        p = Path(raw)
        if p.is_symlink() or not p.exists():
            raise HarnessError(f"Missing/symlink dataset {name}: {p}")
        result[name] = {"path": str(p), "files": tree_manifest(p) if p.is_dir()
                        else {p.name: file_hash(p)}}
    return result


def dataset_digest(manifest: dict) -> str:
    # Absolute paths are provenance, not dataset identity.
    return digest({k: v["files"] for k, v in manifest.items()})


def git_info(workspace: Path) -> dict:
    result = {}
    for label, args in (("commit", ["rev-parse", "HEAD"]),
                        ("status", ["status", "--porcelain"]),
                        ("diff_stat", ["diff", "--stat", "HEAD"])):
        try:
            p = subprocess.run(["git", "-C", str(workspace), *args], capture_output=True,
                               text=True, errors="replace", timeout=8)
            result[label] = p.stdout if p.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            result[label] = None
    return result


def process_token(pid: int) -> str | None:
    if sys.platform.startswith("linux"):
        try:
            text = Path(f"/proc/{pid}/stat").read_text()
            return text[text.rfind(")") + 2:].split()[19]  # field 22, start time
        except (OSError, IndexError):
            return None
    return None


def process_alive(pid: int | None, token: str | None = None) -> bool:
    if not pid or pid < 1:
        return False
    if token is not None and process_token(pid) != token:
        return False
    if sys.platform.startswith("linux"):
        try:
            text = Path(f"/proc/{pid}/stat").read_text()
            if text[text.rfind(")") + 2:].split()[0] == "Z":
                return False
        except OSError:
            return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied: fail closed.
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def terminate_process(proc: subprocess.Popen, grace: float = 1.5) -> None:
    """Terminate a normal local process tree (not a hostile sandbox escape)."""
    if os.name == "nt":
        if proc.poll() is None:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    end = time.monotonic() + grace
    while time.monotonic() < end:
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def owner_info() -> dict:
    return {"owner_pid": os.getpid(), "owner_token": process_token(os.getpid()),
            "host": socket.gethostname()}
