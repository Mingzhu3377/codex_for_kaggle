"""Bounded observations of local runs. Alerts never rewrite scientific results."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import re
import time

from .contracts import TERMINAL
from .util import digest, dumps, finite_number, now, process_alive, redact

PATTERNS = {
    "traceback": r"Traceback \(most recent call last\)",
    "out_of_memory": r"out of memory|CUDA.*allocat|MemoryError",
    "nonfinite": r"(?:loss|metric)\s*[=:]\s*(?:nan|inf)\b",
}


def tail(path: Path, size: int = 65536) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - size))
        data = f.read(size)
    return data.decode("utf-8", errors="replace")


def observe(h, run_id: str, *, stale_seconds: float = 60, persist: bool = True) -> dict:
    stale = finite_number(stale_seconds, "stale_seconds", minimum=0.1)
    run = h.store.get(run_id)
    rd = h.store.run_dir(run_id)
    logs = {name: tail(rd / name) for name in
            ("train.stdout.log", "train.stderr.log", "evaluate.stdout.log", "evaluate.stderr.log")}
    curves = tail(rd / "output" / "curves.csv")
    last = {}
    if curves:
        header = ""
        with (rd / "output" / "curves.csv").open(encoding="utf-8", errors="replace") as f:
            header = f.readline()
        # Only complete lines are progress evidence; a writer may be midway through a row.
        lines = curves.splitlines()
        if not curves.endswith(("\n", "\r")):
            lines = lines[:-1]
        if len(lines) > 1:
            try:
                parsed = list(csv.DictReader(io.StringIO(header + lines[-1] + "\n")))
                if parsed and parsed[0].get("step"):
                    last = {k: parsed[0].get(k) for k in ("step", "epoch", "train_loss", "learning_rate")}
            except (csv.Error, ValueError):
                pass
    fingerprint = digest({"status": run["status"], "logs": logs, "curve_tail": curves})
    previous = h.store.db.execute("SELECT * FROM monitor_state WHERE run_id=?", (run_id,)).fetchone()
    changed = previous is None or previous["fingerprint"] != fingerprint
    observed_at = time.time()
    changed_at = observed_at if changed else previous["changed_at"]
    terminal = run["status"] in TERMINAL
    alerts = []
    if not terminal:
        for name, pattern in PATTERNS.items():
            if any(re.search(pattern, body, re.I) for body in logs.values()):
                alerts.append({"kind": name, "severity": "warning",
                               "reason": "A log pattern matched; inspect the run before drawing a conclusion"})
        if run["status"] == "running":
            if not process_alive(run["owner_pid"], run["owner_token"]):
                alerts.append({"kind": "controller_missing", "severity": "error",
                               "reason": "Controller is not alive; use recover after verifying ownership"})
            elif run["heartbeat"] and observed_at - run["heartbeat"] > stale:
                alerts.append({"kind": "heartbeat_stale", "severity": "warning"})
            if observed_at - changed_at > stale:
                alerts.append({"kind": "no_log_progress", "severity": "info",
                               "reason": "Logs have not changed; this alone does not prove a hang"})
    previous_alerts = json.loads(previous["summary_json"]).get("alerts", []) if previous else []
    alerts_changed = alerts != previous_alerts
    result = {"run_id": run_id, "observed_at": now(), "status": run["status"], "terminal": terminal,
              "changed": changed, "progress": last, "seconds_without_log_change": max(0, observed_at - changed_at),
              "alerts": alerts, "alerts_changed": alerts_changed,
              "error": redact(run["error"] or "") if terminal else None}
    if persist:
        with h.store.transaction():
            h.store.db.execute("""INSERT INTO monitor_state VALUES(?,?,?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET fingerprint=excluded.fingerprint,
                changed_at=excluded.changed_at,observed_at=excluded.observed_at,summary_json=excluded.summary_json""",
                (run_id, fingerprint, changed_at, observed_at, dumps(result)))
            if changed or alerts_changed:
                h.store.event("monitor_observed", result, run_id)
    return result


def watch(h, run_id: str, *, interval: float = 2, max_polls: int = 30, stale_seconds: float = 60):
    from .util import HarnessError, integer
    wait = finite_number(interval, "interval", minimum=0.1)
    integer(max_polls, "max_polls", 1)
    if max_polls > 10000 or wait > 60:
        raise HarnessError("Use <=10000 polls and <=60 seconds per interval")
    for i in range(max_polls):
        result = observe(h, run_id, stale_seconds=stale_seconds)
        if result["changed"] or result["alerts_changed"] or result["terminal"]:
            yield result
        if result["terminal"]:
            return
        if i + 1 < max_polls:
            time.sleep(wait)
