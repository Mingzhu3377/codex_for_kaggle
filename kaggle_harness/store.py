from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
from typing import Iterator

from .util import HarnessError, dumps, now

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs(
 id TEXT PRIMARY KEY, parent_id TEXT REFERENCES runs(id), purpose TEXT NOT NULL,
 status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 proposal_json TEXT NOT NULL, snapshot_hash TEXT, protocol_hash TEXT,
 started_at REAL, ended_at REAL, elapsed_seconds REAL NOT NULL DEFAULT 0,
 timeout_seconds REAL NOT NULL, owner_pid INTEGER, owner_token TEXT, host TEXT,
 child_pid INTEGER, child_token TEXT, heartbeat REAL,
 result_json TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS events(
 seq INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL,
 run_id TEXT REFERENCES runs(id), kind TEXT NOT NULL, payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS champion(
 singleton INTEGER PRIMARY KEY CHECK(singleton=1), run_id TEXT NOT NULL REFERENCES runs(id),
 promoted_at TEXT NOT NULL, reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notes(
 id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL,
 run_id TEXT REFERENCES runs(id), kind TEXT NOT NULL,
 summary TEXT NOT NULL, tags_json TEXT NOT NULL, author TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
 BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
 BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS notes_no_update BEFORE UPDATE ON notes
 BEGIN SELECT RAISE(ABORT, 'notes are append-only; supersede with a new note'); END;
CREATE TRIGGER IF NOT EXISTS notes_no_delete BEFORE DELETE ON notes
 BEGIN SELECT RAISE(ABORT, 'notes are append-only'); END;
CREATE TRIGGER IF NOT EXISTS runs_identity_immutable BEFORE UPDATE OF id,parent_id,purpose,proposal_json,created_at ON runs
 BEGIN SELECT RAISE(ABORT, 'experiment identity is immutable'); END;
CREATE TRIGGER IF NOT EXISTS runs_no_delete BEFORE DELETE ON runs
 BEGIN SELECT RAISE(ABORT, 'experiment records cannot be deleted through the ledger'); END;
"""


class Store:
    def __init__(self, root: Path, create: bool = False):
        self.root = root.resolve()
        if create:
            self.root.mkdir(parents=True, exist_ok=False)
            (self.root / "runs").mkdir()
            (self.root / "agent_sessions").mkdir()
        elif not (self.root / "ledger.sqlite3").is_file():
            raise HarnessError(f"Not an initialized harness store: {self.root}")
        self.db = sqlite3.connect(self.root / "ledger.sqlite3", timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=30000")
        if create:
            self.db.executescript(SCHEMA)
            self.set_meta("schema_version", 1)
        elif self.get_meta("schema_version") != 1:
            raise HarnessError("Unsupported ledger version")

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield self.db
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        else:
            self.db.execute("COMMIT")

    def event(self, kind: str, payload: dict, run_id: str | None = None) -> None:
        self.db.execute("INSERT INTO events(at,run_id,kind,payload_json) VALUES(?,?,?,?)",
                        (now(), run_id, kind, dumps(payload)))

    def set_meta(self, key: str, value) -> None:
        self.db.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, dumps(value)))

    def get_meta(self, key: str):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            raise HarnessError(f"Missing store metadata: {key}")
        return json.loads(row[0])

    @staticmethod
    def decode(row) -> dict:
        item = dict(row)
        item["proposal"] = json.loads(item.pop("proposal_json"))
        value = item.pop("result_json")
        item["result"] = json.loads(value) if value else None
        return item

    def get(self, run_id: str) -> dict:
        row = self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise HarnessError(f"Unknown run: {run_id}")
        return self.decode(row)

    def rows(self, limit: int = 100) -> list[dict]:
        return [self.decode(r) for r in self.db.execute(
            "SELECT * FROM runs ORDER BY created_at DESC, id DESC LIMIT ?", (limit,))]

    def run_dir(self, run_id: str) -> Path:
        self.get(run_id)  # Do not resolve an arbitrary path from user/model text.
        return self.root / "runs" / run_id

    def budget(self, policy: dict) -> dict:
        row = self.db.execute("""SELECT
          COALESCE(SUM(CASE WHEN status='running' THEN timeout_seconds ELSE elapsed_seconds END),0),
          SUM(CASE WHEN status='running' THEN 1 ELSE 0 END)
          FROM runs""").fetchone()
        committed, active = float(row[0]), int(row[1] or 0)
        return {"total_wall_seconds": policy["total_wall_seconds"],
                "consumed_or_reserved_seconds": committed,
                "remaining_seconds": max(0., policy["total_wall_seconds"] - committed),
                "active_runs": active}

    def champion(self) -> dict | None:
        row = self.db.execute("SELECT * FROM champion WHERE singleton=1").fetchone()
        if row is None:
            return None
        result = dict(row)
        result["run"] = self.get(row["run_id"])
        return result

    def note(self, run_id: str | None, summary: str, tags: list[str], author: str,
             kind: str = "interpretation") -> int:
        if run_id is not None:
            self.get(run_id)
        if kind not in ("interpretation", "hypothesis", "decision"):
            raise HarnessError("note kind must be interpretation, hypothesis or decision")
        if not isinstance(summary, str) or not 1 <= len(summary.strip()) <= 1200:
            raise HarnessError("Active note must be 1..1200 characters; keep full logs in the archive")
        if any(not isinstance(t, str) or not t for t in tags):
            raise HarnessError("tags must be nonempty strings")
        with self.transaction():
            cur = self.db.execute("INSERT INTO notes(at,run_id,kind,summary,tags_json,author) VALUES(?,?,?,?,?,?)",
                                  (now(), run_id, kind, summary, dumps(tags), author))
            self.event("note_added", {"note_id": cur.lastrowid, "kind": kind}, run_id)
        return int(cur.lastrowid)
