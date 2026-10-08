"""Training-side API. Does not write the ledger or change the champion."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from .util import HarnessError, atomic_json, finite_number, integer, now


class Recorder:
    def __init__(self, output: Path, resolved_config: dict):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "curves.csv").exists():
            raise HarnessError("Recorder output already exists; never reuse a run directory")
        atomic_json(self.output / "resolved_config.json", resolved_config)
        self.stream = (self.output / "curves.csv").open("x", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.stream, fieldnames=[
            "step", "epoch", "train_loss", "validation_metric", "learning_rate", "at", "extra_json"])
        self.writer.writeheader()
        self.last_step = 0
        self.rows = 0
        self.finished = False
        self._sync()

    @classmethod
    def from_env(cls, resolved_config: dict) -> "Recorder":
        raw = os.environ.get("KH_OUTPUT_DIR")
        if not raw:
            raise HarnessError("KH_OUTPUT_DIR is unset; start training through the harness")
        return cls(Path(raw), resolved_config)

    def _sync(self) -> None:
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def log(self, *, step: int, train_loss: float, learning_rate: float,
            epoch: int | None = None, validation_metric: float | None = None,
            extra: dict[str, Any] | None = None) -> None:
        if self.finished:
            raise HarnessError("Cannot log after finish()")
        integer(step, "step", 1)
        if step <= self.last_step:
            raise HarnessError("Training steps must be strictly increasing")
        finite_number(train_loss, "train_loss")
        finite_number(learning_rate, "learning_rate", minimum=0)
        if epoch is not None:
            integer(epoch, "epoch", 0)
        if validation_metric is not None:
            finite_number(validation_metric, "validation_metric")
        self.writer.writerow({"step": step, "epoch": epoch, "train_loss": train_loss,
                              "validation_metric": validation_metric, "learning_rate": learning_rate,
                              "at": now(), "extra_json": json.dumps(extra or {}, allow_nan=False)})
        self.last_step = step
        self.rows += 1
        self._sync()

    def finish(self, *, stop_reason: str, details: str = "") -> None:
        if self.finished:
            raise HarnessError("finish() called twice")
        if stop_reason not in ("budget_complete", "early_stopping", "smoke_complete"):
            raise HarnessError("Unknown stop_reason")
        atomic_json(self.output / "training_summary.json", {
            "schema_version": 1, "finished": True, "completed_steps": self.last_step,
            "curve_rows": self.rows, "stop_reason": stop_reason, "details": details,
            "finished_at": now()})
        self.finished = True

    def close(self) -> None:
        if not self.stream.closed:
            self._sync()
            self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            atomic_json(self.output / "training_error.json", {
                "type": exc_type.__name__, "message": str(exc),
                "last_step": self.last_step, "at": now()})
        self.close()
