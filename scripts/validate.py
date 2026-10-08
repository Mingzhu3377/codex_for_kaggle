"""Run the offline suite and save a machine-readable local report. No network/Codex call."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import platform
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "validation-local")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; use a new one to preserve validation history")
    args.output.mkdir(parents=True)
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    started = time.monotonic()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (args.output / "test-output.txt").write_text(log.getvalue(), encoding="utf-8")
    summary = {"tests_run": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "skipped": len(result.skipped),
               "successful": result.wasSuccessful(), "wall_seconds": time.monotonic() - started,
               "python": sys.version, "platform": platform.platform(),
               "codex_live_tested": False, "gpu_training_tested": False,
               "test_failures": [{"test": str(t), "traceback": tb} for t, tb in result.failures + result.errors]}
    (args.output / "test-results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
