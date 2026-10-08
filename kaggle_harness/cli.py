from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys

from . import __version__
from .codex import ask, cycle, doctor, example_proposal
from .contracts import DEFAULT_POLICY
from .demo import run_demo
from .engine import Harness
from .report import write_report
from .util import HarnessError, atomic_json, load_json


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kh", description="Evidence-first Kaggle experiment harness")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--store", type=Path, default=os.environ.get("KH_STORE"),
                   help="External experiment store; alternatively set KH_STORE")
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Initialize a NEW store outside the code workspace")
    init.add_argument("--workspace", type=Path, required=True)
    init.add_argument("--policy", type=Path, required=True)
    template = sub.add_parser("policy-template")
    template.add_argument("--output", type=Path, required=True)
    demo = sub.add_parser("demo", help="Run a real CPU demo plus fault injections")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--examples", type=Path)
    doc = sub.add_parser("doctor", help="Check optional Codex CLI capabilities")
    doc.add_argument("--codex-bin", default="codex")
    prop = sub.add_parser("proposal", help="Write an editable proposal; does not start training")
    prop.add_argument("--output", type=Path, required=True)
    prop.add_argument("--smoke", action="store_true")
    reg = sub.add_parser("submit", help="Register and snapshot a proposal; does not start training")
    reg.add_argument("--proposal", type=Path, required=True)
    run = sub.add_parser("run", help="Run one registered or newly supplied proposal")
    choice = run.add_mutually_exclusive_group(required=True)
    choice.add_argument("--id")
    choice.add_argument("--proposal", type=Path)
    ls = sub.add_parser("list")
    ls.add_argument("--limit", type=int, default=100)
    show = sub.add_parser("show", help="Explicitly inspect a full archived experiment record")
    show.add_argument("id")
    diff = sub.add_parser("compare")
    diff.add_argument("parent")
    diff.add_argument("candidate")
    promote = sub.add_parser("promote")
    promote.add_argument("id")
    promote.add_argument("--reason", default="")
    promote.add_argument("--yes", action="store_true", help="Actually update champion, subject to gates")
    sub.add_parser("champion")
    verify = sub.add_parser("verify")
    verify.add_argument("id")
    recover = sub.add_parser("recover")
    recover.add_argument("--stale-seconds", type=float, default=30.)
    export = sub.add_parser("export", help="Restore snapshot/config/artifacts into a NEW directory")
    export.add_argument("id")
    export.add_argument("--to", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("--output", type=Path, required=True)
    context = sub.add_parser("context", help="Export compact active evidence without raw archived logs")
    context.add_argument("--tag", action="append", default=[])
    context.add_argument("--evidence", action="append", default=[])
    note = sub.add_parser("note")
    note.add_argument("--run")
    note.add_argument("--text", required=True)
    note.add_argument("--tag", action="append", default=[])
    note.add_argument("--author", default="human")
    note.add_argument("--kind", choices=["interpretation", "hypothesis", "decision"], default="interpretation")
    logs = sub.add_parser("logs")
    logs.add_argument("id")
    logs.add_argument("--stage", choices=["train", "evaluate"], default="train")
    logs.add_argument("--stderr", action="store_true")
    logs.add_argument("--tail", type=int, default=50)
    for name in ("ask", "cycle"):
        agent = sub.add_parser(name)
        agent.add_argument("--objective", required=True)
        agent.add_argument("--codex-bin", default="codex")
        agent.add_argument("--model")
        agent.add_argument("--timeout", type=float, default=600.)
        if name == "ask":
            agent.add_argument("--dry-run", action="store_true")
            agent.add_argument("--tag", action="append", default=[])
            agent.add_argument("--evidence", action="append", default=[])
        else:
            agent.add_argument("--steps", type=int, default=1)
            agent.add_argument("--execute", action="store_true", help="Explicitly authorize experiment execution")
            agent.add_argument("--promote", action="store_true", help="Explicitly enable gated auto-promotion")
    return p


def new_json(path: Path, value) -> None:
    if path.exists():
        raise HarnessError(f"Refusing to overwrite: {path}")
    atomic_json(path, value)


def dispatch(args):
    c = args.command
    if c == "doctor":
        return doctor(args.codex_bin)
    if c == "demo":
        return run_demo(args.output, args.examples)
    if c == "policy-template":
        new_json(args.output, DEFAULT_POLICY)
        return {"path": str(args.output.resolve())}
    if not args.store:
        raise HarnessError("Pass --store PATH before the subcommand, or set KH_STORE")
    if c == "init":
        with Harness.initialize(args.workspace, Path(args.store), load_json(args.policy)) as h:
            return {"store": str(h.store.root), "workspace": str(h.workspace), "status": "initialized"}
    with Harness(Path(args.store)) as h:
        if c == "proposal":
            p = example_proposal(h, formal=not args.smoke)
            if args.smoke:
                p["config"][h.policy["training_budget_key"]] = 2
            new_json(args.output, p)
            return {"proposal": str(args.output.resolve()), "executed": False}
        if c == "submit":
            rid = h.register(load_json(args.proposal))
            return {"run_id": rid, "status": "ready", "executed": False}
        if c == "run":
            rid = args.id or h.register(load_json(args.proposal))
            return h.run(rid)
        if c == "list":
            if not 1 <= args.limit <= 10000:
                raise HarnessError("limit must be in 1..10000")
            return [{"id": r["id"], "parent_id": r["parent_id"], "purpose": r["purpose"],
                     "status": r["status"], "value": r["result"]["value"] if r["result"] else None,
                     "elapsed_seconds": r["elapsed_seconds"]} for r in h.store.rows(args.limit)]
        if c == "show":
            return h.store.get(args.id)
        if c == "compare":
            return h.compare(args.parent, args.candidate)
        if c == "promote":
            if args.yes:
                return h.promote(args.id, args.reason)
            h.verify(args.id)
            run = h.store.get(args.id)
            if run["status"] != "completed" or run["purpose"] != "experiment":
                raise HarnessError("Not eligible for promotion")
            current = h.store.champion()
            return {"dry_run": True, "candidate": args.id,
                    "comparison": h.compare(current["run_id"], args.id) if current else "first complete baseline",
                    "message": "Add --yes --reason TEXT to promote; no change made"}
        if c == "champion":
            return h.store.champion()
        if c == "verify":
            return h.verify(args.id)
        if c == "recover":
            return h.recover(args.stale_seconds)
        if c == "export":
            return h.export(args.id, args.to)
        if c == "report":
            return write_report(h, args.output)
        if c == "context":
            return h.context(args.tag, args.evidence)
        if c == "note":
            return {"note_id": h.store.note(args.run, args.text, args.tag, args.author, args.kind)}
        if c == "logs":
            if not 1 <= args.tail <= 10000:
                raise HarnessError("tail must be in 1..10000")
            name = f"{args.stage}.{'stderr' if args.stderr else 'stdout'}.log"
            path = h.store.run_dir(args.id) / name
            with path.open(encoding="utf-8", errors="replace") as stream:
                return {"run_id": args.id, "log": name, "tail": "".join(deque(stream, maxlen=args.tail))}
        if c == "ask":
            return ask(h, args.objective, binary=args.codex_bin, model=args.model, timeout=args.timeout,
                       dry_run=args.dry_run, tags=args.tag, evidence_ids=args.evidence)
        if c == "cycle":
            if args.promote and not args.execute:
                raise HarnessError("--promote requires --execute")
            return cycle(h, args.objective, steps=args.steps, execute=args.execute,
                         auto_promote=args.promote, binary=args.codex_bin, model=args.model, timeout=args.timeout)
        raise HarnessError(f"Unknown command: {c}")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if args.command == "run" and result["status"] != "completed":
            return 1
        if args.command == "cycle" and any(x.get("status") not in (None, "completed") for x in result):
            return 1
        return 0
    except (HarnessError, OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; inspect the ledger before restarting.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
