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
from .module_demo import run_module_demo
from .engine import Harness
from .report import write_report
from . import integrity, kaggle, monitor, research
from . import module_usage
from .modules import ModuleLibrary, card_template
from .util import HarnessError, atomic_json, dumps, file_hash, load_json


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kh", description="Evidence-first Kaggle experiment harness")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--store", type=Path, default=os.environ.get("KH_STORE"),
                   help="External experiment store; alternatively set KH_STORE")
    p.add_argument("--library", type=Path, default=os.environ.get("KH_MODULE_LIBRARY"),
                   help="Shared source-backed module library; selected explicitly")
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Initialize a NEW store outside the code workspace")
    init.add_argument("--workspace", type=Path, required=True)
    init.add_argument("--policy", type=Path, required=True)
    template = sub.add_parser("policy-template")
    template.add_argument("--output", type=Path, required=True)
    demo = sub.add_parser("demo", help="Run a real CPU demo plus fault injections")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--examples", type=Path)
    module_demo = sub.add_parser("module-demo", help="Run actual module variants, preflight, cases and cross-task reuse on CPU")
    module_demo.add_argument("--output", type=Path, required=True)
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
    audit = sub.add_parser("audit-report", help="Compare structured report data with the ledger/archive")
    audit.add_argument("--data", type=Path, required=True)
    check = sub.add_parser("check", help="Check SQLite, graph, snapshots, results and remote job manifests")
    check.add_argument("--shallow", action="store_true")
    tree_cmd = sub.add_parser("tree", help="Research/experiment graph derived from the single ledger")
    tree_cmd.add_argument("--output", type=Path)
    research_add = sub.add_parser("research-add")
    research_add.add_argument("--record", type=Path, required=True)
    research_add.add_argument("--revision", type=int, required=True)
    annotate = sub.add_parser("annotate", help="Append a method/failure interpretation; never changes metrics")
    annotate.add_argument("id")
    annotate.add_argument("--record", type=Path, required=True)
    annotate.add_argument("--revision", type=int, required=True)
    branch = sub.add_parser("branch-next")
    branch.add_argument("--reference")
    branch.add_argument("--record-visit", action="store_true")
    replay = sub.add_parser("replay", help="Compare recorded-tree exploration policies, without rerunning")
    replay.add_argument("--budget-seconds", type=float, required=True)
    replay.add_argument("--steps", type=int, default=100)
    replay.add_argument("--reference")
    mon = sub.add_parser("monitor", help="Observe local progress/errors without mutating experiment results")
    mon.add_argument("id")
    mon.add_argument("--watch", action="store_true")
    mon.add_argument("--interval", type=float, default=2)
    mon.add_argument("--max-polls", type=int, default=30)
    mon.add_argument("--stale-seconds", type=float, default=60)
    module = sub.add_parser("module", help="Manually maintained module source/variant/case library")
    modsub = module.add_subparsers(dest="module_action", required=True)
    for name in ("init", "attach", "template", "add", "list", "show", "verify", "check", "diff", "export", "import", "cases", "case-add", "uses", "bindings"):
        cmd = modsub.add_parser(name)
        if name == "template":
            cmd.add_argument("--output", type=Path, required=True)
        if name == "add":
            cmd.add_argument("--card", type=Path, required=True)
            cmd.add_argument("--folder", type=Path, required=True)
        if name in ("show", "verify", "export", "cases", "case-add", "uses", "bindings"):
            cmd.add_argument("id")
        if name == "list":
            cmd.add_argument("--family")
            cmd.add_argument("--tag")
            cmd.add_argument("--limit", type=int, default=100)
        if name == "diff":
            cmd.add_argument("parent")
            cmd.add_argument("candidate")
        if name == "export":
            cmd.add_argument("--to", type=Path, required=True)
        if name == "import":
            cmd.add_argument("--folder", type=Path, required=True)
        if name == "case-add":
            cmd.add_argument("--run", required=True)
            cmd.add_argument("--record", type=Path, required=True)
            cmd.add_argument("--baseline")
    kg = sub.add_parser("kaggle", help="Capability-checked Kaggle CLI; account aliases contain no tokens")
    kgsub = kg.add_subparsers(dest="kaggle_action", required=True)
    for name in ("doctor", "quota", "accounts", "account-add", "competitions", "files", "pages",
                 "topics", "kernels", "status", "logs", "launch", "poll", "jobs", "collect", "pull", "reconcile"):
        cmd = kgsub.add_parser(name)
        cmd.add_argument("--kaggle-bin", default="kaggle")
        cmd.add_argument("--account")
        cmd.add_argument("--cli-timeout", type=float, default=30)
        if name == "account-add":
            cmd.add_argument("name")
            cmd.add_argument("--config-dir", type=Path, required=True)
            cmd.add_argument("--default", action="store_true")
        if name in ("files", "pages", "topics"):
            cmd.add_argument("competition")
        if name in ("status", "logs"):
            cmd.add_argument("ref")
        if name in ("competitions", "kernels"):
            cmd.add_argument("--search")
            cmd.add_argument("--limit", type=int, default=10)
            if name == "kernels":
                cmd.add_argument("--competition")
        if name == "launch":
            cmd.add_argument("--folder", type=Path, required=True)
            cmd.add_argument("--timeout-seconds", type=int, default=3600)
            cmd.add_argument("--accelerator")
            cmd.add_argument("--execute", action="store_true")
        if name == "poll":
            cmd.add_argument("job_id")
        if name == "reconcile":
            cmd.add_argument("job_id")
            cmd.add_argument("--version", type=int, required=True)
            cmd.add_argument("--reason", required=True)
        if name == "collect":
            cmd.add_argument("competition")
            cmd.add_argument("--output", type=Path, required=True)
            cmd.add_argument("--limit", type=int, default=10)
        if name == "pull":
            cmd.add_argument("ref")
            cmd.add_argument("--output", type=Path, required=True)
    context = sub.add_parser("context", help="Export compact active evidence without raw archived logs")
    context.add_argument("--tag", action="append", default=[])
    context.add_argument("--evidence", action="append", default=[])
    context.add_argument("--module", action="append", default=[], help="Explicitly selected module version")
    note = sub.add_parser("note")
    note.add_argument("--run")
    note.add_argument("--text", required=True)
    note.add_argument("--tag", action="append", default=[])
    note.add_argument("--author", default="human")
    note.add_argument("--kind", choices=["interpretation", "hypothesis", "decision"], default="interpretation")
    logs = sub.add_parser("logs")
    logs.add_argument("id")
    logs.add_argument("--stage", choices=["train", "evaluate", "preflight"], default="train")
    logs.add_argument("--stderr", action="store_true")
    logs.add_argument("--tail", type=int, default=50)
    for name in ("ask", "cycle"):
        agent = sub.add_parser(name)
        agent.add_argument("--objective", required=True)
        agent.add_argument("--codex-bin", default="codex")
        agent.add_argument("--model")
        agent.add_argument("--timeout", type=float, default=600.)
        agent.add_argument("--module", action="append", default=[], help="Read this frozen module's actual source")
        if name == "ask":
            agent.add_argument("--dry-run", action="store_true")
            agent.add_argument("--tag", action="append", default=[])
            agent.add_argument("--evidence", action="append", default=[])
            agent.add_argument("--parent", help="Inspect/propose from this completed frozen parent")
        else:
            agent.add_argument("--steps", type=int, default=1)
            agent.add_argument("--execute", action="store_true", help="Explicitly authorize experiment execution")
            agent.add_argument("--promote", action="store_true", help="Explicitly enable gated auto-promotion")
            agent.add_argument("--selection", choices=["champion", "balanced"], default="champion")
    return p


def new_json(path: Path, value) -> None:
    if path.exists():
        raise HarnessError(f"Refusing to overwrite: {path}")
    atomic_json(path, value)


def dispatch(args):
    c = args.command
    if c == "module-demo":
        return run_module_demo(args.output)
    if c == "module":
        action = args.module_action
        if action == "template":
            new_json(args.output, card_template())
            return {"path": str(args.output.resolve()), "executed": False}
        if action == "bindings":
            if not args.store:
                raise HarnessError("module bindings requires --store")
            with Harness(Path(args.store)) as h:
                h.verify(args.id)
                return module_usage.bindings(h, args.id)
        if not args.library:
            raise HarnessError("module requires --library PATH or KH_MODULE_LIBRARY")
        if action == "init":
            with ModuleLibrary(args.library, create=True) as lib:
                return {"library": str(lib.root), "status": "initialized"}
        if action in ("attach", "case-add"):
            if not args.store:
                raise HarnessError("This module operation requires --store")
            with Harness(Path(args.store)) as h:
                if action == "attach":
                    return module_usage.attach(h, args.library)
                with ModuleLibrary(args.library) as lib:
                    return lib.add_case(h, args.id, load_json(args.record), run_id=args.run, baseline_id=args.baseline)
        with ModuleLibrary(args.library) as lib:
            if action == "add":
                return lib.add(load_json(args.card), args.folder)
            if action == "list":
                return lib.list(family=args.family, tag=args.tag, limit=args.limit)
            if action == "show":
                record = lib.verify(args.id)
                return {**record, "source_directory": str(lib.source(args.id))}
            if action == "verify":
                r = lib.verify(args.id)
                return {"ok": True, "module_id": r["id"], "content_hash": r["content_hash"], "checks": r["static_checks"]}
            if action == "check":
                return lib.check()
            if action == "diff":
                return lib.diff(args.parent, args.candidate)
            if action == "export":
                return lib.export(args.id, args.to)
            if action == "import":
                return lib.import_bundle(args.folder)
            if action == "cases":
                return lib.cases(args.id)
            if action == "uses":
                return lib.uses(args.id)
        raise HarnessError("Unknown module operation")
    if c == "doctor":
        return doctor(args.codex_bin)
    if c == "demo":
        return run_demo(args.output, args.examples)
    if c == "policy-template":
        new_json(args.output, DEFAULT_POLICY)
        return {"path": str(args.output.resolve())}
    if c == "kaggle":
        h = Harness(Path(args.store)) if args.store else None
        try:
            action = args.kaggle_action
            if action in ("accounts", "account-add", "jobs", "poll", "reconcile") or (action == "launch" and args.execute):
                if h is None:
                    raise HarnessError("This Kaggle operation requires --store")
            if action == "accounts":
                return kaggle.accounts(h)
            if action == "account-add":
                return kaggle.add_account(h, args.name, args.config_dir, make_default=args.default)
            if action == "jobs":
                return [kaggle._job(h, r[0]) for r in h.store.db.execute("SELECT id FROM kaggle_jobs ORDER BY at,id")]
            if args.account and h is None:
                raise HarnessError("--account requires an initialized store with account aliases")
            account, directory = kaggle.account_directory(h, args.account) if h else ("environment", None)
            cli = kaggle.KaggleCLI(args.kaggle_bin, config_dir=directory, account=account, timeout=args.cli_timeout)
            if action == "doctor":
                return cli.doctor()
            if action == "launch":
                return kaggle.launch(h, cli, args.folder, timeout_seconds=args.timeout_seconds,
                                     accelerator=args.accelerator, execute=args.execute)
            if action == "poll":
                return kaggle.poll(h, cli, args.job_id)
            if action == "reconcile":
                return kaggle.bind_version(h, cli, args.job_id, args.version, args.reason)
            if action == "collect":
                result = kaggle.collect(cli, args.competition, args.output, limit=args.limit)
                if h:
                    record = {"kind": "research", "title": "Kaggle source collection",
                              "summary": dumps(result["coverage"]) + "; " + result["limitations"],
                              "parents": [], "sources": [str(args.output.resolve() / "manifest.json"),
                                                        f"https://www.kaggle.com/competitions/{result['competition']}"],
                              "evidence_files": [{"path": str(args.output.resolve() / name),
                                                  "sha256": file_hash(args.output.resolve() / name)}
                                                 for name in [*result["files"], "manifest.json"]],
                              "author": "collector"}
                    result["research_node"] = research.add_node(h, record, h.store.revision())
                return result
            if action == "pull":
                return kaggle.pull(cli, args.ref, args.output)
            return cli.query(action, competition=getattr(args, "competition", None),
                             ref=getattr(args, "ref", None), search=getattr(args, "search", None),
                             limit=getattr(args, "limit", 10))
        finally:
            if h:
                h.close()
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
        if c == "check":
            return integrity.check(h, deep=not args.shallow)
        if c == "audit-report":
            return integrity.audit_report(h, load_json(args.data))
        if c == "tree":
            result = research.tree(h)
            if args.output:
                new_json(args.output, result)
            return result
        if c == "research-add":
            return research.add_node(h, load_json(args.record), args.revision)
        if c == "annotate":
            return research.annotate(h, args.id, load_json(args.record), args.revision)
        if c == "branch-next":
            return research.select(h, args.reference, record_visit=args.record_visit)
        if c == "replay":
            return research.replay(h, budget_seconds=args.budget_seconds, steps=args.steps, reference=args.reference)
        if c == "monitor":
            if args.watch:
                for observation in monitor.watch(h, args.id, interval=args.interval, max_polls=args.max_polls,
                                                 stale_seconds=args.stale_seconds):
                    print(json.dumps(observation, ensure_ascii=False, allow_nan=False), flush=True)
                return {"watch": "finished", "run_id": args.id}
            return monitor.observe(h, args.id, stale_seconds=args.stale_seconds)
        if c == "context":
            context = h.context(args.tag, args.evidence)
            context["selected_modules"] = module_usage.selected_context(h, args.module)
            return context
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
                       dry_run=args.dry_run, tags=args.tag, evidence_ids=args.evidence, parent_id=args.parent,
                       module_ids=args.module)
        if c == "cycle":
            if args.promote and not args.execute:
                raise HarnessError("--promote requires --execute")
            return cycle(h, args.objective, steps=args.steps, execute=args.execute,
                         auto_promote=args.promote, binary=args.codex_bin, model=args.model, timeout=args.timeout,
                         selection=args.selection, module_ids=args.module)
        raise HarnessError(f"Unknown command: {c}")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if isinstance(result, dict) and result.get("ok") is False:
            return 1
        if args.command == "kaggle" and isinstance(result, dict):
            if result.get("available") is False or result.get("status") in ("unknown", "failed", "error"):
                return 1
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
