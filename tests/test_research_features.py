from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from kaggle_harness import integrity, kaggle, monitor, research
from kaggle_harness.cli import main
from kaggle_harness.codex import ask, cycle, example_proposal
from kaggle_harness.engine import Harness
from kaggle_harness.report import write_report
from kaggle_harness.util import HarnessError, atomic_json, dumps, load_json

ROOT = Path(__file__).resolve().parents[1]
FAKE_CLI = r'''
import json, os, sys, time
from pathlib import Path
a = sys.argv[1:]
if '--version' in a:
    print('Kaggle CLI fixture 2.2.4'); sys.exit(0)
if '--help' in a:
    print('--format --page-size --search --competition --content --path --timeout --accelerator --metadata')
    sys.exit(0)
if 'hang' in a:
    time.sleep(10)
if a[:1] == ['quota']:
    print(json.dumps({'remaining': 30, 'token': os.environ.get('KAGGLE_API_TOKEN','unset'),
                      'key': os.environ.get('KAGGLE_KEY','unset'),
                      'config_dir': os.environ.get('KAGGLE_CONFIG_DIR')})); sys.exit(0)
if a[:3] == ['competitions','topics','list']:
    assert a[3] == 'toy-competition'
elif a[:2] == ['competitions','topics']:
    raise SystemExit('wrong topic argument order')
if a[:2] == ['competitions','pages']:
    assert a[2] == '--competition' and a[3] == 'toy-competition' and '--content' in a
if a[:2] == ['competitions','files']:
    print('Next Page Token = fixture-cursor')
if a[:3] == ['competitions','topics','list']:
    print('Warning: page size is ignored')
if a[:2] == ['kernels','push']:
    folder = Path(a[a.index('--path')+1])
    assert json.loads((folder/'kernel-metadata.json').read_text())['is_private']
    assert '--timeout' in a
    if os.environ.get('KH_FIXTURE_FAIL'):
        print('uncertain transport', file=sys.stderr); sys.exit(1)
    print('Kernel version 7 successfully pushed.'); sys.exit(0)
if a[:2] == ['kernels','status']:
    print(a[2] + ' has status "complete"'); sys.exit(0)
if a[:2] == ['kernels','logs']:
    print('epoch 1 loss=0.3'); sys.exit(0)
if a[:2] == ['kernels','pull']:
    folder = Path(a[a.index('--path')+1]); (folder/'downloaded.py').write_text('print(1)\n')
    print('downloaded'); sys.exit(0)
print(json.dumps([{'ref':'owner/toy', 'arguments':a}]))
'''


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kh-features-")
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        shutil.copytree(ROOT / "examples" / "toy_competition", self.workspace)
        policy = load_json(ROOT / "examples" / "toy_policy.json")
        policy.update(max_run_seconds=30., total_wall_seconds=300.)
        self.h = Harness.initialize(self.workspace, self.root / "state", policy)

    def tearDown(self):
        self.h.close()
        self.temp.cleanup()

    def run_one(self, *, parent=None, formal=True, **config):
        p = example_proposal(self.h, formal=formal, parent_id=parent)
        p["config"].update(config)
        p["timeout_seconds"] = 10.
        p["hypothesis"] = "A controlled change tests convergence."
        p["expected_observation"] = "Complete recorded training and a fixed-protocol evaluation."
        if not formal:
            p["config"]["steps"] = 2
        rid = self.h.register(p)
        self.h.run(rid)
        return rid

    def annotation(self, family="learning-rate", verdict="keep", layer=None):
        return {"family": family, "operator": "improve", "verdict": verdict, "failure_layer": layer,
                "reason": "Observed under this protocol; does not establish cross-task effectiveness.",
                "sources": ["local-only"], "author": "test-reviewer"}

    def cli(self, **kwargs):
        script = self.root / "fake_kaggle.py"
        script.write_text(FAKE_CLI, encoding="utf-8")
        return kaggle.KaggleCLI([sys.executable, str(script)], **kwargs)

    def notebook(self):
        folder = self.root / "notebook"
        folder.mkdir()
        (folder / "main.py").write_text("print('a private CPU smoke test')\n", encoding="utf-8")
        atomic_json(folder / "kernel-metadata.json",
                    {"id": "owner/toy", "code_file": "main.py", "is_private": True})
        return folder


class ResearchTests(Fixture):
    def test_bounded_context_and_malformed_interpretation(self):
        record = {"kind": "research", "title": "A documented method", "summary": "s" * 3000,
                  "parents": [], "sources": ["local-only"], "author": "reviewer"}
        research.add_node(self.h, record, self.h.store.revision())
        brief = self.h.context()["research_brief"]
        self.assertEqual(len(brief["recent_research"][0]["summary"]), 1200)
        record["kind"] = {}
        with self.assertRaises(HarnessError):
            research.add_node(self.h, record, self.h.store.revision())
        bad = self.annotation()
        bad["operator"] = {}
        with self.assertRaises(HarnessError):
            research.annotate(self.h, "unknown", bad, self.h.store.revision())

    def test_graph_uses_actual_metrics_and_failure_facts(self):
        baseline = self.run_one()
        child = self.run_one(parent=baseline, learning_rate=.06)
        failed = self.run_one(parent=child, failure_mode="crash")
        nodes = {r["id"]: r for r in research.tree(self.h)["nodes"]}
        self.assertEqual(nodes[child]["parents"], [baseline])
        self.assertEqual(nodes[child]["metric"], self.h.store.get(child)["result"]["value"])
        self.assertEqual(nodes[failed]["status"], "failed")
        self.assertIsNone(nodes[failed]["metric"])

    def test_read_revision_and_immutable_research(self):
        revision = research.tree(self.h)["revision"]
        record = {"kind": "hypothesis", "title": "State/action interaction", "summary": "Preserve order.",
                  "parents": [], "sources": ["local-only"], "author": "reviewer"}
        node = research.add_node(self.h, record, revision)
        with self.assertRaisesRegex(HarnessError, "revision"):
            research.add_node(self.h, record, revision)
        with self.assertRaises(sqlite3.IntegrityError):
            self.h.store.db.execute("DELETE FROM research_nodes")
        record["parents"] = [node["id"]]
        self.assertTrue(research.add_node(self.h, record, research.tree(self.h)["revision"])["id"])
        record["parents"] = ["does-not-exist"]
        with self.assertRaisesRegex(HarnessError, "Unknown"):
            research.add_node(self.h, record, self.h.store.revision())

    def test_failure_layer_and_smoke_keep_guards(self):
        smoke = self.run_one(formal=False)
        with self.assertRaisesRegex(HarnessError, "formal"):
            research.annotate(self.h, smoke, self.annotation(), self.h.store.revision())
        with self.assertRaisesRegex(HarnessError, "failure_layer"):
            research.annotate(self.h, smoke, self.annotation(verdict="revert"), self.h.store.revision())
        research.annotate(self.h, smoke, self.annotation(verdict="revert", layer="execution"), self.h.store.revision())
        self.assertIsNone(self.h.store.champion())

    def test_selected_parent_sends_its_actual_source_through_portable_codex_transport(self):
        train = self.workspace / "train.py"
        train.write_text(train.read_text() + "\n# parent-baseline-marker\n", encoding="utf-8")
        baseline = self.run_one()
        p = example_proposal(self.h, parent_id=baseline)
        p["timeout_seconds"] = 10.
        p["config"]["learning_rate"] = .06
        p["edits"] = [{"path": "train.py", "content": train.read_text().replace("parent-baseline-marker", "parent-champion-marker")}]
        champion = self.h.register(p)
        self.assertEqual(self.h.run(champion)["status"], "completed")
        self.h.promote(champion, "Actual improvement")
        proposal = example_proposal(self.h, parent_id=baseline)
        envelope = {"decision": "experiment", "reason": "Source checked by a CLI fixture.",
                    "proposal_json": json.dumps(proposal)}
        fake = self.root / "source_codex.py"
        fake.write_text(
            "import sys,json\nfrom pathlib import Path\n"
            "if '--version' in sys.argv: print('fixture');sys.exit(0)\n"
            "if '--help' in sys.argv: print('--json --output-schema --output-last-message --sandbox --skip-git-repo-check --ignore-user-config --ephemeral --cd');sys.exit(0)\n"
            "sys.stdin.read()\n"
            "source=(Path(sys.argv[sys.argv.index('--cd')+1])/'train.py').read_text()\n"
            "assert 'parent-baseline-marker' in source and 'parent-champion-marker' not in source\n"
            f"Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text({json.dumps(json.dumps(envelope))})\n",
            encoding="utf-8")
        result = ask(self.h, "Inspect the separate source", parent_id=baseline,
                     binary=[sys.executable, str(fake)], timeout=10)
        self.assertEqual(result["status"], "validated")
        self.assertEqual(load_json(Path(result["proposal_path"]))["parent_id"], baseline)

    def test_selection_excludes_failure_smoke_and_other_protocol(self):
        baseline = self.run_one()
        best = self.run_one(parent=baseline, learning_rate=.06)
        self.h.promote(best, "A real fixed-protocol improvement")
        failed = self.run_one(parent=best, failure_mode="crash")
        smoke = self.run_one(formal=False)
        foreign = self.run_one(seed=99)
        selection = research.select(self.h)
        ids = {r["id"] for r in selection["ranked"]}
        self.assertEqual(ids, {baseline, best})
        self.assertTrue({failed, smoke, foreign}.isdisjoint(ids))
        self.assertEqual(selection["selected"], best)
        (self.h.store.run_dir(best) / "output" / "model.json").write_text("{}")
        selection = research.select(self.h)
        self.assertEqual(selection["selected"], baseline)
        self.assertEqual(selection["rejected"][0]["id"], best)

    def test_visits_cool_selection_and_monitor_does_not_invalidate_revision(self):
        baseline = self.run_one()
        self.assertEqual(research.select(self.h, record_visit=True)["selected"], baseline)
        self.assertEqual(research.tree(self.h)["nodes"][0]["visits"], 1)
        revision = self.h.store.revision()
        monitor.observe(self.h, baseline)
        self.assertEqual(self.h.store.revision(), revision)
        self.assertLess(research.select(self.h)["ranked"][0]["cooling"], 1)

    def test_explicit_parent_controls_codex_source_and_template(self):
        baseline = self.run_one()
        best = self.run_one(parent=baseline, learning_rate=.06)
        self.h.promote(best, "Actual improvement")
        proposal = example_proposal(self.h, parent_id=baseline)
        self.assertEqual(proposal["parent_id"], baseline)
        self.assertEqual(proposal["config"]["learning_rate"], .002)
        reply = ask(self.h, "Review a separate branch", parent_id=baseline, dry_run=True)
        context = load_json(Path(reply["directory"]) / "context.json")
        self.assertEqual(context["selected_parent"]["id"], baseline)
        self.assertEqual(context["champion"]["id"], best)
        with patch("kaggle_harness.codex.ask", return_value={"decision": "stop"}) as call:
            cycle(self.h, "Compare branches", steps=1, selection="balanced")
            self.assertIn(call.call_args.kwargs["parent_id"], {baseline, best})

    def test_replay_is_pure_and_cannot_peek_at_a_child_metric(self):
        baseline = self.run_one()
        self.run_one(parent=baseline, learning_rate=.06)
        revision = self.h.store.revision()
        a = research.replay(self.h, budget_seconds=20, steps=2)
        b = research.replay(self.h, budget_seconds=20, steps=2)
        self.assertEqual(a, b)
        self.assertEqual(revision, self.h.store.revision())
        self.assertEqual({p["policy"] for p in a["policies"]}, {"chronological", "greedy", "balanced"})
        first = research.replay(self.h, budget_seconds=10, steps=1)
        for policy in first["policies"]:
            self.assertEqual(policy["order"], [baseline])
            self.assertEqual(policy["best_value"], self.h.store.get(baseline)["result"]["value"])
        self.assertIn("Unrecorded", a["limitation"])

    def test_legacy_store_upgrade_retains_original_facts(self):
        original = self.h.store.revision()
        for table in ("monitor_state", "kaggle_jobs", "kaggle_accounts", "branch_visits",
                      "run_annotations", "research_nodes"):
            self.h.store.db.execute("DROP TABLE " + table)
        self.h.close()
        self.h = Harness(self.root / "state")
        self.assertEqual(self.h.store.revision(), original)
        self.assertEqual(research.tree(self.h)["nodes"], [])
        self.assertTrue(integrity.check(self.h)["ok"])


class MonitorAndIntegrityTests(Fixture):
    def test_completion_wins_over_old_error_log_patterns(self):
        rid = self.run_one()
        # Old text must not make a terminal completed run look active/failed.
        path = self.h.store.run_dir(rid) / "train.stderr.log"
        path.write_text("Traceback (most recent call last)\n")
        result = monitor.observe(self.h, rid)
        self.assertTrue(result["terminal"])
        self.assertEqual(result["alerts"], [])
        self.assertEqual(result["progress"]["step"], "40")
        self.assertFalse(integrity.check(self.h)["ok"])  # Sealed log tampering is still caught.

    def test_engine_attaches_monitor_and_retains_terminal_state(self):
        rid = self.run_one()
        state = self.h.store.db.execute("SELECT summary_json FROM monitor_state WHERE run_id=?", (rid,)).fetchone()
        self.assertEqual(json.loads(state[0])["status"], "completed")
        self.assertTrue(integrity.check(self.h)["ok"])
        self.assertEqual(len(list(monitor.watch(self.h, rid, max_polls=2))), 1)
        self.assertFalse(monitor.observe(self.h, rid)["alerts_changed"])

    def test_report_audit_catches_number_and_coverage_drift(self):
        self.run_one()
        result = write_report(self.h, self.root / "report.html")
        document = load_json(Path(result["data_path"]))
        self.assertTrue(integrity.audit_report(self.h, document)["ok"])
        document["runs"][0]["value"] += 1
        self.assertFalse(integrity.audit_report(self.h, document)["ok"])
        document["runs"] = []
        self.assertFalse(integrity.audit_report(self.h, document)["ok"])
        document["scientific_significance"] = "assessed"
        self.assertFalse(integrity.audit_report(self.h, document)["ok"])

    def test_ledger_metric_drift_is_detected_against_evaluator_file(self):
        rid = self.run_one()
        result = copy.deepcopy(self.h.store.get(rid)["result"])
        result["value"] += 100
        self.h.store.db.execute("UPDATE runs SET result_json=? WHERE id=?", (dumps(result), rid))
        self.assertFalse(integrity.check(self.h)["ok"])
        self.assertFalse(integrity.audit_report(self.h, integrity.report_data(self.h))["ok"])

    def test_cli_failure_exit_code_for_audit(self):
        atomic_json(self.root / "bad.json", {"schema_version": 1, "runs": []})
        with patch("sys.stdout"), patch("sys.stderr"):
            code = main(["--store", str(self.root / "state"), "audit-report", "--data", str(self.root / "bad.json")])
        self.assertEqual(code, 1)


class KaggleTests(Fixture):
    def test_cli_collection_registers_research_in_the_same_store(self):
        fixture = self.cli()
        cli_type = kaggle.KaggleCLI
        def factory(binary, **kwargs):
            return cli_type(fixture.prefix, **kwargs)
        with patch("kaggle_harness.kaggle.KaggleCLI", new=factory), patch("sys.stdout"):
            code = main(["--store", str(self.h.store.root), "kaggle", "collect", "toy-competition",
                         "--output", str(self.root / "cli-collected"), "--limit", "2"])
        self.assertEqual(code, 0)
        self.assertEqual(len(research.tree(self.h)["nodes"]), 1)
        self.assertEqual(self.h.context()["research_brief"]["recent_research"][0]["kind"], "research")
        self.assertTrue(integrity.check(self.h)["ok"])
        (self.root / "cli-collected" / "pages.json").write_text("{}")
        self.assertFalse(integrity.check(self.h)["ok"])

    def test_doctor_and_real_subprocess_query(self):
        cli = self.cli()
        doctor = cli.doctor()
        self.assertTrue(doctor["available"])
        self.assertTrue(all(v["available"] for v in doctor["capabilities"].values()))
        self.assertEqual(cli.query("quota")["data"]["remaining"], 30)
        self.assertTrue(cli.query("topics", competition="toy-competition")["ok"])
        self.assertTrue(cli.query("pages", competition="toy-competition")["ok"])
        result = cli.query("files", competition="toy-competition")
        self.assertIsInstance(result["data"], list)
        self.assertIn("fixture-cursor", result["notices"])

    def test_credentials_are_redacted_and_aliases_override_global_credentials(self):
        folder = self.root / "config"
        folder.mkdir()
        atomic_json(folder / "kaggle.json", {"username": "fixture", "key": "legacy-fixture-key"})
        with patch.dict(os.environ, {"KAGGLE_API_TOKEN": "KGAT_fixture_secret_123", "KAGGLE_KEY": "global-fixture-key"}):
            native = self.cli().query("quota")
            self.assertNotIn("KGAT_", dumps(native))
            self.assertNotIn("global-fixture-key", dumps(native))
            account = self.cli(config_dir=folder, account="fixture").query("quota")
            self.assertEqual(account["data"]["token"], "unset")
            self.assertEqual(account["data"]["key"], "unset")
            self.assertEqual(account["data"]["config_dir"], str(folder))
        kaggle.add_account(self.h, "fixture", folder, make_default=True)
        self.assertEqual(kaggle.account_directory(self.h, None), ("fixture", folder))
        self.assertNotIn("legacy-fixture-key", dumps(kaggle.accounts(self.h)))

    def test_query_guards_and_missing_capability(self):
        cli = self.cli()
        with self.assertRaises(HarnessError):
            cli.query("submit")
        with self.assertRaises(HarnessError):
            cli.query("files", competition="--danger")
        cli._help["quota"] = "--csv"
        with self.assertRaisesRegex(HarnessError, "required flags"):
            cli.query("quota")
        for bad in ("https://evil.test/competitions/x", None, "owner/x/../../"):
            with self.assertRaises(HarnessError):
                kaggle.reference(bad)

    def test_cli_timeout_is_enforced(self):
        cli = self.cli(timeout=.2)
        with self.assertRaisesRegex(HarnessError, "timed out"):
            cli.invoke(["hang"])

    def test_parallel_collection_preserves_sources_and_does_not_run_code(self):
        result = kaggle.collect(self.cli(), "toy-competition", self.root / "collection", limit=2)
        self.assertTrue(result["ok"])
        self.assertEqual(set(result["files"]), {"pages.json", "files.json", "topics.json", "kernels.json"})
        self.assertEqual(kaggle.accounts(self.h)["accounts"], [])
        self.assertTrue(integrity.check(self.h)["ok"])
        with self.assertRaises(HarnessError):
            kaggle.collect(self.cli(), "toy-competition", self.root / "collection")

    def test_notebook_pull_preserves_manifest_and_never_executes(self):
        result = kaggle.pull(self.cli(), "owner/toy", self.root / "pulled")
        self.assertTrue(result["ok"])
        self.assertFalse(result["executed"])
        self.assertIn("downloaded.py", result["files"])
        self.assertTrue((self.root / "pulled" / "download-manifest.json").is_file())

    def test_private_launch_preview_and_version_pinned_poll(self):
        cli, folder = self.cli(), self.notebook()
        preview = kaggle.launch(self.h, cli, folder)
        self.assertTrue(preview["dry_run"])
        self.assertEqual(self.h.store.db.execute("SELECT COUNT(*) FROM kaggle_jobs").fetchone()[0], 0)
        job = kaggle.launch(self.h, cli, folder, execute=True)
        self.assertEqual(job["status"], "queued", job)
        self.assertEqual(job["record"]["version_ref"], "owner/toy/7")
        with self.assertRaisesRegex(HarnessError, "active"):
            kaggle.launch(self.h, cli, folder, execute=True)
        with self.assertRaisesRegex(HarnessError, "active"):
            kaggle.launch(self.h, self.cli(account="another-alias"), folder, execute=True)
        polled = kaggle.poll(self.h, cli, job["job_id"])
        self.assertEqual(polled["status"], "complete")
        self.assertEqual(polled["scientific_result"], "not_imported")
        self.assertIsNone(self.h.store.champion())
        self.assertTrue(integrity.check(self.h)["ok"])
        self.assertEqual(research.tree(self.h)["nodes"][0]["kind"], "kaggle_job")

    def test_uncertain_remote_launch_cannot_be_retried(self):
        cli, folder = self.cli(), self.notebook()
        with patch.dict(os.environ, {"KH_FIXTURE_FAIL": "1"}):
            job = kaggle.launch(self.h, cli, folder, execute=True)
        self.assertEqual(job["status"], "unknown")
        self.assertEqual(kaggle.poll(self.h, cli, job["job_id"])["status"], "unknown")
        with self.assertRaisesRegex(HarnessError, "active"):
            kaggle.launch(self.h, cli, folder, execute=True)
        reconciled = kaggle.bind_version(self.h, cli, job["job_id"], 7, "Identified the private version manually")
        self.assertEqual(reconciled["status"], "complete")
        self.assertIsNone(self.h.store.champion())

    def test_upload_guards_and_snapshot_tamper(self):
        cli, folder = self.cli(), self.notebook()
        metadata = load_json(folder / "kernel-metadata.json")
        metadata["is_private"] = False
        atomic_json(folder / "kernel-metadata.json", metadata)
        with self.assertRaisesRegex(HarnessError, "private"):
            kaggle.launch(self.h, cli, folder, execute=True)
        metadata["is_private"] = True
        atomic_json(folder / "kernel-metadata.json", metadata)
        job = kaggle.launch(self.h, cli, folder, execute=True)
        source = self.h.store.root / "kaggle_jobs" / job["job_id"] / "source" / "main.py"
        source.write_text("changed")
        with self.assertRaisesRegex(HarnessError, "snapshot"):
            kaggle.poll(self.h, cli, job["job_id"])
        self.assertFalse(integrity.check(self.h)["ok"])

    def test_upload_rejects_credential_content_and_preserves_failed_identity(self):
        cli, folder = self.cli(), self.notebook()
        (folder / "main.py").write_text("token = 'KGAT_fixture_secret_123'\n")
        result = kaggle.launch(self.h, cli, folder, execute=True)
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("KGAT_", dumps(result))
        self.assertEqual(self.h.store.db.execute("SELECT COUNT(*) FROM kaggle_jobs").fetchone()[0], 1)
        self.assertIsNone(self.h.store.champion())
        copied = self.h.store.root / "kaggle_jobs" / result["job_id"] / "source" / "main.py"
        self.assertFalse(copied.exists())

    def test_large_upload_source_is_scanned_before_copy(self):
        cli, folder = self.cli(), self.notebook()
        (folder / "main.py").write_text("# " + "x" * 2_100_000 + "\ntoken='KGAT_fixture_secret_123'\n")
        result = kaggle.launch(self.h, cli, folder, execute=True)
        self.assertEqual(result["status"], "failed")
        self.assertFalse((self.h.store.root / "kaggle_jobs" / result["job_id"] / "source" / "main.py").exists())


if __name__ == "__main__":
    unittest.main()
