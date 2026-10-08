from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from kaggle_harness.codex import ask, cycle, doctor, example_proposal
from kaggle_harness.contracts import validate_proposal
from kaggle_harness.engine import Harness
from kaggle_harness.recorder import Recorder
from kaggle_harness.util import HarnessError, atomic_json, load_json, process_alive, safe_relative

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kh-test-")
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        shutil.copytree(EXAMPLES / "toy_competition", self.workspace)
        policy = load_json(EXAMPLES / "toy_policy.json")
        policy["max_run_seconds"] = 80.
        policy["total_wall_seconds"] = 300.
        self.h = Harness.initialize(self.workspace, self.root / "state", policy)

    def tearDown(self):
        self.h.close()
        self.temp.cleanup()

    def proposal(self, **config):
        p = example_proposal(self.h)
        p["hypothesis"] = "Test that an explicit controlled configuration is evaluated correctly."
        p["expected_observation"] = "A complete trace or an explicit non-success status."
        p["timeout_seconds"] = 10.
        p["config"].update(config)
        return p

    def run_config(self, **config):
        rid = self.h.register(self.proposal(**config))
        result = self.h.run(rid)
        return rid, result

    def completed(self, **config):
        rid, run = self.run_config(**config)
        self.assertEqual(run["status"], "completed", run.get("error"))
        return rid

    def test_real_training_and_all_trace_files(self):
        rid = self.completed()
        rd = self.h.store.run_dir(rid)
        for rel in ["requested_config.json", "proposal.json", "source_manifest.json", "datasets.json",
                    "environment.json", "git.json", "diff.json", "train.stdout.log", "train.stderr.log",
                    "evaluate.stdout.log", "evaluate.stderr.log", "evaluation.json", "outcome.json",
                    "output/curves.csv", "output/resolved_config.json", "output/training_summary.json",
                    "output/model.json", "output/cleaning.json"]:
            self.assertTrue((rd / rel).is_file(), rel)
        cleaning = load_json(rd / "output/cleaning.json")
        self.assertEqual(cleaning["rejected"], 3)
        with (rd / "output/curves.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 40)

    def test_smoke_test_cannot_be_champion(self):
        p = self.proposal(steps=2)
        p["purpose"] = "smoke_test"
        rid = self.h.register(p)
        self.assertEqual(self.h.run(rid)["status"], "completed")
        with self.assertRaises(HarnessError):
            self.h.promote(rid, "not eligible")
        self.assertIsNone(self.h.store.champion())

    def test_formal_two_step_proposal_rejected_before_execution(self):
        with self.assertRaisesRegex(HarnessError, "min_formal_steps"):
            self.h.register(self.proposal(steps=2))
        self.assertEqual(self.h.store.rows(), [])

    def test_formal_short_run_is_incomplete(self):
        rid, run = self.run_config(failure_mode="short")
        self.assertEqual(run["status"], "incomplete")
        self.assertIn("min_formal_steps", run["error"])
        self.assertTrue((self.h.store.run_dir(rid) / "output/curves.csv").exists())

    def test_crash_preserves_partial_curve_and_stderr(self):
        rid, run = self.run_config(failure_mode="crash")
        self.assertEqual(run["status"], "failed")
        rd = self.h.store.run_dir(rid)
        self.assertIn("Intentional test failure", (rd / "train.stderr.log").read_text())
        with (rd / "output/curves.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_timeout_is_recorded_and_child_stopped(self):
        p = self.proposal(failure_mode="timeout")
        p["timeout_seconds"] = .6
        rid = self.h.register(p)
        before = time.monotonic()
        run = self.h.run(rid)
        self.assertEqual(run["status"], "timed_out")
        self.assertLess(time.monotonic() - before, 8)
        self.assertIsNone(run["child_pid"])

    def test_runtime_config_drift_rejected(self):
        _, run = self.run_config(failure_mode="silent_default")
        self.assertEqual(run["status"], "incomplete")
        self.assertIn("resolved config differs", run["error"])

    def test_no_curve_or_completion_is_incomplete(self):
        p = self.proposal()
        p["edits"] = [{"path": "train.py", "content": "print('I am done; score is great')\n"}]
        rid = self.h.register(p)
        self.assertEqual(self.h.run(rid)["status"], "incomplete")

    def test_promotion_requires_real_improvement(self):
        a = self.completed()
        self.h.promote(a, "baseline")
        b = self.completed(learning_rate=.06)
        comparison = self.h.compare(a, b)
        self.assertTrue(comparison["passes_threshold"])
        self.h.promote(b, "controlled improvement")
        self.assertEqual(self.h.store.champion()["run_id"], b)
        with self.assertRaises(HarnessError):
            self.h.promote(a, "worse rollback must fail")
        self.assertEqual(self.h.store.champion()["run_id"], b)

    def test_failed_run_cannot_replace_champion(self):
        a = self.completed()
        self.h.promote(a, "baseline")
        b, _ = self.run_config(failure_mode="crash")
        with self.assertRaises(HarnessError):
            self.h.promote(b, "bad")
        self.assertEqual(self.h.store.champion()["run_id"], a)

    def test_repeated_execution_of_same_id_rejected(self):
        rid = self.completed()
        with self.assertRaisesRegex(HarnessError, "Cannot start"):
            self.h.run(rid)

    def test_parent_source_survives_workspace_changes(self):
        a = self.completed()
        self.h.promote(a, "baseline")
        (self.workspace / "train.py").write_text("raise RuntimeError('broken workspace')\n")
        b = self.completed(learning_rate=.06)
        self.assertEqual(self.h.store.get(b)["parent_id"], a)
        self.assertNotIn("broken workspace", (self.h.store.run_dir(b) / "source/train.py").read_text())

    def test_code_edits_are_snapshotted_and_diffed(self):
        a = self.completed()
        self.h.promote(a, "baseline")
        source = (self.workspace / "train.py").read_text()
        p = self.proposal()
        p["edits"] = [{"path": "train.py", "content": source + "\n# reviewed candidate edit\n"}]
        b = self.h.register(p)
        diff = load_json(self.h.store.run_dir(b) / "diff.json")
        self.assertEqual(diff["code"]["modified"], ["train.py"])
        self.assertNotIn("reviewed candidate edit", (self.workspace / "train.py").read_text())

    def test_uncommitted_workspace_bytes_are_copied(self):
        source = self.workspace / "extra.py"
        source.write_text("new_uncommitted_value = 42\n")
        rid = self.h.register(self.proposal())
        self.assertEqual((self.h.store.run_dir(rid) / "source/extra.py").read_text(), source.read_text())

    def test_protected_evaluator_edit_rejected(self):
        p = self.proposal()
        p["edits"] = [{"path": "evaluate.py", "content": "print(999)"}]
        with self.assertRaisesRegex(HarnessError, "protected"):
            self.h.register(p)

    def test_workspace_evaluator_drift_retains_failed_registration(self):
        (self.workspace / "evaluate.py").write_text("print('wrong metric')")
        with self.assertRaises(HarnessError):
            self.h.register(self.proposal())
        rows = self.h.store.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertIn("Protected", rows[0]["error"])

    def test_path_traversal_edit_rejected(self):
        p = self.proposal()
        p["edits"] = [{"path": "../../ledger.py", "content": "oops"}]
        with self.assertRaises(HarnessError):
            self.h.register(p)

    def test_unapproved_new_file_rejected(self):
        p = self.proposal()
        p["edits"] = [{"path": "kaggle_harness/engine.py", "content": "oops"}]
        with self.assertRaises(HarnessError):
            self.h.register(p)

    @unittest.skipUnless(os.name == "posix", "POSIX symlink test")
    def test_source_symlink_rejected(self):
        (self.workspace / "linked.py").symlink_to(self.workspace / "train.py")
        with self.assertRaisesRegex(HarnessError, "symlink"):
            self.h.register(self.proposal())

    def test_known_secret_files_excluded_from_snapshot(self):
        for name in (".env", "auth.json", "kaggle.json", "private.key"):
            (self.workspace / name).write_text("not-for-model-context")
        rid = self.h.register(self.proposal())
        source = self.h.store.run_dir(rid) / "source"
        for name in (".env", "auth.json", "kaggle.json", "private.key"):
            self.assertFalse((source / name).exists())

    @unittest.skipUnless(shutil.which("git"), "Git is optional")
    def test_git_metadata_does_not_capture_secret_patch_contents(self):
        subprocess.run(["git", "init", "-q", str(self.workspace)], check=True)
        secret = self.workspace / ".env"
        secret.write_text("TOKEN=old-placeholder\n")
        subprocess.run(["git", "-C", str(self.workspace), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.workspace), "-c", "user.name=Harness Test",
                        "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
        secret.write_text("TOKEN=PRIVATE-VALUE-NOT-FOR-LOGS\n")
        rid = self.h.register(self.proposal())
        provenance = (self.h.store.run_dir(rid) / "git.json").read_text()
        self.assertNotIn("PRIVATE-VALUE-NOT-FOR-LOGS", provenance)
        self.assertIn("diff_stat", provenance)

    def test_dataset_change_after_registration_rejected(self):
        rid = self.h.register(self.proposal())
        with (self.workspace / "data/train.csv").open("a") as stream:
            stream.write("1,2\n")
        run = self.h.run(rid)
        self.assertEqual(run["status"], "incomplete")
        self.assertIn("Dataset changed", run["error"])

    def test_dataset_change_before_registration_keeps_failed_record(self):
        (self.workspace / "data/train.csv").write_text("x,y\n1,2\n")
        with self.assertRaises(HarnessError):
            self.h.register(self.proposal())
        self.assertEqual(self.h.store.rows()[0]["status"], "failed")

    def test_artifact_tampering_detected(self):
        rid = self.completed()
        (self.h.store.run_dir(rid) / "output/model.json").write_text("{}")
        with self.assertRaisesRegex(HarnessError, "Artifact integrity"):
            self.h.verify(rid)
        with self.assertRaises(HarnessError):
            self.h.promote(rid, "tampered")

    def test_source_tampering_detected_before_execution(self):
        rid = self.h.register(self.proposal())
        (self.h.store.run_dir(rid) / "source/train.py").write_text("print('tampered')")
        with self.assertRaises(HarnessError):
            self.h.run(rid)
        self.assertEqual(self.h.store.get(rid)["status"], "ready")

    def test_different_seed_protocols_not_compared(self):
        a = self.completed(seed=17)
        b = self.completed(seed=18)
        with self.assertRaisesRegex(HarnessError, "Incomparable"):
            self.h.compare(a, b)

    def test_default_context_excludes_failed_configs_and_logs(self):
        _, run = self.run_config(failure_mode="crash")
        text = json.dumps(self.h.context())
        self.assertNotIn("failure_mode", text)
        self.assertNotIn("Intentional test failure", text)
        self.assertNotIn("train.stderr.log", text)
        explicit = self.h.context(evidence_ids=[run["id"]])
        self.assertIn("failure_mode", json.dumps(explicit))

    def test_compact_note_and_tag_retrieval(self):
        a = self.completed()
        self.h.store.note(a, "Under-convergence remains a plausible explanation.", ["optimizer"], "human")
        self.assertEqual(len(self.h.context(tags=["optimizer"])["active_notes"]), 1)
        self.assertEqual(len(self.h.context(tags=["augmentation"])["active_notes"]), 0)
        with self.assertRaises(HarnessError):
            self.h.store.note(a, "x" * 1201, [], "human")

    def test_no_validated_claim_without_evidence(self):
        p = self.proposal()
        p["decisions"][0]["origin"] = "validated"
        with self.assertRaisesRegex(HarnessError, "cite experimental evidence"):
            self.h.register(p)

    def test_failed_evidence_cannot_be_labeled_validated(self):
        rid, _ = self.run_config(failure_mode="crash")
        p = self.proposal()
        p["decisions"][0].update(origin="validated", evidence_ids=[rid])
        with self.assertRaisesRegex(HarnessError, "completed formal"):
            self.h.register(p)

    def test_nonfinite_parameter_rejected(self):
        with self.assertRaises(HarnessError):
            self.h.register(self.proposal(learning_rate=float("nan")))

    def test_boolean_step_budget_rejected(self):
        with self.assertRaises(HarnessError):
            self.h.register(self.proposal(steps=True))

    def test_missing_decision_rejected(self):
        p = self.proposal()
        p["decisions"] = []
        with self.assertRaisesRegex(HarnessError, "explicit decision"):
            self.h.register(p)

    def test_unknown_proposal_field_rejected(self):
        p = self.proposal()
        p["skip_logging"] = True
        with self.assertRaises(HarnessError):
            self.h.register(p)

    def test_budget_reservation_is_enforced(self):
        rid = self.h.register(self.proposal())
        self.h.policy["total_wall_seconds"] = 1
        with self.assertRaisesRegex(HarnessError, "Insufficient"):
            self.h.run(rid)
        self.assertEqual(self.h.store.get(rid)["status"], "ready")

    def test_atomic_claim_prevents_duplicate_execution(self):
        rid = self.h.register(self.proposal())
        state = self.h.store.root
        def claim_once(_):
            with Harness(state) as other:
                try:
                    other._claim(rid)
                    return "claimed"
                except HarnessError:
                    return "rejected"
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(claim_once, range(2)))
        self.assertEqual(sorted(outcomes), ["claimed", "rejected"])

    def test_concurrency_limit_is_transactional(self):
        a = self.h.register(self.proposal())
        b = self.h.register(self.proposal())
        self.h._claim(a)
        with self.assertRaisesRegex(HarnessError, "Concurrency"):
            self.h._claim(b)

    def test_event_and_identity_records_are_append_only(self):
        rid = self.h.register(self.proposal())
        with self.assertRaises(sqlite3.IntegrityError):
            self.h.store.db.execute("DELETE FROM events")
        with self.assertRaises(sqlite3.IntegrityError):
            self.h.store.db.execute("UPDATE runs SET proposal_json='{}' WHERE id=?", (rid,))
        with self.assertRaises(sqlite3.IntegrityError):
            self.h.store.db.execute("DELETE FROM runs WHERE id=?", (rid,))

    def test_live_controller_is_not_recovered(self):
        rid = self.h.register(self.proposal())
        self.h._claim(rid)
        self.h.store.db.execute("UPDATE runs SET heartbeat=0 WHERE id=?", (rid,))
        self.assertEqual(self.h.recover(0), [])
        self.assertEqual(self.h.store.get(rid)["status"], "running")

    def test_stale_dead_controller_is_marked_interrupted(self):
        rid = self.h.register(self.proposal())
        self.h._claim(rid)
        self.h.store.db.execute("UPDATE runs SET owner_pid=99999999,owner_token=NULL,heartbeat=1 WHERE id=?", (rid,))
        result = self.h.recover(0)
        self.assertEqual(result[0]["action"], "marked_interrupted")
        self.assertEqual(self.h.store.get(rid)["status"], "interrupted")
        self.assertGreaterEqual(self.h.store.get(rid)["elapsed_seconds"], 10)
        self.assertTrue((self.h.store.run_dir(rid) / "proposal.json").exists())

    def test_export_restores_into_new_directory_only(self):
        rid = self.completed()
        dest = self.root / "restored"
        self.h.export(rid, dest)
        self.assertTrue((dest / "output/model.json").is_file())
        self.assertTrue((dest / "source/train.py").is_file())
        self.assertFalse((dest / "source/data/train.csv").exists())
        with self.assertRaises(HarnessError):
            self.h.export(rid, dest)

    def test_store_must_be_outside_workspace(self):
        with self.assertRaisesRegex(HarnessError, "disjoint"):
            Harness.initialize(self.workspace, self.workspace / "state", load_json(EXAMPLES / "toy_policy.json"))

    def test_credentials_not_forwarded_to_training_environment(self):
        rid = self.h.register(self.proposal())
        old = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "DO_NOT_FORWARD_TEST_VALUE"
        try:
            env = self.h._environment(self.h.store.run_dir(rid), "test-protocol")
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("CODEX_API_KEY", env)
        finally:
            if old is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = old

    def test_codex_dry_run_requires_no_codex_installation(self):
        reply = ask(self.h, "Inspect learning rate evidence", binary="not-installed-codex", dry_run=True)
        self.assertEqual(reply["status"], "dry_run")
        self.assertTrue((Path(reply["directory"]) / "prompt.txt").exists())
        self.assertEqual(self.h.store.rows(), [])

    def test_codex_doctor_missing_executable(self):
        self.assertFalse(doctor("not-installed-codex")["available"])

    def _fake_codex(self, malformed: bool = False):
        fake = self.root / "fake-codex"
        p = self.proposal()
        envelope = {"decision": "experiment", "reason": "Mock adapter test, not a live model.",
                    "proposal_json": json.dumps(p)}
        if malformed:
            envelope["proposal_json"] = '{"skip_logging":true}'
        code = f'''#!{sys.executable}
import json, sys
from pathlib import Path
if '--version' in sys.argv:
    print('codex MOCK for tests'); sys.exit(0)
if '--help' in sys.argv:
    print('--json --output-schema --output-last-message --sandbox --skip-git-repo-check --ignore-user-config --ephemeral --cd');sys.exit(0)
prompt = sys.stdin.read()
assert 'CURRENT EVIDENCE' in prompt
assert 'read-only' in sys.argv
Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text({json.dumps(json.dumps(envelope))})
print(json.dumps({{"type":"turn.completed", "test_double":True}}))
'''
        fake.write_text(code)
        fake.chmod(0o755)
        return str(fake)

    @unittest.skipUnless(os.name == "posix", "Executable fixture uses a POSIX shebang")
    def test_mock_codex_cli_to_validated_proposal(self):
        answer = ask(self.h, "Propose one controlled experiment", binary=self._fake_codex(), timeout=10)
        self.assertEqual(answer["status"], "validated")
        self.assertEqual(answer["decision"], "experiment")
        self.assertEqual(self.h.store.rows(), [])  # ask never executes or registers training.
        p = load_json(Path(answer["proposal_path"]))
        self.assertEqual(p["config"]["optimizer"], "sgd")
        self.assertTrue((Path(answer["directory"]) / "events.jsonl").exists())

    @unittest.skipUnless(os.name == "posix", "Executable fixture uses a POSIX shebang")
    def test_mock_invalid_codex_reply_is_archived_and_rejected(self):
        with self.assertRaises(HarnessError):
            ask(self.h, "Propose one experiment", binary=self._fake_codex(malformed=True), timeout=10)
        sessions = list((self.h.store.root / "agent_sessions").iterdir())
        self.assertEqual(len(sessions), 1)
        self.assertTrue((sessions[0] / "raw_reply.json").exists())
        self.assertEqual(load_json(sessions[0] / "outcome.json")["status"], "failed")
        self.assertEqual(self.h.store.rows(), [])

    @unittest.skipUnless(os.name == "posix", "Executable fixture uses a POSIX shebang")
    def test_mock_cycle_without_execute_only_proposes(self):
        history = cycle(self.h, "Controlled learning-rate study", steps=3, binary=self._fake_codex())
        self.assertEqual(len(history), 1)
        self.assertEqual(self.h.store.rows(), [])

    @unittest.skipUnless(os.name == "posix", "Executable fixture uses a POSIX shebang")
    def test_mock_cycle_executes_and_gates_promotion(self):
        history = cycle(self.h, "Establish a reproducible baseline", steps=1, execute=True,
                        auto_promote=True, binary=self._fake_codex(), timeout=10)
        self.assertEqual(history[0]["status"], "completed")
        self.assertEqual(self.h.store.champion()["run_id"], history[0]["run_id"])

    def test_installed_demo_assets_match_source_examples(self):
        from kaggle_harness.util import tree_manifest
        self.assertEqual(tree_manifest(EXAMPLES / "toy_competition"),
                         tree_manifest(ROOT / "kaggle_harness/assets/toy_competition"))
        self.assertEqual(load_json(EXAMPLES / "toy_policy.json"),
                         load_json(ROOT / "kaggle_harness/assets/toy_policy.json"))

    def _start_sleeping_runner(self):
        p = self.proposal(failure_mode="timeout")
        p["timeout_seconds"] = 70
        rid = self.h.register(p)
        proc = subprocess.Popen([sys.executable, "-m", "kaggle_harness", "--store", str(self.h.store.root),
                                 "run", "--id", rid], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            run = self.h.store.get(rid)
            log = self.h.store.run_dir(rid) / "train.stdout.log"
            if run["child_pid"] and log.exists() and log.stat().st_size:
                return rid, proc
            if proc.poll() is not None:
                self.fail(f"Runner exited early: {run}")
            time.sleep(.05)
        proc.kill()
        proc.wait()
        self.fail("Sleeping runner did not start")

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux process-token recovery test")
    def test_actual_sigkill_recovery_preserves_logs_and_kills_orphan(self):
        rid, proc = self._start_sleeping_runner()
        child = self.h.store.get(rid)
        try:
            proc.kill()
            proc.wait(timeout=5)
            self.h.recover(0)
            self.assertEqual(self.h.store.get(rid)["status"], "interrupted")
            self.assertIn("step=1", (self.h.store.run_dir(rid) / "train.stdout.log").read_text())
            deadline = time.monotonic() + 3
            while process_alive(child["child_pid"], child["child_token"]) and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertFalse(process_alive(child["child_pid"], child["child_token"]))
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait()
            if process_alive(child["child_pid"], child["child_token"]):
                os.killpg(child["child_pid"], signal.SIGKILL)

    @unittest.skipUnless(os.name == "posix", "POSIX SIGTERM test")
    def test_actual_sigterm_marks_cancelled(self):
        rid, proc = self._start_sleeping_runner()
        try:
            proc.terminate()
            proc.wait(timeout=6)
            self.assertEqual(self.h.store.get(rid)["status"], "cancelled")
            self.assertTrue((self.h.store.run_dir(rid) / "output/curves.csv").exists())
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait()


class RecorderTests(unittest.TestCase):
    def test_duplicate_steps_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Recorder(Path(tmp), {"steps": 2}) as r:
                r.log(step=1, train_loss=2., learning_rate=.1)
                with self.assertRaises(HarnessError):
                    r.log(step=1, train_loss=1., learning_rate=.1)

    def test_nonfinite_curve_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Recorder(Path(tmp), {}) as r:
                with self.assertRaises(HarnessError):
                    r.log(step=1, train_loss=float("nan"), learning_rate=.1)

    def test_no_finish_record_on_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                with Recorder(Path(tmp), {}) as r:
                    r.log(step=1, train_loss=1., learning_rate=.1)
                    raise RuntimeError("crash")
            self.assertFalse((Path(tmp) / "training_summary.json").exists())
            self.assertTrue((Path(tmp) / "training_error.json").exists())

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "duplicate.json"
            p.write_text('{"steps":2,"steps":100}')
            with self.assertRaises(HarnessError):
                load_json(p)

    def test_safe_paths_reject_cross_platform_escape(self):
        for text in ("../x", "/tmp/x", "C:/x", "a\\b", "a//b", "a/./b"):
            with self.subTest(path=text), self.assertRaises(HarnessError):
                safe_relative(text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
