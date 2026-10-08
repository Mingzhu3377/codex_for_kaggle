from __future__ import annotations

import copy
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

from kaggle_harness import integrity, module_usage
from kaggle_harness.cli import main
from kaggle_harness.codex import ask, example_proposal
from kaggle_harness.engine import Harness
from kaggle_harness.modules import ModuleLibrary, card_template, validate_card, _record_hash
from kaggle_harness.module_demo import run_module_demo
from kaggle_harness.preflight import validate_preflight
from kaggle_harness.util import HarnessError, atomic_json, digest, load_json, tree_manifest

ROOT = Path(__file__).resolve().parents[1]


class ModuleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kh-modules-")
        self.root = Path(self.temp.name)
        self.folder = self.root / "module_source"
        self.folder.mkdir()
        self.source = self.folder / "reference.py"
        self.source.write_text("VALUE = 1\n# frozen-reference-marker\n", encoding="utf-8")
        self.lib = ModuleLibrary(self.root / "bank", create=True)
        self.card = card_template()
        self.card.update(family="test-family", name="Reference", mechanism="A controlled source reference.", author="test")
        self.workspace = self.root / "workspace"
        shutil.copytree(ROOT / "examples" / "toy_competition", self.workspace)
        self.policy = load_json(ROOT / "examples" / "toy_policy.json")
        self.h = Harness.initialize(self.workspace, self.root / "state", self.policy)
        module_usage.attach(self.h, self.lib.root)

    def tearDown(self):
        self.h.close()
        if self.lib is not None:
            self.lib.close()
        self.temp.cleanup()

    def add(self, **updates):
        card = copy.deepcopy(self.card)
        card.update(updates)
        return self.lib.add(card, self.folder)["id"]

    def proposal(self, mid=None, *, parent=None, edits=None, **config):
        p = example_proposal(self.h, parent_id=parent)
        p["timeout_seconds"] = 10.
        p["config"].update(config)
        p["edits"] = edits or []
        if mid:
            p["modules"] = [{"module_id": mid, "files": [{"source": "reference.py", "target": "features/reference.py"}],
                             "adaptation": "Reference copied for this controlled fixture.", "mode": "copy"}]
        return p

    def completed(self, mid=None, **kwargs):
        rid = self.h.register(self.proposal(mid, **kwargs))
        run = self.h.run(rid)
        self.assertEqual(run["status"], "completed", run["error"])
        return rid

    def interpretation(self, outcome="neutral"):
        return {"outcome": outcome, "conditions": "Synthetic regression, same protocol and seed; source inclusion only.",
                "reason": "This fixture does not infer a causal benefit of the referenced file.",
                "failure_layer": None, "author": "test"}

    def test_registration_freezes_bytes_without_executing_module(self):
        self.source.write_text("raise RuntimeError('MUST NOT EXECUTE')\n", encoding="utf-8")
        mid = self.add()
        r = self.lib.verify(mid)
        self.assertEqual(r["static_checks"]["runtime"], "not_tested")
        self.source.write_text("VALUE=999\n", encoding="utf-8")
        self.assertIn("MUST NOT EXECUTE", (self.lib.source(mid) / "reference.py").read_text())

    def test_variant_preserves_parent_and_reports_actual_file_difference(self):
        parent = self.add()
        self.source.write_text("VALUE=2\n", encoding="utf-8")
        child = self.add(name="Variant", parents=[parent])
        self.assertEqual(self.lib.diff(parent, child)["modified"], ["reference.py"])
        self.assertIn("VALUE = 1", (self.lib.source(parent) / "reference.py").read_text())
        self.assertTrue(self.lib.check()["ok"])

    def test_missing_parent_does_not_register_a_variant(self):
        with self.assertRaises(HarnessError):
            self.add(parents=["M-"+"0"*16])
        self.assertEqual(self.lib.list(), [])

    def test_invalid_syntax_retains_attempt_but_not_a_registered_module(self):
        self.source.write_text("def broken(:\n", encoding="utf-8")
        with self.assertRaises(HarnessError):
            self.add()
        self.assertEqual(self.lib.list(), [])
        self.assertEqual(self.lib.db.execute("SELECT COUNT(*) FROM events WHERE kind='module_import_failed'").fetchone()[0], 1)

    def test_card_schema_rejects_unbounded_or_claimed_validation_fields(self):
        for update in ({"validated": True}, {"schema_version": True}, {"parents": [{}]}, {"family": "../outside"},
                       {"limitations": ["x"*3000]*5}):
            with self.subTest(update=update), self.assertRaises(HarnessError):
                validate_card(dict(self.card, **update))

    def test_source_and_card_tampering_are_detected(self):
        mid = self.add()
        source = self.lib.source(mid) / "reference.py"
        source.write_text("VALUE=3\n")
        with self.assertRaises(HarnessError):
            self.lib.verify(mid)
        self.assertFalse(self.lib.check()["ok"])

    def test_immutable_module_and_edge_records(self):
        parent = self.add(); child = self.add(parents=[parent])
        for sql in ("UPDATE modules SET family='other'", "DELETE FROM modules", "DELETE FROM module_edges"):
            with self.assertRaises(sqlite3.IntegrityError):
                self.lib.db.execute(sql)
        self.assertEqual(self.lib.verify(child)["card"]["parents"], [parent])

    def test_credentials_rejected_before_their_file_is_archived(self):
        token = "KGAT_" + "a"*40
        self.source.write_text("#" + "x"*130000 + "\nTOKEN='" + token + "'\n")
        with self.assertRaises(HarnessError):
            self.add()
        self.assertFalse(any(token in p.read_text(errors="replace") for p in (self.lib.root / "versions").rglob("*.py")))

    def test_known_nonstandard_secret_rejected(self):
        self.source.write_text("value='opaque-known-credential-12345'\n")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "opaque-known-credential-12345"}), self.assertRaises(HarnessError):
            self.add()

    def test_bundle_round_trip_includes_ancestry_and_is_idempotent(self):
        parent = self.add(); child = self.add(name="Child", parents=[parent])
        bundle = self.root / "export"
        self.lib.export(child, bundle)
        with ModuleLibrary(self.root / "another", create=True) as other:
            result = other.import_bundle(bundle)
            self.assertEqual(result["versions_added"], 2)
            self.assertEqual(other.verify(child), self.lib.verify(child))
            self.assertEqual(other.import_bundle(bundle)["versions_added"], 0)

    def test_bundle_corruption_rejected_before_database_registration(self):
        mid = self.add(); folder = self.root / "export"; self.lib.export(mid, folder)
        (folder / "versions" / mid / "source" / "reference.py").write_text("VALUE=99\n")
        with ModuleLibrary(self.root / "another", create=True) as other:
            with self.assertRaises(HarnessError):
                other.import_bundle(folder)
            self.assertEqual(other.list(), [])

    def test_bundle_path_traversal_rejected(self):
        mid = self.add(); folder = self.root / "export"; self.lib.export(mid, folder)
        bundle = load_json(folder / "bundle.json"); bundle["files"]["../outside.py"] = "0"*64
        atomic_json(folder / "bundle.json", bundle)
        with ModuleLibrary(self.root / "another", create=True) as other, self.assertRaises(HarnessError):
            other.import_bundle(folder)

    def test_bundle_rejects_extra_material_even_with_updated_manifest(self):
        mid = self.add(); folder = self.root / "export"; self.lib.export(mid, folder)
        (folder / "unlisted.py").write_text("print('unexpected')\n")
        bundle = load_json(folder / "bundle.json")
        bundle["files"] = tree_manifest(folder); bundle["files"].pop("bundle.json")
        atomic_json(folder / "bundle.json", bundle)
        with ModuleLibrary(self.root / "another", create=True) as other, self.assertRaises(HarnessError):
            other.import_bundle(folder)

    def test_export_omits_ignored_caches_in_reference_archive(self):
        cache = self.folder / "__pycache__"; cache.mkdir()
        (cache / "reference.pyc").write_bytes(b"not source")
        mid = self.add()
        self.assertFalse((self.lib.source(mid) / "__pycache__").exists())
        folder = self.root / "export"; self.lib.export(mid, folder)
        self.assertFalse(any(p.name == "__pycache__" for p in folder.rglob("*")))
        self.assertTrue(self.lib.check()["ok"])

    def test_unexpected_files_in_frozen_source_are_detected(self):
        mid = self.add()
        (self.lib.source(mid) / "unexpected.py").write_text("VALUE=123\n")
        with self.assertRaises(HarnessError):
            self.lib.verify(mid)

    def test_explicit_source_copy_and_adaptation_are_separately_frozen(self):
        mid = self.add()
        rid = self.h.register(self.proposal(mid, edits=[{"path":"features/reference.py", "content":"VALUE=7\n# adapted-marker\n"}]))
        binding = module_usage.bindings(self.h, rid)[0]
        self.assertTrue(binding["files"][0]["adapted"])
        self.assertTrue(binding["files"][0]["changed_after_copy"])
        rd = self.h.store.run_dir(rid)
        self.assertIn("VALUE = 1", (rd/"modules"/mid/"source"/"reference.py").read_text())
        self.assertIn("VALUE=7", (rd/"source"/"features"/"reference.py").read_text())
        self.h.verify(rid)

    def test_parent_branch_keeps_adapted_source_instead_of_resetting_original(self):
        mid = self.add()
        parent = self.completed(mid, edits=[{"path":"features/reference.py", "content":"VALUE=7\n"}])
        p = self.proposal(parent=parent)
        p.pop("modules")  # Backward-style callers still inherit provenance deterministically.
        child = self.h.register(p)
        self.assertEqual((self.h.store.run_dir(child)/"source"/"features"/"reference.py").read_text(), "VALUE=7\n")
        self.assertEqual(module_usage.bindings(self.h, child)[0]["inherited_from"], parent)
        self.h.verify(child)

    def test_inherited_source_can_be_used_when_original_library_is_offline(self):
        mid = self.add(); parent = self.completed(mid)
        self.lib.close(); self.lib = None
        bank_db = self.root/"bank"/"library.sqlite3"
        bank_db.rename(bank_db.with_name("offline.sqlite3"))
        child = self.h.register(self.proposal(parent=parent))
        self.h.verify(child)
        self.assertEqual(self.h.run(child)["status"], "completed")

    def test_explicit_empty_references_clear_active_module_bindings(self):
        mid = self.add(); parent = self.completed(mid)
        p = self.proposal(parent=parent); p["modules"] = []
        child = self.h.register(p)
        self.assertEqual(module_usage.bindings(self.h, child), [])
        self.h.verify(child)

    def test_target_restrictions_prevent_protected_evaluator_replacement(self):
        mid = self.add()
        for target in ("evaluate.py", "../outside.py", "secret.txt"):
            p = self.proposal(mid); p["modules"][0]["files"][0]["target"] = target
            with self.subTest(target=target), self.assertRaises(HarnessError):
                self.h.register(p)

    def test_module_binding_archive_tampering_is_detected(self):
        mid = self.add(); rid = self.h.register(self.proposal(mid))
        path = self.h.store.run_dir(rid)/"modules"/mid/"source"/"reference.py"
        path.write_text("VALUE=9\n")
        with self.assertRaises(HarnessError):self.h.verify(rid)
        self.assertFalse(integrity.check(self.h)["ok"])

    def test_report_covers_module_provenance_and_detects_omission(self):
        mid = self.add(); self.completed(mid)
        data = integrity.report_data(self.h)
        self.assertTrue(integrity.audit_report(self.h,data)["ok"])
        data["runs"][0].pop("modules")
        self.assertFalse(integrity.audit_report(self.h,data)["ok"])

    def test_usage_index_reads_actual_experiment_state(self):
        mid = self.add(); rid = self.h.register(self.proposal(mid))
        self.assertEqual(self.lib.uses(mid)[0]["live_evidence"]["run_status"], "ready")
        self.h.run(rid)
        self.assertEqual(self.lib.uses(mid)[0]["live_evidence"]["run_status"], "completed")

    def test_unwritable_advisory_index_does_not_fail_a_valid_experiment(self):
        mid = self.add()
        with patch.object(ModuleLibrary, "record_use", side_effect=sqlite3.OperationalError("index is temporarily unwritable")):
            rid = self.h.register(self.proposal(mid))
        self.assertEqual(self.h.store.get(rid)["status"], "ready")
        self.assertEqual(self.h.store.db.execute("SELECT COUNT(*) FROM events WHERE run_id=? AND kind='module_index_unavailable'", (rid,)).fetchone()[0], 1)
        self.h.verify(rid)
        self.assertEqual(self.h.run(rid)["status"], "completed")

    def test_case_keeps_conditional_interpretation_and_original_evidence(self):
        mid = self.add(); rid = self.completed(mid)
        case = self.lib.add_case(self.h,mid,self.interpretation(),run_id=rid)
        self.assertEqual(case["evidence"]["result"],self.h.store.get(rid)["result"])
        self.assertEqual(self.lib.cases(mid)[0]["live_evidence"]["status"], "metadata_matches")
        self.assertIsNone(self.h.store.champion())
        with self.assertRaises(sqlite3.IntegrityError):self.lib.db.execute("DELETE FROM module_cases")

    def test_smoke_and_failed_runs_do_not_establish_method_outcomes(self):
        mid = self.add()
        p=self.proposal(mid);p["purpose"]="smoke_test";p["config"]["steps"]=2
        rid=self.h.register(p);self.h.run(rid)
        with self.assertRaises(HarnessError):self.lib.add_case(self.h,mid,self.interpretation("positive"),run_id=rid)
        self.lib.add_case(self.h,mid,self.interpretation("undecided"),run_id=rid)

    def test_case_requires_actual_module_binding(self):
        mid=self.add();rid=self.completed()
        with self.assertRaises(HarnessError):self.lib.add_case(self.h,mid,self.interpretation(),run_id=rid)

    def test_case_baseline_requires_same_protocol(self):
        mid=self.add();baseline=self.completed(mid);different=self.completed(mid,seed=18)
        with self.assertRaises(HarnessError):self.lib.add_case(self.h,mid,self.interpretation(),run_id=different,baseline_id=baseline)

    def test_case_bundle_reuses_metadata_but_does_not_need_to_execute_modules(self):
        mid=self.add();rid=self.completed(mid)
        self.lib.add_case(self.h,mid,self.interpretation(),run_id=rid)
        bundle=self.root/"export";self.lib.export(mid,bundle)
        with ModuleLibrary(self.root/"another",create=True) as other:
            self.assertEqual(other.import_bundle(bundle)["cases_added"],1)
            self.assertEqual(other.cases(mid)[0]["live_evidence"]["status"],"metadata_matches")
            self.assertTrue(other.check()["ok"])

    def test_case_archive_tampering_is_rejected_by_lookup_and_export(self):
        mid = self.add(); rid = self.completed(mid)
        case = self.lib.add_case(self.h, mid, self.interpretation(), run_id=rid)
        path = self.lib.root / "cases" / (case["id"] + ".json")
        changed = copy.deepcopy(case); changed["interpretation"]["outcome"] = "positive"
        atomic_json(path, changed)
        with self.assertRaises(HarnessError):
            self.lib.cases(mid)
        with self.assertRaises(HarnessError):
            self.lib.export(mid, self.root / "export")
        self.assertFalse(self.lib.check()["ok"])

    def test_offline_original_experiment_retains_explicit_frozen_case(self):
        mid = self.add(); rid = self.completed(mid)
        self.lib.add_case(self.h, mid, self.interpretation(), run_id=rid)
        self.h.close()
        db = self.root / "state" / "ledger.sqlite3"
        db.rename(db.with_name("offline.sqlite3"))
        try:
            self.assertEqual(self.lib.cases(mid)[0]["live_evidence"]["status"], "unavailable")
            self.assertTrue(self.lib.check()["ok"])
            self.assertEqual(len(self.lib.check()["warnings"]), 1)
        finally:
            db.with_name("offline.sqlite3").rename(db)
            self.h = Harness(self.root / "state")

    def test_case_interpretation_rejects_unhashable_values_and_credentials(self):
        mid = self.add(); rid = self.completed(mid)
        for update in ({"outcome": []}, {"failure_layer": {}}, {"reason": "KGAT_" + "x"*40}):
            with self.subTest(update=update), self.assertRaises(HarnessError):
                self.lib.add_case(self.h, mid, dict(self.interpretation(), **update), run_id=rid)
        self.assertEqual(self.lib.cases(mid), [])

    def test_bundle_case_cannot_relabel_another_source_version(self):
        mid = self.add(); rid = self.completed(mid)
        case = self.lib.add_case(self.h, mid, self.interpretation(), run_id=rid)
        folder = self.root / "export"; self.lib.export(mid, folder)
        binding = case["evidence"]["binding"]
        binding["content_hash"] = "0"*64
        binding["binding_hash"] = digest({k:v for k,v in binding.items() if k != "binding_hash"})
        case["content_hash"] = _record_hash(case)
        atomic_json(folder / "cases" / (case["id"] + ".json"), case)
        bundle = load_json(folder / "bundle.json")
        bundle["files"] = tree_manifest(folder); bundle["files"].pop("bundle.json")
        atomic_json(folder / "bundle.json", bundle)
        with ModuleLibrary(self.root / "another", create=True) as other:
            with self.assertRaises(HarnessError):
                other.import_bundle(folder)
            self.assertEqual(other.list(), [])

    def test_same_library_tracks_two_disjoint_competition_stores(self):
        mid = self.add(); first = self.h.register(self.proposal(mid))
        other_workspace = self.root / "other_workspace"
        shutil.copytree(self.workspace, other_workspace)
        with Harness.initialize(other_workspace, self.root / "other_state", dict(self.policy, competition_id="second-task")) as other:
            module_usage.attach(other, self.lib.root)
            second = other.register(self.proposal(mid))
            other.verify(second)
        self.assertEqual({u["run_id"] for u in self.lib.uses(mid)}, {first, second})

    def test_cpu_module_demo_calls_source_and_transports_conditional_case(self):
        result = run_module_demo(self.root / "module_demo")
        self.assertGreater(result["task_a"]["baseline_mse"], .1)
        self.assertLess(result["task_a"]["candidate_mse"], .001)
        self.assertEqual(result["import"]["cases_added"], 1)
        with ModuleLibrary(self.root / "module_demo" / "imported_library") as lib:
            outcomes = [c["interpretation"]["outcome"] for c in lib.cases(result["module_ids"]["quadratic"])]
        self.assertEqual(outcomes, ["positive", "undecided"])

    def test_default_context_does_not_discover_modules(self):
        self.add()
        answer=ask(self.h,"Inspect baseline",dry_run=True)
        context=load_json(Path(answer["directory"])/"context.json")
        self.assertEqual(context["selected_modules"],[])
        self.assertNotIn("Reference",json.dumps(context))

    def test_large_accumulated_cases_do_not_block_reading_one_selected_source(self):
        mid = self.add(); rid = self.completed(mid)
        for _ in range(8):
            interpretation = dict(self.interpretation(), conditions="x"*5000, reason="y"*5000)
            self.lib.add_case(self.h, mid, interpretation, run_id=rid)
        answer = ask(self.h, "Read the implementation with bounded case history", module_ids=[mid], dry_run=True)
        context = load_json(Path(answer["directory"]) / "context.json")
        selected = context["selected_modules"][0]
        self.assertEqual(len(selected["cases"]), 1)
        self.assertEqual(selected["recent_cases_omitted_for_budget"], 7)
        self.assertEqual(len(self.lib.cases(mid)), 8)
        self.assertTrue((Path(context["selected_module_source_paths"][0]["path"]) / "reference.py").is_file())

    def test_explicit_module_source_is_available_to_codex_transport(self):
        mid=self.add()
        proposal=self.proposal(mid)
        envelope={"decision":"experiment","reason":"Inspect real source.","proposal_json":json.dumps(proposal)}
        fake=self.root/"codex_fixture.py"
        fake.write_text("import sys,json\nfrom pathlib import Path\n"
            "if '--version' in sys.argv: print('fixture');sys.exit(0)\n"
            "if '--help' in sys.argv: print('--json --output-schema --output-last-message --sandbox --skip-git-repo-check --ignore-user-config --ephemeral --cd');sys.exit(0)\n"
            "prompt=sys.stdin.read()\ncontext=json.loads(prompt.split('CURRENT EVIDENCE:\\n',1)[1].split('\\n\\nPROPOSAL SHAPE',1)[0])\n"
            "p=Path(context['selected_module_source_paths'][0]['path'])/'reference.py'\n"
            "assert 'frozen-reference-marker' in p.read_text()\n"
            f"Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text({json.dumps(json.dumps(envelope))},encoding='utf-8')\n",encoding="utf-8")
        result=ask(self.h,"Reuse selected source",module_ids=[mid],binary=[sys.executable,str(fake)],timeout=10)
        self.assertEqual(result["status"],"validated")

    def test_unselected_module_cannot_be_added_by_model(self):
        mid=self.add();proposal=self.proposal(mid)
        fake=self.root/"codex_fixture.py"
        envelope={"decision":"experiment","reason":"Unselected.","proposal_json":json.dumps(proposal)}
        fake.write_text("import sys\nfrom pathlib import Path\n"
            "if '--version' in sys.argv: print('fixture');sys.exit(0)\n"
            "if '--help' in sys.argv: print('--json --output-schema --output-last-message --sandbox --skip-git-repo-check --ignore-user-config --ephemeral --cd');sys.exit(0)\n"
            "sys.stdin.read()\n"
            f"Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text({json.dumps(json.dumps(envelope))})\n",encoding="utf-8")
        with self.assertRaises(HarnessError):ask(self.h,"Do not select modules",binary=[sys.executable,str(fake)],timeout=10)

    def test_cli_module_add_list_show_and_integrity(self):
        card=self.root/"card.json";atomic_json(card,self.card)
        with redirect_stdout(io.StringIO()) as out,redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--library",str(self.lib.root),"module","add","--folder",str(self.folder),"--card",str(card)]),0)
        mid=json.loads(out.getvalue())["id"]
        for action in ("list","check"):
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):self.assertEqual(main(["--library",str(self.lib.root),"module",action]),0)
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):self.assertEqual(main(["--library",str(self.lib.root),"module","show",mid]),0)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="kh-preflight-");self.root=Path(self.temp.name)
        self.workspace=self.root/"workspace";shutil.copytree(ROOT/"examples"/"toy_competition",self.workspace)

    def tearDown(self):self.temp.cleanup()

    def harness(self, script):
        (self.workspace/"preflight.py").write_text(script,encoding="utf-8")
        policy=load_json(ROOT/"examples"/"toy_policy.json")
        policy["preflight_command"]=["{python}","preflight.py"]
        policy["protected_files"].append("preflight.py")
        return Harness.initialize(self.workspace,self.root/"state",policy)

    def script(self, passed):
        report={"schema_version":1,"scope":"Concrete controlled check.","checks":[{"name":"interface","passed":passed,"details":"Fixture result."}]}
        return "import os,json\nfrom pathlib import Path\nPath(os.environ['KH_PREFLIGHT_OUTPUT']).write_text("+repr(json.dumps(report))+",encoding='utf-8')\n"

    def test_successful_preflight_is_sealed_to_exact_source(self):
        with self.harness(self.script(True)) as h:
            rid=h.register(example_proposal(h));run=h.run(rid)
            self.assertEqual(run["status"],"completed",run["error"])
            self.assertIn("preflight_verified.json",run["result"]["sealed_files"])
            self.assertEqual(load_json(h.store.run_dir(rid)/"preflight_verified.json")["snapshot_hash"],run["snapshot_hash"])
            h.verify(rid)

    def test_failed_preflight_stops_before_training(self):
        with self.harness(self.script(False)) as h:
            rid=h.register(example_proposal(h));run=h.run(rid)
            self.assertEqual(run["status"],"incomplete")
            self.assertFalse((h.store.run_dir(rid)/"train.stdout.log").exists())
            self.assertIsNone(run["result"])

    def test_preflight_mutating_source_is_rejected(self):
        script=self.script(True)+"Path('train.py').write_text('raise RuntimeError()')\n"
        with self.harness(script) as h:
            rid=h.register(example_proposal(h));run=h.run(rid)
            self.assertEqual(run["status"],"incomplete")
            self.assertFalse((h.store.run_dir(rid)/"train.stdout.log").exists())

    def test_preflight_cannot_change_training_copy_while_own_copy_passes(self):
        script = self.script(True) + "(Path(os.environ['KH_SOURCE_ROOT']).with_name('train_work')/'train.py').write_text('raise RuntimeError()')\n"
        with self.harness(script) as h:
            rid = h.register(example_proposal(h)); run = h.run(rid)
            self.assertEqual(run["status"], "incomplete")
            self.assertFalse((h.store.run_dir(rid) / "train.stdout.log").exists())

    def test_training_cannot_change_source_after_passing_preflight(self):
        train = self.workspace / "train.py"
        train.write_text(train.read_text() + "\nPath('train.py').write_text('changed')\n")
        with self.harness(self.script(True)) as h:
            rid = h.register(example_proposal(h)); run = h.run(rid)
            self.assertEqual(run["status"], "incomplete")
            self.assertFalse((h.store.run_dir(rid) / "evaluate.stdout.log").exists())

    def test_source_root_points_to_each_actual_stage(self):
        for filename, expected in (("train.py", "train_work"), ("evaluate.py", "eval_work")):
            path = self.workspace / filename
            path.write_text("import os\nfrom pathlib import Path\nassert Path(os.environ['KH_SOURCE_ROOT']).name == " + repr(expected) + "\n" + path.read_text())
        script = self.script(True) + "assert Path(os.environ['KH_SOURCE_ROOT']).name == 'preflight_work'\n"
        with self.harness(script) as h:
            rid = h.register(example_proposal(h)); run = h.run(rid)
            self.assertEqual(run["status"], "completed", run["error"])

    def test_preflight_timeout_uses_existing_run_reservation(self):
        with self.harness("import time\ntime.sleep(10)\n") as h:
            p=example_proposal(h);p["timeout_seconds"]=.3
            rid=h.register(p);run=h.run(rid)
            self.assertEqual(run["status"],"timed_out")
            self.assertGreater(run["elapsed_seconds"],.2)

    def test_preflight_rejects_empty_duplicate_or_nonboolean_checks(self):
        base={"schema_version":1,"scope":"scope","checks":[{"name":"shape","passed":True,"details":"checked"}]}
        wrong=[]
        for checks in ([],base["checks"]*2,[{"name":"shape","passed":1,"details":"checked"}]):
            wrong.append(dict(base,checks=checks))
        wrong.append(dict(base,schema_version=True))
        for report in wrong:
            with self.subTest(report=report),self.assertRaises(HarnessError):validate_preflight(report)

    def test_preflight_command_is_part_of_new_protocol_hash(self):
        with self.harness(self.script(True)) as h:
            rid = h.register(example_proposal(h))
            data = load_json(h.store.run_dir(rid) / "datasets.json")
            original = h._protocol_hash(data, 17)
            h.policy["preflight_command"] = ["{python}", "preflight.py", "--another-check"]
            self.assertNotEqual(h._protocol_hash(data, 17), original)


if __name__=="__main__":unittest.main()
