from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

from kaggle_harness.modules import ModuleLibrary
from kaggle_harness.util import HarnessError, atomic_json

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("yue_import", ROOT / "scripts" / "import_yue_modules.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class DocumentImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kh-docimport-")
        self.root = Path(self.temp.name)
        self.document = self.root / "modules.md"
        self.raw = b"# 1. Reference\r\n\r\n```python\r\n# 99. Not a module heading\r\nraise RuntimeError('do not execute')\r\n```\r\n"
        self.document.write_bytes(self.raw)
        self.index = self.root / "review.json"
        self.review = {"id": 1, "name": "Reference", "source_heading": "1. Reference", "source_line": 1,
                       "source_sha256": hashlib.sha256(self.raw).hexdigest(), "mechanism": "Reference-only test.",
                       "input_output": "Unknown until task-specific review.", "conditions_and_findings": "Do not execute during import.",
                       "review_status": "Unverified runtime.", "per_module_contributor": "Unknown"}
        atomic_json(self.index, [self.review])

    def tearDown(self):
        self.temp.cleanup()

    def run_import(self, **kwargs):
        return helper.import_document(self.document, self.index, self.root / "bank", self.root / "extracted", **kwargs)

    def test_exact_code_bytes_and_numbered_comments_are_preserved_without_execution(self):
        result = self.run_import(selected=[1])
        self.assertEqual(len(result["registered"]), 1)
        self.assertFalse(result["modules_executed"])
        mid = result["registered"][0]["module_id"]
        with ModuleLibrary(self.root / "bank") as lib:
            source = (lib.source(mid) / "module.py").read_bytes()
            self.assertEqual(source, b"# 99. Not a module heading\r\nraise RuntimeError('do not execute')\r\n")
            self.assertIn("Unknown source license", lib.verify(mid)["card"]["provenance"]["license"])

    def test_review_for_another_document_is_rejected_before_any_write(self):
        self.document.write_bytes(self.raw + b"Changed\r\n")
        with self.assertRaises(HarnessError):
            self.run_import(all_sections=True)
        self.assertFalse((self.root / "bank").exists())
        self.assertFalse((self.root / "extracted").exists())

    def test_heading_location_must_match_review_index(self):
        atomic_json(self.index, [dict(self.review, source_line=2)])
        with self.assertRaises(HarnessError):
            self.run_import(all_sections=True)

    def test_selection_is_explicit_and_nonexistent_section_is_rejected(self):
        for selection in ({}, {"selected": [2]}, {"selected": [1], "all_sections": True}):
            with self.subTest(selection=selection), self.assertRaises(HarnessError):
                self.run_import(**selection)

    def test_unclosed_fence_is_rejected(self):
        with self.assertRaises(HarnessError):
            helper.extract_sections("# 1. Module\n```python\nVALUE=1\n")


if __name__ == "__main__":
    unittest.main()
