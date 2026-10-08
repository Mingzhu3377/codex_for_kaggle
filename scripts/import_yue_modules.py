"""Explicitly register selected Python sections of a local YUE-style document.

No matching, download, dependency installation, module execution or publication.
The review index is supplied by a human and must describe the exact document SHA.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kaggle_harness.modules import ModuleLibrary, card_template
from kaggle_harness.util import HarnessError, atomic_json, load_json


def extract_sections(text):
    sections, current, fence, language, block, block_line = [], None, None, None, [], None
    for line_number, line in enumerate(text.splitlines(keepends=True), 1):
        stripped = line.strip()
        marker = re.fullmatch(r"(`{3,}|~{3,})(.*)", stripped)
        if fence is not None:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                if language in {"python", "py"}:
                    if current is None:
                        raise HarnessError("Python block must belong to a numbered top-level section")
                    current["blocks"].append({"line": block_line, "source": "".join(block)})
                fence, language, block = None, None, []
            else:
                block.append(line)
            continue
        if marker:
            fence, language, block_line = marker[1], marker[2].strip().lower(), line_number + 1
            continue
        heading = re.match(r"^#\s+(\d+)\s*\.\s*(.+?)\s*$", line)
        if heading:
            number = int(heading[1])
            if any(s["section"] == number for s in sections):
                raise HarnessError("Duplicate numbered module section")
            current = {"section": number, "heading": line.lstrip("#").strip(), "line": line_number, "blocks": []}
            sections.append(current)
    if fence is not None:
        raise HarnessError("Unclosed fenced code block")
    return sections


def import_document(document, review_index, library, work_dir, *, all_sections=False, selected=None):
    raw = document.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    sections = extract_sections(raw.decode("utf-8-sig"))
    reviews = load_json(review_index)
    required = {"id", "name", "source_heading", "source_line", "source_sha256", "mechanism", "input_output",
                "conditions_and_findings", "review_status", "per_module_contributor"}
    if not isinstance(reviews, list) or not reviews:
        raise HarnessError("Review index must be a nonempty list")
    by_section = {}
    for row in reviews:
        if not isinstance(row, dict) or not required.issubset(row) or not isinstance(row["id"], int) or isinstance(row["id"], bool):
            raise HarnessError("Malformed human review index")
        if row["id"] in by_section or row["source_sha256"] != source_sha:
            raise HarnessError("Review index duplicates a section or refers to another document SHA")
        by_section[row["id"]] = row
    if all_sections == bool(selected):
        raise HarnessError("Choose exactly --all or one or more explicit --section values")
    choices = {s["section"] for s in sections if s["blocks"]} if all_sections else set(selected)
    if not choices or choices - {s["section"] for s in sections if s["blocks"]}:
        raise HarnessError("Selected section has no Python source")
    chosen = [s for s in sections if s["section"] in choices]
    for s in chosen:
        review = by_section.get(s["section"])
        if review is None or review["source_heading"] != s["heading"] or review["source_line"] != s["line"]:
            raise HarnessError("Review heading/line does not match the exact source section")
    # All document/index checks happen before creating a destination or library.
    work_dir = work_dir.resolve()
    if work_dir.exists():
        raise HarnessError("Extraction requires a NEW work directory")
    work_dir.mkdir(parents=True)
    mapping = []
    with ModuleLibrary(library, create=not library.exists()) as lib:
        for s in chosen:
            row = by_section[s["section"]]
            folder = work_dir / f"section_{s['section']:02d}"
            folder.mkdir()
            for i, b in enumerate(s["blocks"], 1):
                name = "module.py" if len(s["blocks"]) == 1 else f"module_{i:02d}.py"
                (folder / name).write_bytes(b["source"].encode("utf-8"))
            (folder / "README.md").write_text(
                f"# {s['heading']}\n\nSource: {document.resolve()}:{s['line']}\n\nDocument SHA-256: {source_sha}\n\n"
                f"Human review: {row['review_status']}\n\n{row['conditions_and_findings']}\n\n"
                "Registration checks syntax only. No dependency installation, forward/backward test or task benefit is implied.\n",
                encoding="utf-8")
            card = card_template()
            card.update(family=f"yue-section-{s['section']:02d}", name=row["name"], mechanism=row["mechanism"],
                        author="local-document-review", tags=["yue-reference", "runtime-unverified"])
            card["interface"] = {"inputs": row["input_output"], "outputs": "See the actual frozen code and stated input/output relation.",
                                 "constraints": [row["conditions_and_findings"]], "invariants": []}
            card["usage"] = {"insertion_points": ["Choose manually after inspecting the task and frozen implementation."],
                             "initialization": "Read the frozen implementation; runtime behavior is not yet verified here.",
                             "adaptation_notes": "Original extracted reference. Register adaptations as child versions."}
            card["provenance"] = {"references": [{"source": str(document.resolve()), "locator": f"{s['heading']}; line {s['line']}",
                                                   "revision": source_sha}],
                                  "license": "Unknown source license; local research reference; check rights before redistribution.",
                                  "contributors": [row["per_module_contributor"]]}
            card["limitations"] = [row["review_status"], row["conditions_and_findings"],
                                    "No PyTorch forward/backward test or cross-task effectiveness is established by registration."]
            record = lib.add(card, folder)
            mapping.append({"section": s["section"], "name": row["name"], "module_id": record["id"],
                            "content_hash": record["content_hash"], "source_files": record["files"], "checks": record["static_checks"]})
        checked = lib.check()
    result = {"document_sha256": source_sha, "library": str(library.resolve()), "registered": mapping,
              "integrity": checked, "modules_executed": False, "published": False}
    atomic_json(work_dir / "import_results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--review-index", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--all", action="store_true")
    choice.add_argument("--section", type=int, action="append")
    args = parser.parse_args()
    try:
        result = import_document(args.document, args.review_index, args.library, args.work_dir,
                                 all_sections=args.all, selected=args.section)
        print(f"Registered {len(result['registered'])} local reference versions; integrity={result['integrity']['ok']}; no module execution.")
        return 0 if result["integrity"]["ok"] else 1
    except (HarnessError, OSError, ValueError) as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
