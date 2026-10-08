from __future__ import annotations

import html
from pathlib import Path

from .engine import Harness
from .util import HarnessError, dumps, now


def write_report(h: Harness, path: Path) -> dict:
    """Portable local HTML. Does not publish anything or read raw failed-run logs."""
    if path.exists():
        raise HarnessError("Report destination already exists; choose a new filename")
    rows = []
    champion = h.store.champion()
    champion_id = champion["run_id"] if champion else "None"
    for run in h.store.rows(limit=10000):
        score = str(run["result"]["value"]) if run["result"] else "—"
        values = [run["id"], run["parent_id"] or "—", run["purpose"], run["status"], score,
                  f"{run['elapsed_seconds']:.3f}", run["proposal"]["hypothesis"]]
        rows.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in values) + "</tr>")
    content = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Kaggle Research Harness · Experiment report</title>
<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1300px;margin:3rem auto;padding:0 1.5rem}
h1{line-height:1.2}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:.7rem;
text-align:left;vertical-align:top;border-bottom:1px solid}code,pre{font-size:13px}pre{white-space:pre-wrap}
small{opacity:.7}td:first-child{white-space:nowrap}</style>"""
    content += f"<h1>Experiment ledger</h1><p><strong>Champion:</strong> {html.escape(champion_id)}</p>"
    content += f"<p><small>Generated {now()} · local report · maximum 10,000 recent runs</small></p>"
    content += "<h2>Budget</h2><pre>" + html.escape(dumps(h.store.budget(h.policy))) + "</pre>"
    content += "<table><thead><tr>" + "".join(f"<th>{name}</th>" for name in
        ["Run", "Parent", "Purpose", "Status", "Metric", "Wall seconds", "Hypothesis"]) + "</tr></thead>"
    content += "<tbody>" + "".join(rows) + "</tbody></table>"
    content += "<p>Full experiment files remain in the store. This report does not inject archived failures into model context.</p></html>"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": str(path.resolve()), "rows": len(rows)}
