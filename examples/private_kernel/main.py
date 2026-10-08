"""Private CPU compatibility example; this is not a scored competition submission."""
import json
from pathlib import Path

result = {"kind": "private_cpu_smoke", "value": sum(i * i for i in range(1000)),
          "scientific_result": "not_imported"}
Path("/kaggle/working/kh-smoke.json").write_text(json.dumps(result), encoding="utf-8")
print(json.dumps(result))
