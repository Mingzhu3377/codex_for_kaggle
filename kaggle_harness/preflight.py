"""Validate a competition-owned preflight report, without importing model dependencies."""
from .util import HarnessError


def validate_preflight(report):
    if not isinstance(report, dict) or set(report) != {"schema_version", "scope", "checks"}:
        raise HarnessError("Preflight report needs schema_version, scope, checks")
    if isinstance(report["schema_version"], bool) or report["schema_version"] != 1:
        raise HarnessError("Unsupported preflight schema")
    if not isinstance(report["scope"], str) or not 1 <= len(report["scope"].strip()) <= 3000:
        raise HarnessError("Preflight needs a concrete check scope")
    checks = report["checks"]
    if not isinstance(checks, list) or not 1 <= len(checks) <= 64:
        raise HarnessError("Preflight needs 1..64 checks")
    names = set()
    for check in checks:
        if not isinstance(check, dict) or set(check) != {"name", "passed", "details"}:
            raise HarnessError("Each preflight check needs name, passed, details")
        name = check["name"]
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120 or name in names:
            raise HarnessError("Preflight check names must be distinct nonempty strings")
        names.add(name)
        if not isinstance(check["passed"], bool) or not isinstance(check["details"], str) or len(check["details"]) > 3000:
            raise HarnessError("Preflight needs boolean outcomes and bounded details")
        if not check["passed"]:
            raise HarnessError(f"Preflight check did not pass: {name}")
    return report
