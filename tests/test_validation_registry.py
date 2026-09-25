"""The validation registry (validation.yaml) stays consistent with the code.

Run as a script to see where each method stands and which validated ones changed
since they were validated::

    python tests/test_validation_registry.py
"""

import importlib
import inspect
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "validation.yaml"
STATUSES = {"validated", "issues", "failed", "pending"}
REVIEWED_FIELDS = ("date", "commit", "data", "how", "evidence")


def load_registry() -> List[Dict[str, Any]]:
    yaml = pytest.importorskip("yaml")
    if not REGISTRY.exists():
        pytest.skip("validation.yaml is not next to the tests (e.g. testing a wheel)")
    with open(REGISTRY, encoding="utf-8") as fh:
        return list(yaml.safe_load(fh))


def resolve(path: str) -> Any:
    """The object at an import path such as ``pkg.module.function``."""
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(path)


def source_files(entry: Dict[str, Any]) -> List[str]:
    """Repo-relative files the entry's result depends on."""
    own = Path(inspect.getsourcefile(resolve(entry["target"])) or "").resolve()
    files = [str(own.relative_to(REPO))] if REPO in own.parents else []
    return files + list(entry.get("files", []))


def changed_since(entry: Dict[str, Any]) -> Optional[bool]:
    """Whether the entry's files changed since its commit; None if git cannot tell."""
    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                str(entry["commit"]),
                "--",
                *source_files(entry),
            ],
            cwd=REPO,
            capture_output=True,
        )
    except OSError:
        return None
    return {0: False, 1: True}.get(result.returncode)


def test_entries_are_well_formed() -> None:
    entries = load_registry()
    ids = [e.get("id") for e in entries]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for entry in entries:
        assert {"id", "target", "status"} <= entry.keys(), entry
        assert entry["status"] in STATUSES, entry["id"]
        if entry["status"] != "pending":
            missing = [f for f in REVIEWED_FIELDS if not entry.get(f)]
            assert not missing, f"{entry['id']} lacks {missing}"


def test_targets_exist() -> None:
    """A renamed or removed function must not leave a stale entry behind."""
    for entry in load_registry():
        resolve(entry["target"])


def test_listed_files_exist() -> None:
    for entry in load_registry():
        for name in entry.get("files", []):
            assert (REPO / name).exists(), f"{entry['id']}: {name}"


def main() -> None:
    entries = load_registry()
    width = max(len(e["id"]) for e in entries)
    for entry in entries:
        line = f"{entry['id']:<{width}}  {entry['status']:<9}"
        if entry["status"] != "pending":
            changed = changed_since(entry)
            line += f"  {entry['date']} @ {entry['commit']}"
            if changed:
                line += "  ** code changed since: validate again **"
            elif changed is None:
                line += "  (git cannot compare against that commit)"
        print(line)


if __name__ == "__main__":
    main()
