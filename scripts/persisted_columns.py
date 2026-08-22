"""Print the adapter columns the serving runtime actually reads.

The deploy workflow probes production with this list so the schema gate fails when a column the
runtime depends on has not landed yet. It reads the list out of ``persistence.py`` rather than
restating it, so the gate cannot drift away from the runtime.

Parsed, not imported: the deploy job installs only ``modal`` and ``python-dotenv``, while
``persistence.py`` imports ``httpx`` at module scope. Importing it there raises
``ModuleNotFoundError``, which the workflow's fail-closed guard turns into a blocked production
deploy. ``ast`` keeps this to the standard library.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

CONSTANT = "_PERSISTED_COLUMNS"
SOURCE = Path(__file__).resolve().parents[1] / "flash" / "serving" / "src" / "persistence.py"


def persisted_columns(source: str) -> str | None:
    """Return the string assigned to ``_PERSISTED_COLUMNS``, or None if it is absent or unusable."""
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == CONSTANT for t in node.targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            # a non-literal (say, built from a helper call) cannot be read without executing the
            # module, which is the thing this script exists to avoid.
            return None
        return value if isinstance(value, str) and value else None
    return None


def main() -> int:
    columns = persisted_columns(SOURCE.read_text(encoding="utf-8"))
    if columns is None:
        print(f"could not read {CONSTANT} from {SOURCE}", file=sys.stderr)
        return 1
    print(columns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
