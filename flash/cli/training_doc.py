"""The TRAINING.md playbook scaffolded by `flash env setup`.

The playbook itself is prose, so it lives in `TRAINING.md` next to this module rather than
inside a multi-hundred-line string literal. Edit that file, not a copy.
"""

from __future__ import annotations

from pathlib import Path

TRAINING_MD = (Path(__file__).parent / "TRAINING.md").read_text(encoding="utf-8")
