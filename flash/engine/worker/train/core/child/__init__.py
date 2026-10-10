"""parent import bridge for stdlib-only child modules."""

from __future__ import annotations

import sys

from flash.content import reasoning_normalization

sys.modules.setdefault("flash_reasoning_normalization", reasoning_normalization)
