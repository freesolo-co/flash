"""regenerate the pinned image sft processor oracle."""

from __future__ import annotations

import json

from tests.test_image_sft_processor_parity import (
    ORACLE_PATH,
    _load_live_image_stack,
    capture_processor_oracle,
)


def main() -> None:
    oracle = capture_processor_oracle(*_load_live_image_stack())
    ORACLE_PATH.write_text(
        json.dumps(oracle, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
