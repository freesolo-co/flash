"""Supplied reward for easy-mode numeric-answer cases (frozen — the agent must not edit).

Contract for an autoenv supplied reward: a pure ``score(gold, response) -> float`` in
``[0, 1]``. The generated easy-mode ``environment.py`` wraps it into a Freesolo
``RewardResult``. This one credits a response whose final number matches the gold's final
number (the GSM8K-style final-answer convention).
"""

from __future__ import annotations

import re

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def score(gold: str, response: str) -> float:
    gold_nums = _NUMBER.findall(gold or "")
    resp_nums = _NUMBER.findall(response or "")
    if not gold_nums or not resp_nums:
        return 0.0
    try:
        return 1.0 if float(gold_nums[-1]) == float(resp_nums[-1]) else 0.0
    except ValueError:
        return 0.0
