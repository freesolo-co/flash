"""Held-out functional probes for the live environment evaluation/reporting path."""

import re

from flash.envs.meta.evaluations import EvalCase


class ServingSmokeSuite:
    name = "serving_smoke"

    def cases(self):
        return [
            EvalCase(id="addition", input="What is 2+2? Reply with a single number.", expected="4"),
            EvalCase(
                id="subtraction", input="What is 9-2? Reply with a single number.", expected="7"
            ),
        ]

    def score(self, case, response):
        return bool(re.search(r"\b" + re.escape(case.expected) + r"\b", response))


def load_evaluations():
    return [ServingSmokeSuite()]
