"""Synthetic environment for paid Modal lifecycle qualification, not model-quality evaluation."""

import hashlib

from freesolo.environments import EnvironmentSingleTurn, RewardResult


class ModalQualificationEnvironment(EnvironmentSingleTurn):
    def build_prompt_messages(self, example, prompt_text):
        return [{"role": "user", "content": example.input}]

    def score_response(self, example, response_text):
        digest = hashlib.sha256(response_text.encode()).digest()
        return RewardResult(score=int.from_bytes(digest[:4], "big") / 2**32)


def load_environment(**kwargs):
    return ModalQualificationEnvironment()
