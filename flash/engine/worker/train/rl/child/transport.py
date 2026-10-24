"""Keep the contract's request ID intact across upstream load balancing."""

from verl.workers.rollout.llm_server import LLMServerClient


class RolloutContractTransport(LLMServerClient):
    def _vllm_request_id(self, request_id):
        # the agent already supplies a fresh UUID for every generation turn.
        return request_id
