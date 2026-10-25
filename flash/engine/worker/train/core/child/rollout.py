"""Request and completion policy via upstream's vLLM server extension points."""

import ray
from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

from flash.engine.worker.train.core.child.rollout_contract import RolloutContractServer


class FlashVLLMServer(RolloutContractServer, vLLMHttpServer):
    def _init_model_config(self, model_config):
        from verl.utils.config import omega_conf_to_dataclass

        from flash.engine.worker.train.core.child.configuration import FlashModelConfig

        return omega_conf_to_dataclass(model_config, dataclass_type=FlashModelConfig)

    async def generate(self, *args, **kwargs):
        if self.lora_as_adapter and VLLM_LORA_INT_ID not in await self.engine.list_loras():
            raise RuntimeError(
                "the training LoRA adapter is not loaded; refusing base-model generation"
            )
        return await super().generate(*args, **kwargs)

    def _postprocess_output(self, output, final_res, sampling_params):
        output = super()._postprocess_output(output, final_res, sampling_params)
        if final_res.outputs:
            output.extra_fields["finish_reason"] = final_res.outputs[0].finish_reason
        return output


class FlashVLLMReplica(vLLMReplica):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(FlashVLLMServer)

    def _get_server_name_prefix(self):
        # The native adapter derives its lookup name from rollout.name; the replica defaults
        # to vllm_ even for registered subclasses. Both sides must share the same namespace.
        return f"{self.config.name}_"
