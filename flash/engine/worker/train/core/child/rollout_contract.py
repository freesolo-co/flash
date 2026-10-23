"""Bind the pinned vLLM weight-sync hooks to the engine-independent V1 contract."""

from flash.engine.worker.train.core.rollout.contract import (
    RolloutRequest,
    digest,
    validate_tokens,
)
from flash.engine.worker.train.core.rollout.identity import model_identity, policy_identity
from flash.engine.worker.train.core.rollout.lifecycle import PolicyLease

REQUEST_FIELD = "_flash_rollout_request"
RESULT_FIELD = "flash_rollout_result"


class RolloutContractServer:
    """Mixin for the Flash vLLM server; GRPO cannot opt out of validation."""

    def _contract(self):
        if self.model_config.runtime_options.get("algorithm") != "grpo":
            return None
        if not hasattr(self, "_policy_lease"):
            self._policy_lease = PolicyLease()
            self._policy_identity = model_identity(self.model_config)
            self._weights_received = False
        return self._policy_lease

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):  # noqa: ASYNC109
        # upstream RPC signature; timeout is forwarded to the engine.
        lease = self._contract()
        loading = isinstance(method, str) and method.startswith("update_weights")
        if lease is not None and loading:
            lease.invalidate()
            self._weights_received = False
            if method != "update_weights_from_ipc":
                raise ValueError("rollout V1 supports full weight synchronization only")
        result = await super().collective_rpc(method, timeout=timeout, args=args, kwargs=kwargs)
        if lease is not None and loading:
            self._weights_received = True
        return result

    async def set_global_steps(self, global_steps):
        lease = self._contract()
        if lease is not None:
            if not self._weights_received:
                raise ValueError(
                    "rollout policy acknowledgement requires successful weight loading"
                )
            lease.invalidate()
            self._weights_received = False
            policy = policy_identity(self._policy_identity, global_steps)
            result = await super().set_global_steps(global_steps)
            lease.acknowledge(policy)
            return result
        return await super().set_global_steps(global_steps)

    async def generate(self, prompt_ids, sampling_params, request_id, **kwargs):
        lease = self._contract()
        if lease is None:
            return await super().generate(
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                request_id=request_id,
                **kwargs,
            )
        params = dict(sampling_params)
        request = RolloutRequest.decode(params.pop(REQUEST_FIELD, None))
        if request.request_id != request_id or request.prompt_digest != digest(prompt_ids):
            raise ValueError("rollout request id or prompt does not match its envelope")
        if request.logprob_definition != self.config.logprobs_mode:
            raise ValueError("rollout logprob definition does not match the engine")
        if not params.get("logprobs"):
            raise ValueError("rollout V1 requires generation-time token logprobs")
        if (
            request.logprob_definition == "processed_logprobs"
            and params.get("temperature", 1.0) < 1e-5
        ):
            raise ValueError("processed rollout logprobs require stochastic sampling")
        acknowledgement = lease.admit(request)
        output = await super().generate(
            prompt_ids=prompt_ids,
            sampling_params=params,
            request_id=request_id,
            **kwargs,
        )
        lease.finish(acknowledgement)
        cancelled = output.stop_reason == "aborted"
        engine_prompt_digest = output.extra_fields.pop("_flash_engine_prompt_digest", None)
        if not cancelled:
            if engine_prompt_digest != request.prompt_digest:
                raise ValueError("rollout engine conditioning tokens do not match its request")
            validate_tokens(output.token_ids, output.log_probs)
        output.extra_fields[RESULT_FIELD] = {
            "request": request.encode(),
            "load_ack": acknowledgement,
            "termination": "cancelled" if cancelled else output.extra_fields["finish_reason"],
            "sampling": output.extra_fields.pop("_flash_effective_sampling", {}),
            "token_digest": digest([output.token_ids, output.log_probs]),
        }
        return output

    def _postprocess_output(self, output, final_res, sampling_params):
        output = super()._postprocess_output(output, final_res, sampling_params)
        if self._contract() is not None:
            import msgspec

            if output.stop_reason != "aborted":
                # RequestOutput carries the engine's tokens after multimodal expansion,
                # not the deduplicated placeholders passed to engine.generate().
                prompt_ids = getattr(final_res, "prompt_token_ids", None)
                if prompt_ids is None:
                    raise ValueError("rollout engine did not report its conditioning tokens")
                output.extra_fields["_flash_engine_prompt_digest"] = digest(prompt_ids)
            # capture the effective engine parameters after upstream applies defaults/caps.
            output.extra_fields["_flash_effective_sampling"] = {
                name: msgspec.to_builtins(getattr(sampling_params, name))
                for name in sampling_params.__struct_fields__
            }
        return output
