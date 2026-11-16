"""Preserve authenticated cache scope over the private router-to-engine RPC."""

import re


def internal_generation_payload(payload):
    result = payload.model_dump()
    # an unset candidate list stays off the wire, so engines that predate the field accept it.
    if result.get("logprob_token_ids", 0) is None:
        del result["logprob_token_ids"]
    namespace = getattr(payload, "_cache_namespace", None)
    if namespace is not None:
        result["_cache_namespace"] = namespace
    return result


def split_cache_namespace(payload):
    data = dict(payload)
    namespace = data.pop("_cache_namespace", None)
    if namespace is not None and (
        not isinstance(namespace, str) or re.fullmatch(r"[0-9a-f]{64}", namespace) is None
    ):
        raise ValueError("Invalid internal cache namespace")
    return data, namespace
