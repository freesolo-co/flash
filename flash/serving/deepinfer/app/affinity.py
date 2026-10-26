"""Opaque conversation hints for best-effort routing between GPU replicas."""

import hashlib
import json

from flash.serving.deepinfer.contract.config import MODEL_REVISION


def conversation_affinity(body, namespace):
    """Keep an opening user turn stable as a conversation grows.

    A shared system prompt alone must not pin an organization's unrelated chats
    to one replica. Missing authoritative tenancy or a user turn disables the
    hint. Native token-exact cache matching remains the source of truth.
    """
    if not namespace:
        return None
    opening = []
    for message in body["messages"]:
        opening.append(message)
        if message["role"] == "user":
            identity = [
                "deepinfer-affinity-v1",
                namespace,
                MODEL_REVISION,
                body["chat_template_kwargs"].get("enable_thinking", False),
                opening,
            ]
            encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()
            return hashlib.sha256(encoded).hexdigest()
    return None
