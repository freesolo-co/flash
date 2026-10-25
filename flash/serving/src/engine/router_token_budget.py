"""Exact text admission without starting or occupying a GPU replica."""

import asyncio
from collections import OrderedDict

from flash.serving.src.engine.model_config import base_models
from flash.serving.src.engine.text_tokenizer import TextPromptTokenizer
from flash.serving.src.engine.token_budget import token_budget
from flash.serving.src.io.cache_namespace import internal_generation_payload
from flash.serving.src.io.schemas import internal_adapter_payload


def _load(base_model):
    from flash.serving.src.engine.boot import load_tokenizer
    from flash.serving.src.store import settings as cfg
    from flash.serving.src.store.settings import Settings

    owner = TextPromptTokenizer()
    owner.base_model = base_model
    _processor, owner.tokenizer = load_tokenizer(base_model, Settings(), cfg)
    owner._prompt_token_cache = OrderedDict()
    owner._prompt_cache_size = cfg.PROMPT_TOKEN_CACHE_SIZE
    return owner


class RouterTokenBudgets:
    def __init__(self):
        self._owners = {}

    async def start(self):
        # These are the same immutable processor/tokenizer revisions and rendering
        # methods used by each engine. No model weights or CUDA context are loaded.
        models = base_models()
        # Transformers' lazy module imports are not safe to initialize from
        # several loader threads at once. Startup stays off the ASGI event loop.
        owners = await asyncio.to_thread(lambda: [_load(model) for model in models])
        self._owners.update(zip(models, owners, strict=True))

    async def count(self, base_model, payload, record):
        if base_model not in self._owners:
            raise RuntimeError("Router tokenizers have not completed startup")
        return await token_budget(
            self._owners[base_model],
            internal_generation_payload(payload),
            internal_adapter_payload(record),
        )
