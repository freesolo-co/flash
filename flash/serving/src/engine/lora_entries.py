"""one synchronized lifecycle entry for each vllm lora adapter."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from flash.serve.contract.provenance import engine_adapter_name, record_key
from flash.serving.src.engine.support import _adapter_source_ident


@dataclass(frozen=True, slots=True)
class _LoraEntry:
    source_ident: tuple[str, str, str, str, str | None]
    lora_request: Any
    state: Literal["reserved", "loaded", "unconfirmed"]


def entries_for(owner: Any) -> dict[tuple[str, str], _LoraEntry]:
    """lazily initialize entries for modal instances whose base initializer never ran."""
    entries = getattr(owner, "_lora_entries", None)
    if entries is None:
        entries = {}
        owner._lora_entries = entries
    return entries


def cached_lora_request(owner: Any, record: Any, path: Path) -> Any:
    source_ident = _adapter_source_ident(record)
    adapter_key = record_key(record)
    adapter_name = engine_adapter_name(*adapter_key)
    entries = entries_for(owner)
    entry = entries.get(adapter_key)
    if entry is not None:
        if entry.state == "unconfirmed":
            raise RuntimeError("LoRA registration is unconfirmed on this engine")
        if entry.source_ident == source_ident:
            return entry.lora_request
        raise RuntimeError("previous LoRA removal is unconfirmed on this engine")

    from vllm.lora.request import LoRARequest

    from flash.serving.src.store.registry import lora_int_id

    # reserve around int32 collisions, including ids that vllm may retain after failed removal.
    used = {entry.lora_request.lora_int_id for entry in entries.values()}
    int_id = lora_int_id(adapter_name)
    while int_id in used:
        int_id = int_id + 1 if int_id < 0x7FFFFFFF else 1

    request = LoRARequest(adapter_name, int_id, str(path))
    entries[adapter_key] = _LoraEntry(source_ident, request, "reserved")
    return request
