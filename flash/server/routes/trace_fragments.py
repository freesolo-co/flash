"""Assembly of a streamed reply from its `data:` fragments, for the recording proxy.

Separated from the SSE framing in `trace_sse` because the two answer different questions: framing
decides where an event ends, this decides what the accumulated reply IS -- how pieces of text are
joined, how a tool call's identity is told from its name, and what bounds retention while a stream
is still open.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import islice
from typing import Any

from flash.server.platform import traces as platform_traces

_FRAGMENT_COMPACT_THRESHOLD = 256


class _StringFragments:
    """Streamed text held as pieces, so the reply is assembled once rather than on every delta.

    Joining on each append is quadratic in the reply length, so the pieces are retained. But
    retaining one object per delta has its own cost the byte budget cannot see: a one-character
    token is a few bytes of text carried by roughly fifty bytes of interpreter overhead plus a list
    slot, so a reply streamed token by token held an order of magnitude more memory than the budget
    that admitted it.

    Pieces are therefore compacted in two levels. At most `_FRAGMENT_COMPACT_THRESHOLD` pieces are
    pending at a time; when that many have arrived they become one block, and the block list is
    compacted the same way. Retention stays bounded at roughly twice the threshold while each byte
    is copied a constant number of times on average, so ingestion is still linear.
    """

    def __init__(self, value: str) -> None:
        self._blocks: list[str] = []
        self._pending: list[str] = [value] if value else []

    @property
    def parts(self) -> list[str]:
        """The retained pieces, in order. Their concatenation is the accumulated text."""
        return [*self._blocks, *self._pending]

    def append(self, value: str) -> None:
        if not value:
            return
        self._pending.append(value)
        if len(self._pending) < _FRAGMENT_COMPACT_THRESHOLD:
            return
        self._blocks.append("".join(self._pending))
        self._pending.clear()
        if len(self._blocks) >= _FRAGMENT_COMPACT_THRESHOLD:
            self._blocks = ["".join(self._blocks)]

    def text(self) -> str:
        return "".join(self.parts)


def _utf8_safe_text(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return value.encode("utf-16", errors="surrogatepass").decode("utf-16", errors="replace")
    return value


def _materialize_fragments(
    value: Any,
    *,
    depth: int = 0,
    note_defect: Callable[[str], None] | None = None,
) -> Any:
    if depth >= platform_traces._MAX_PAYLOAD_DEPTH:
        if note_defect is not None:
            note_defect("stream output exceeded the maximum nesting depth")
        return "[redacted]"
    if isinstance(value, _StringFragments):
        value = value.text()
    if isinstance(value, str):
        return _utf8_safe_text(value)
    if isinstance(value, dict):
        return {
            _utf8_safe_text(key) if isinstance(key, str) else key: _materialize_fragments(
                item, depth=depth + 1, note_defect=note_defect
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _materialize_fragments(item, depth=depth + 1, note_defect=note_defect) for item in value
        ]
    return value


def _content_parts(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, _StringFragments):
        value = value.text()
    if isinstance(value, str):
        return [{"type": "text", "text": value}] if value else []
    return [value]


def _carries_response(fragment: dict[str, Any]) -> bool:
    """Whether a message or delta actually carries reply data.

    `role` alone restates what the choice already established, and an explicit null is the ordinary
    provider spelling for "no fragment", so neither makes a trailing event a continuation.
    """
    return any(key != "role" and value is not None for key, value in fragment.items())


def _append_fragment(
    target: dict[str, Any],
    key: str,
    value: Any,
    *,
    on_collection_bound: Callable[[], None] | None = None,
) -> None:
    current = target.get(key)
    if isinstance(value, str):
        if isinstance(current, _StringFragments):
            current.append(value)
        elif isinstance(current, str):
            target[key] = _StringFragments(current)
            target[key].append(value)
        elif isinstance(current, list):
            current.extend(_room_for(current, _content_parts(value), on_collection_bound))
        else:
            target[key] = _StringFragments(value)
    elif isinstance(value, list):
        if isinstance(current, list):
            current.extend(_room_for(current, value, on_collection_bound))
        elif isinstance(current, str | _StringFragments):
            parts = _content_parts(current)
            target[key] = [*parts, *_room_for(parts, value, on_collection_bound)]
        else:
            target[key] = list(value)
    elif value is not None:
        target[key] = value


def _room_for(
    current: list[Any],
    addition: list[Any],
    on_collection_bound: Callable[[], None] | None,
) -> list[Any]:
    """As much of `addition` as fits under the storage bound once appended to `current`.

    A single event's collections are trimmed on the way in, so this is the cross-event half of the
    same bound: without it a provider could split one over-wide array across two deltas and rebuild
    it here, past the width storage keeps.
    """
    room = platform_traces._MAX_PAYLOAD_COLLECTION - len(current)
    if len(addition) <= room:
        return addition
    if on_collection_bound is not None:
        on_collection_bound()
    return addition[: max(room, 0)]


_TOO_DEEP_DEFECT = "stream fragment exceeded the payload depth bound"

_IDENTITY_KEYS = frozenset({"id", "type"})

# a choice or tool-call slot retains a nest of dicts (message, tool_calls, logprobs, extensions)
# whose ~700 bytes are the structure itself, not the index that named it. the byte budget charged
# only the index, so 100k empty `{"index":N,"delta":{}}` entries -- ~70 MB of retained state -- sat
# inside an 8 MiB budget. slots are bounded by COUNT rather than folded into the byte budget: the
# budget is also what a small deployment lowers to cap recorded text, and charging structure
# against it made the first choice of a 256-byte stream unrecordable. a provider's `n` and its
# parallel tool calls are small; this is the runaway bound, not a working limit.
_MAX_ENTRY_SLOTS = 4096


def _bound_collections(value: Any, *, depth: int = 0) -> tuple[Any, bool]:
    """Trim a retained value to the member count storage will keep. Returns it and whether it was.

    The byte budget charges a value's SERIALIZED size, and a mapping of shallow members serializes
    far smaller than it expands to: a 4 MB event carrying 300k `{"k":1}` pairs sat well inside an
    8 MiB budget while retaining the whole expanded dict per concurrent stream, for members
    `sanitize_json_value` was going to drop at the storage boundary anyway. Trimming here discards
    them before they are held rather than after.

    Whether anything was dropped is returned rather than noted as a defect: an over-wide extension
    is a truncated payload, which the span already reports through `payload_truncated`, not a stream
    that failed to deliver its reply.
    """
    if depth >= platform_traces._MAX_PAYLOAD_DEPTH:
        return value, False
    if isinstance(value, dict):
        bounded = len(value) > platform_traces._MAX_PAYLOAD_COLLECTION
        trimmed: dict[Any, Any] = {}
        for key, item in islice(value.items(), platform_traces._MAX_PAYLOAD_COLLECTION):
            trimmed[key], item_bounded = _bound_collections(item, depth=depth + 1)
            bounded |= item_bounded
        return trimmed, bounded
    if isinstance(value, list):
        bounded = len(value) > platform_traces._MAX_PAYLOAD_COLLECTION
        trimmed_list: list[Any] = []
        for item in islice(value, platform_traces._MAX_PAYLOAD_COLLECTION):
            entry, item_bounded = _bound_collections(item, depth=depth + 1)
            trimmed_list.append(entry)
            bounded |= item_bounded
        return trimmed_list, bounded
    return value, False


def _restates_call_identity(fragment: dict[str, Any]) -> bool:
    """Whether a tool-call delta re-announces WHICH call it is, rather than continuing one.

    A provider streaming a long function name sends the name alone; a delta that repeats the call
    header is describing a call in full, so a second such header is a different call.
    """
    return any(key in fragment for key in _IDENTITY_KEYS)


def _merge_fragment_dict(
    target: dict[str, Any],
    fragment: dict[str, Any],
    *,
    depth: int = 0,
    on_identity_conflict: Callable[[], None] | None = None,
    on_collection_bound: Callable[[], None] | None = None,
    restates_identity: bool = False,
) -> bool:
    """Merge a streamed fragment into `target`. Returns False if the depth bound truncated it.

    Recording must never be able to take down the paid call it is observing. This runs on the
    proxy's own task before each chunk is relayed, so unbounded recursion here would interrupt the
    upstream response and withhold an otherwise relayable event from the caller.
    """
    if depth >= platform_traces._MAX_PAYLOAD_DEPTH:
        return False
    bounded = True
    for key, value in fragment.items():
        # merging widens the target, so a provider could rebuild an over-wide mapping out of narrow
        # deltas that each pass the per-event bound. a key already present is an update rather than
        # growth and still merges; a NEW one past the bound is dropped, since storage keeps only
        # the leading members anyway.
        if key not in target and len(target) >= platform_traces._MAX_PAYLOAD_COLLECTION:
            if on_collection_bound is not None:
                on_collection_bound()
            continue
        if isinstance(value, dict):
            nested = target.setdefault(key, {})
            if isinstance(nested, dict):
                bounded &= _merge_fragment_dict(
                    nested,
                    value,
                    depth=depth + 1,
                    on_identity_conflict=on_identity_conflict,
                    on_collection_bound=on_collection_bound,
                    restates_identity=restates_identity,
                )
            else:
                target[key] = dict(value)
        elif isinstance(value, str):
            current = target.get(key)
            if key in _IDENTITY_KEYS or (key == "name" and restates_identity):
                # identity, not text: `id` and `type` name WHICH call this is, so successive values
                # are alternatives rather than halves. concatenating them stored the nonexistent
                # call `call_Acall_B` with no defect, exportable as a real invocation. the first
                # value wins and a conflicting later one is reported.
                #
                # `name` is BOTH: providers split one function name across deltas, so it can only be
                # read as identity when the delta restates the call header -- a fragment continuing
                # a name carries the name alone. without that distinction two complete headers named
                # `lookup` then `delete` merged into the invented call `lookupdelete`.
                current_text = current.text() if isinstance(current, _StringFragments) else current
                if current_text is not None:
                    if current_text != value and on_identity_conflict is not None:
                        on_identity_conflict()
                    continue
            if isinstance(current, _StringFragments):
                current.append(value)
            elif isinstance(current, str):
                target[key] = _StringFragments(current)
                target[key].append(value)
            else:
                target[key] = _StringFragments(value)
        elif isinstance(value, list):
            _append_fragment(target, key, value, on_collection_bound=on_collection_bound)
        elif value is not None:
            target[key] = value
    return bounded
