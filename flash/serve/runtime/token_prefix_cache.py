"""Bounded, tenant-scoped token IDs for immutable BPE tokenizers.

An added token isolates pre-tokenization on either side. Reusing the rendered
prefix through that boundary preserves exact IDs, including normalization.
This is CPU tokenization reuse, never evidence of a GPU prefix-cache hit.
"""

import hashlib
import json
import time
from array import array
from collections import OrderedDict
from threading import Lock


class TokenPrefixCache:
    def __init__(
        self, tokenizer, marker, added_token, *, max_bytes=64 << 20, max_entries=128, ttl=600
    ):
        self.tokenizer = tokenizer
        self.marker = marker
        self.added_token = added_token
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.ttl = ttl
        self.entries = OrderedDict()
        self.bytes = 0
        self.lock = Lock()
        config = json.loads(tokenizer.to_str())
        normalizer = config.get("normalizer")
        post = config.get("post_processor")
        added = config.get("added_tokens", [])
        boundary = next((t for t in added if t["content"] == added_token), None)
        # Non-normalized added tokens are extracted before normalization. A
        # normalized boundary is safe here only with an identity normalizer.
        normalizer_safe = (
            normalizer is None
            or normalizer == {"type": "Sequence", "normalizers": []}
            or (boundary is not None and not boundary.get("normalized"))
        )
        pretokenizers = (config.get("pre_tokenizer") or {}).get("pretokenizers", [])
        self.supported = bool(
            marker.startswith(added_token)
            and config.get("model", {}).get("type") == "BPE"
            and config.get("model", {}).get("dropout") is None
            and normalizer_safe
            and (post is None or post.get("type") == "ByteLevel")
            and pretokenizers
            and all(
                p["type"] in {"Split", "ByteLevel"} and not p.get("add_prefix_space")
                for p in pretokenizers
            )
            and config.get("truncation") is None
            and config.get("padding") is None
            and boundary is not None
            and not any(boundary.get(k) for k in ("single_word", "lstrip", "rstrip"))
            and not any(added_token in t["content"] for t in added if t != boundary)
        )

    def _encode(self, text):
        return self.tokenizer.encode_batch([text], add_special_tokens=False)[0].ids

    def encode(self, text, namespace):
        position = text.rfind(self.marker) if namespace and self.supported else -1
        if (
            position < 1024
            or self.tokenizer.truncation is not None
            or self.tokenizer.padding is not None
            or self.tokenizer.encode_special_tokens
        ):
            return self._encode(text)
        split = position + len(self.added_token)
        prefix, suffix = text[:split], text[split:]
        key = (namespace, hashlib.sha256(prefix.encode()).digest())
        now = time.monotonic()
        with self.lock:
            cached = self.entries.pop(key, None)
            if cached is not None:
                ids, touched = cached
                if now - touched < self.ttl:
                    self.entries[key] = (ids, now)
                else:
                    self.bytes -= len(ids)
                    cached = None
        if cached is None:
            prefix_ids = self._encode(prefix)
            # Packed immutable storage bounds payload bytes without retaining text
            # or a Python integer object for every cached token.
            packed = array("I", prefix_ids).tobytes()
            if len(packed) <= self.max_bytes and self.max_entries:
                with self.lock:
                    old = self.entries.pop(key, None)
                    if old is not None:
                        self.bytes -= len(old[0])
                    self.entries[key] = (packed, time.monotonic())
                    self.bytes += len(packed)
                    while self.entries and (
                        len(self.entries) > self.max_entries or self.bytes > self.max_bytes
                    ):
                        _, (evicted, _) = self.entries.popitem(last=False)
                        self.bytes -= len(evicted)
        else:
            decoded = array("I")
            decoded.frombytes(cached[0])
            prefix_ids = decoded.tolist()
        return prefix_ids + self._encode(suffix)
