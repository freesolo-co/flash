"""Token reuse must be indistinguishable from full encoding and remain bounded."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers, trainers

from flash.serve.runtime.token_prefix_cache import TokenPrefixCache

MARKER = "<user>"


def tokenizer(*, normalized=False, lstrip=False, normalizer=None):
    value = Tokenizer(models.BPE(unk_token="[UNK]"))
    value.normalizer = normalizer
    value.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.ByteLevel(add_prefix_space=False)]
    )
    value.train_from_iterator(
        ["hello world 123 café e\u0301 世界 🙂\n\t abc def"],
        trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]"]),
    )
    value.add_tokens([AddedToken(MARKER, normalized=normalized, lstrip=lstrip)])
    return value


@pytest.mark.parametrize("nfc", [False, True])
@pytest.mark.parametrize("suffix", ["", "\u0301 hi", "  next\n", "<user> raw", "世界🙂", "e\u0301"])
def test_cached_boundary_matches_full_bpe_for_unicode_and_literal_tokens(nfc, suffix):
    raw = tokenizer(normalizer=normalizers.NFC() if nfc else None)
    cache = TokenPrefixCache(raw, MARKER, MARKER)
    prefix = ("café e\u0301 世界 🙂\n\t" * 200) + MARKER
    for text in [prefix + "first", prefix + suffix]:
        expected = raw.encode(text, add_special_tokens=False).ids
        assert cache.encode(text, "tenant") == expected
        result = cache.encode(text, "tenant")
        assert result == expected
        result[0] = 123456  # Callers must not mutate cached token IDs.
        assert cache.encode(text, "tenant") == expected


def test_normalized_boundary_requires_identity_normalizer():
    raw = tokenizer(normalized=True, normalizer=normalizers.NFC())
    cache = TokenPrefixCache(raw, MARKER, MARKER)
    assert not cache.supported
    text = "e\u0301 " * 1000 + MARKER + "e\u0301"
    assert cache.encode(text, "tenant") == raw.encode(text, add_special_tokens=False).ids
    assert not cache.entries


def test_stripping_boundary_and_unscoped_requests_do_not_cache():
    for raw, namespace in [(tokenizer(lstrip=True), "tenant"), (tokenizer(), None)]:
        cache = TokenPrefixCache(raw, MARKER, MARKER)
        text = "hello " * 300 + "  " + MARKER + " world"
        assert cache.encode(text, namespace) == raw.encode(text, add_special_tokens=False).ids
        assert not cache.entries


def test_cache_is_tenant_scoped_and_expires(monkeypatch):
    raw = tokenizer()
    cache = TokenPrefixCache(raw, MARKER, MARKER, ttl=10)
    now = [100.0]
    monkeypatch.setattr("flash.serve.runtime.token_prefix_cache.time.monotonic", lambda: now[0])
    text = "hello " * 300 + MARKER + " world"
    cache.encode(text, "a")
    cache.encode(text, "b")
    assert {key[0] for key in cache.entries} == {"a", "b"}
    now[0] += 11
    assert cache.encode(text, "a") == raw.encode(text, add_special_tokens=False).ids
    assert next(v[1] for k, v in cache.entries.items() if k[0] == "a") == now[0]
    assert cache.bytes == sum(len(v[0]) for v in cache.entries.values())


def test_concurrent_evictions_preserve_exact_ids_and_memory_bound():
    raw = tokenizer()
    cache = TokenPrefixCache(raw, MARKER, MARKER, max_bytes=3000, max_entries=2)
    texts = [("hello " * 300 + str(i % 8) + MARKER + str(i), str(i % 3)) for i in range(64)]
    expected = [raw.encode(text, add_special_tokens=False).ids for text, _ in texts]
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = list(pool.map(lambda pair: cache.encode(*pair), texts))
    assert actual == expected
    assert cache.bytes <= 3000
    assert len(cache.entries) <= 2
    assert cache.bytes == sum(len(v[0]) for v in cache.entries.values())


def test_sequence_level_truncation_keeps_full_encoding_path():
    raw = tokenizer()
    cache = TokenPrefixCache(raw, MARKER, MARKER)
    raw.enable_truncation(max_length=20)
    text = "hello " * 300 + MARKER + "world"
    assert cache.encode(text, "tenant") == raw.encode(text, add_special_tokens=False).ids
    assert not cache.entries
