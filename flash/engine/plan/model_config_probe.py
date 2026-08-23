"""Reads a pinned commit's ``config.json`` from the hub and checks it against the catalog row.

The catalog describes a model at its DEFAULT revision. A run that pins an exact commit sha loads
different weights than that row describes, so anything sized or shaped from the row is a guess until
the pinned commit's own config confirms it. This module is the probe that fetches that config,
compares it field by field, caches the answer, and reports whether the row can be trusted for this
particular commit.

Every caller asks the same question -- does the commit the worker will actually load agree with the
catalog row? -- and they differ only in what a "no" means: VRAM sizing must fail closed and raise,
while the GPU-width cap degrades to a conservative ceiling. So the rule lives once in
`_certify_model_config` and the policy lives in the callers.

Split out of ``flash.engine.plan.vram`` to keep that module under the file-size limit. ``vram``
re-exports these names, so ``from flash.engine.plan.vram import _validated_revision_geometry`` and
the tests that monkeypatch ``vram.fetch_hf_model_geometry`` keep working unchanged.
"""

from __future__ import annotations


def _config_mismatches(info, params_b, vocab, hidden, layers, heads) -> list[str]:
    """Catalog fields a pinned commit's own geometry contradicts.

    A zero means "the config did not say", not a conflict.
    """
    mismatches: list[str] = []
    if info.params_b > 0 and params_b is not None:
        delta = abs(params_b - info.params_b) / info.params_b
        if delta > 0.05:
            mismatches.append("parameter count")
    if vocab and info.vocab_size and vocab != info.vocab_size:
        mismatches.append("vocabulary size")
    if hidden and info.hidden_size and hidden != info.hidden_size:
        mismatches.append("hidden size")
    if layers and info.num_layers and layers != info.num_layers:
        mismatches.append("layer count")
    if heads and info.num_attention_heads and heads != info.num_attention_heads:
        mismatches.append("attention head count")
    return mismatches


# shared because `allocate()` sizes the run (validating this geometry) and then asks for the head
# cap: an unshared second lookup let a blip between the two narrow a valid eight-card run to four.
# a read is incomplete when the hub omits `safetensors.total`, which is hub metadata rather than
# commit-immutable config geometry. caching that result would keep a valid pin rejected until restart.
_CONFIG_PROBE_MEMO: dict[tuple[str, str], tuple] = {}


def _memoized_config_probe(model_id: str, revision: str) -> tuple:
    """``fetch_hf_model_geometry`` for a pin, reusing an earlier COMPLETE sha read.

    Raises exactly like the strict fetch it wraps when nothing cacheable has succeeded yet, so
    callers that must fail closed still do.
    """
    # imported through `vram` rather than from its defining module so the ~16 tests that
    # monkeypatch `vram.fetch_hf_model_geometry` still intercept this call.
    from flash.engine.plan import vram
    from flash.envs.loading.loader import is_commit_sha

    key = (model_id, revision)
    cached = _CONFIG_PROBE_MEMO.get(key)
    if cached is not None:
        return cached
    geometry = vram.fetch_hf_model_geometry(model_id, revision, strict=True)
    # geometry[0] is params_b; None means the hub answered without a parameter count.
    if is_commit_sha(revision) and geometry[0] is not None:
        _CONFIG_PROBE_MEMO[key] = geometry
    return geometry


def _certify_model_config(model_id: str, revision: str, info) -> tuple:
    """The pinned commit's geometry, or raise saying which catalog field it contradicts.

    The single definition of "certified": the hub gave a parameter count AND nothing in the commit's
    own config contradicts the row. Both callers ask exactly that and differ only in the answer to a
    "no" -- sizing raises, the width cap degrades to its ceiling -- so the rule lives here and the
    policy lives in them. Duplicating it let the two drift: a pin the cap trusts but sizing rejects
    rents the wrong box.
    """
    params_b, vocab, hidden, layers, heads = _memoized_config_probe(model_id, revision)
    # Revision-aware sizing is authoritative and must fail closed. When the pinned commit exposes no
    # parameter-count metadata (no safetensors.total), we cannot derive its size; silently reusing the
    # catalog default-revision count would size the exact-GPU preflight on weights the worker never loads,
    # the precise mis-provisioning this pin exists to prevent.
    if params_b is None:
        raise ValueError(
            f"model_revision for {model_id!r} exposes no parameter-count metadata "
            f"(no safetensors.total); cannot size the pinned revision"
        )
    mismatches = _config_mismatches(info, params_b, vocab, hidden, layers, heads)
    if mismatches:
        raise ValueError(
            f"model_revision for {model_id!r} has geometry incompatible with the catalog: "
            f"{', '.join(mismatches)}"
        )
    return params_b, vocab, hidden, layers, heads


def _validated_revision_geometry(model_id: str, revision: str, info):
    params_b, vocab, _hidden, _layers, _heads = _certify_model_config(model_id, revision, info)
    return params_b, vocab or info.vocab_size


def certified_attention_heads(model_id: str, revision: str) -> int:
    """Query-head count of a PINNED commit, or 0 when it cannot be certified.

    How wide a pinned run may be rented is a divisibility question about the weights the worker
    actually loads, so it is answered from that commit's own ``config.json``, not the catalog row
    describing the default revision.

    Returns 0 -- "not certified" -- for everything `_certify_model_config` will not vouch for:
    an uncataloged model, a hub failure, a commit with no parameter count, geometry that drifted
    from the row, or a config omitting the field. Never raises: widening past the conservative
    ceiling is all this enables, so an uncertifiable pin degrades to that ceiling rather than
    rejecting a run sizing already accepted.

    Reads through `_CONFIG_PROBE_MEMO`, so once ANY caller has read this pin the cap cannot be
    narrowed by a later blip -- `allocate()` sizes before asking for the cap, both via that memo.
    """
    from flash.core.catalog import MODELS

    info = MODELS.get(model_id)
    if info is None or not revision:
        return 0
    try:
        _params_b, _vocab, _hidden, _layers, heads = _certify_model_config(model_id, revision, info)
    except Exception:
        return 0
    return max(int(heads), 0)
