"""`resolve_adapter` must read the adapter's own config, not trust the caller's flags.

The resolver downloads `adapter_config.json` to digest it, then used to discard it and write the
caller-supplied `--lora-rank` / `--model` straight into the immutable manifest. Every one of these
mismatches was therefore caught only by `_validate_adapter_config` *inside the paid GPU container*,
so `--dry-run` reported success and a real deployment could leave billable resources in a failed or
outcome-unknown state.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from flash.serve.deployment import resolve as resolve_module
from flash.serve.deployment.resolve import (
    ADAPTER_CONFIG,
    ADAPTER_WEIGHTS,
    ResolveError,
    resolve_adapter,
)

BASE = "Qwen/Qwen3.5-9B"
BASE_REVISION = "b" * 40
ARTIFACT_REVISION = "a" * 40


def _install_hub(monkeypatch, tmp_path: Path, config: dict) -> None:
    """Stand up the two hub reads the resolver makes, backed by real files on disk."""
    (tmp_path / ADAPTER_CONFIG).write_text(json.dumps(config), encoding="utf-8")
    rank = config.get("r", 32)
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        rank = 32
    module = "base_model.model.layers.0.self_attn.q_proj"
    save_file(
        {
            f"{module}.lora_A.weight": np.zeros((rank, 2), dtype=np.float32),
            f"{module}.lora_B.weight": np.zeros((2, rank), dtype=np.float32),
        },
        tmp_path / ADAPTER_WEIGHTS,
    )

    class _Info:
        sha = ARTIFACT_REVISION

    class _Api:
        def repo_info(self, **_kwargs):
            return _Info()

    def _download(*, filename: str, **_kwargs) -> str:
        return str(tmp_path / Path(filename).name)

    monkeypatch.setattr(resolve_module, "_hub_api", lambda: _Api())
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)


def _resolve(**overrides):
    kwargs = {
        "run_id": "run1",
        "artifact_repo_id": "Freesolo-Co/artifacts",
        "artifact_subfolder": "rl/run1/seed0/adapter",
        "base_model": BASE,
        "base_model_revision": BASE_REVISION,
        "lora_rank": 32,
    }
    kwargs.update(overrides)
    return resolve_adapter(**kwargs)


def test_a_rank_that_disagrees_with_the_config_is_rejected(monkeypatch, tmp_path) -> None:
    # declares an AGREEING base model so the rank is what this case turns on: a config missing
    # `base_model_name_or_path` is refused earlier, and would pass this test for the wrong reason.
    _install_hub(
        monkeypatch, tmp_path, {"peft_type": "LORA", "r": 16, "base_model_name_or_path": BASE}
    )

    with pytest.raises(ResolveError, match="disagrees"):
        _resolve(lora_rank=32)


def test_a_config_without_a_declared_base_model_is_rejected(monkeypatch, tmp_path) -> None:
    """The container compares this field for equality, so an absent one can never match.

    Resolving it to ``None`` skipped the check rather than failing it, which deferred a certain
    rejection until after the provider had allocated and started billing.
    """
    _install_hub(monkeypatch, tmp_path, {"peft_type": "LORA", "r": 32})

    with pytest.raises(ResolveError, match="declares no base_model_name_or_path"):
        _resolve()


def test_a_base_model_that_disagrees_with_the_config_is_rejected(monkeypatch, tmp_path) -> None:
    _install_hub(
        monkeypatch,
        tmp_path,
        {"peft_type": "LORA", "r": 32, "base_model_name_or_path": "Qwen/Qwen3.8-27B"},
    )

    with pytest.raises(ResolveError, match="trained against"):
        _resolve(base_model=BASE)


def test_the_manifest_binds_the_revision_the_adapter_was_trained_against(
    monkeypatch, tmp_path
) -> None:
    # the model repo is mutable, so resolving its tip at deploy time can pair the adapter with
    # weights it never saw. the config's own revision wins.
    trained_against = "c" * 40
    _install_hub(
        monkeypatch,
        tmp_path,
        {
            "peft_type": "LORA",
            "r": 32,
            "base_model_name_or_path": BASE,
            "revision": trained_against,
        },
    )

    resolved = _resolve(base_model_revision=BASE_REVISION)
    assert resolved.adapter.base_model_revision == trained_against


def test_an_agreeing_config_resolves(monkeypatch, tmp_path) -> None:
    _install_hub(
        monkeypatch,
        tmp_path,
        {"peft_type": "LORA", "r": 32, "base_model_name_or_path": BASE},
    )

    resolved = _resolve()
    assert resolved.adapter.lora_rank == 32
    assert resolved.adapter.base_model == BASE
    assert resolved.adapter.artifact_revision == ARTIFACT_REVISION


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        pytest.param("invalid-structure", "structure is invalid", id="invalid-structure"),
        pytest.param("incomplete-pair", "incomplete LoRA tensor pair", id="incomplete-pair"),
        pytest.param("rank-contradiction", "ranks contradict", id="rank-contradiction"),
    ],
)
def test_unusable_adapter_weights_fail_resolution_without_an_extra_download(
    monkeypatch, tmp_path, failure: str, expected: str
) -> None:
    config = {"peft_type": "LORA", "r": 32, "base_model_name_or_path": BASE}
    _install_hub(monkeypatch, tmp_path, config)
    weights_path = tmp_path / ADAPTER_WEIGHTS
    module = "base_model.model.layers.0.self_attn.q_proj"
    if failure == "invalid-structure":
        weights_path.write_bytes(b"nonempty but not safetensors")
    elif failure == "incomplete-pair":
        save_file(
            {f"{module}.lora_A.weight": np.zeros((32, 2), dtype=np.float32)},
            weights_path,
        )
    else:
        save_file(
            {
                f"{module}.lora_A.weight": np.zeros((16, 2), dtype=np.float32),
                f"{module}.lora_B.weight": np.zeros((2, 16), dtype=np.float32),
            },
            weights_path,
        )
    downloads: list[str] = []

    def download(*, filename: str, **_kwargs) -> str:
        downloads.append(filename)
        return str(tmp_path / Path(filename).name)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)

    with pytest.raises(ResolveError, match=expected):
        _resolve()

    assert downloads == [
        "rl/run1/seed0/adapter/adapter_config.json",
        "rl/run1/seed0/adapter/adapter_model.safetensors",
    ]


@pytest.mark.parametrize(
    ("artifact_subfolder", "checkpoint_step", "expected"),
    [
        pytest.param(
            "rl/run1/adapter",
            10,
            "identifies the final adapter",
            id="final-subfolder-with-step",
        ),
        pytest.param(
            "rl/run1/checkpoints/step-10/adapter",
            None,
            "--checkpoint-step is unset",
            id="step-subfolder-without-step",
        ),
        pytest.param(
            "rl/run1/checkpoints/step-10/adapter",
            20,
            "identifies checkpoint step 10",
            id="different-steps",
        ),
    ],
)
def test_checkpoint_selection_must_agree_with_a_canonical_artifact_subfolder(
    monkeypatch, artifact_subfolder: str, checkpoint_step: int | None, expected: str
) -> None:
    # the path selects the actual bytes, so rejecting before the hub read prevents an authored flag
    # from relabeling final weights as a saved step or one saved step as another.
    monkeypatch.setattr(
        resolve_module,
        "_hub_api",
        lambda: pytest.fail("a contradictory checkpoint must fail before hub access"),
    )

    with pytest.raises(ResolveError, match=expected):
        _resolve(artifact_subfolder=artifact_subfolder, checkpoint_step=checkpoint_step)


def test_checkpoint_provenance_is_derived_from_an_agreeing_step_subfolder(
    monkeypatch, tmp_path
) -> None:
    _install_hub(
        monkeypatch,
        tmp_path,
        {"peft_type": "LORA", "r": 32, "base_model_name_or_path": BASE},
    )

    resolved = _resolve(
        artifact_subfolder="rl/run1/checkpoints/step-10/adapter", checkpoint_step=10
    )

    assert resolved.adapter.checkpoint_id == "run1/step-10"
    assert resolved.adapter.artifact_revision == ARTIFACT_REVISION


def test_unrecognized_artifact_layout_is_rejected_before_hub_access(monkeypatch) -> None:
    monkeypatch.setattr(
        resolve_module,
        "_hub_api",
        lambda: pytest.fail("an unattested checkpoint path must fail before hub access"),
    )

    with pytest.raises(ResolveError, match="does not identify a canonical Flash"):
        _resolve(artifact_subfolder="sft/run-1-step-2", checkpoint_step=2)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"peft_type": "IA3"}, "peft_type must be LORA"),
        ({"peft_type": None}, "peft_type must be LORA"),
        ({"task_type": "SEQ_CLS"}, "task_type must be absent or CAUSAL_LM"),
        ({"modules_to_save": ["lm_head"]}, "modules_to_save"),
        ({"revision": 7}, "revision must be a string"),
    ],
)
def test_a_config_the_container_will_refuse_is_rejected_before_provisioning(
    monkeypatch, tmp_path, overrides: dict, expected: str
) -> None:
    """Each of these is decided by the config alone, so paying a gpu to learn it is waste.

    ``_validate_adapter_config`` rejects every one of these inside the serving container, and the
    verdict never depends on anything the gpu observes at runtime. Resolving them anyway meant a
    deployment that could not possibly serve still allocated provider resources and started billing
    before failing for a reason that was readable here, for free, from bytes already on disk.
    """
    _install_hub(
        monkeypatch,
        tmp_path,
        {"peft_type": "LORA", "r": 32, "base_model_name_or_path": BASE, **overrides},
    )

    with pytest.raises(ResolveError, match=expected):
        _resolve()


def test_a_config_the_container_accepts_still_resolves(monkeypatch, tmp_path) -> None:
    """The guard above must reject only what the container rejects, not narrow what deploys.

    ``task_type`` and ``modules_to_save`` are legitimately absent on most adapters, and an empty
    ``modules_to_save`` is what peft writes when nothing is saved -- so treating either as a
    conflict would refuse ordinary adapters that serve correctly today.
    """
    _install_hub(
        monkeypatch,
        tmp_path,
        {
            "peft_type": "LORA",
            "r": 32,
            "base_model_name_or_path": BASE,
            "task_type": "CAUSAL_LM",
            "modules_to_save": [],
        },
    )

    assert _resolve().adapter.lora_rank == 32


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"base_model_name_or_path": f"{BASE} "}, id="base-trailing"),
        pytest.param({"base_model_name_or_path": f" {BASE}"}, id="base-leading"),
        pytest.param({"revision": f"{BASE_REVISION} "}, id="revision-trailing"),
    ],
)
def test_padded_provenance_is_rejected_before_provider_resources_exist(
    monkeypatch, tmp_path, overrides: dict
) -> None:
    """A value that matches only after stripping must not resolve.

    `_validate_adapter_config` compares these raw bytes for equality inside the container, so a
    padded value it will refuse used to pass resolution, provision, and start billing before
    failing -- the exact outcome this guard exists to prevent. Normalizing the padding away
    instead would be worse for `revision`, which the resolver *adopts* into the immutable
    manifest: that would launder a padded string into the deployment record rather than surface it.
    """
    _install_hub(
        monkeypatch,
        tmp_path,
        {"peft_type": "LORA", "r": 32, "base_model_name_or_path": BASE, **overrides},
    )

    with pytest.raises(ResolveError, match="surrounding whitespace"):
        _resolve()


def test_an_unreadable_config_is_a_resolve_error(monkeypatch, tmp_path) -> None:
    (tmp_path / ADAPTER_CONFIG).write_text("{not json", encoding="utf-8")
    (tmp_path / ADAPTER_WEIGHTS).write_bytes(b"weights-bytes")

    class _Info:
        sha = ARTIFACT_REVISION

    class _Api:
        def repo_info(self, **_kwargs):
            return _Info()

    monkeypatch.setattr(resolve_module, "_hub_api", lambda: _Api())
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda *, filename, **_kwargs: str(tmp_path / Path(filename).name),
    )

    with pytest.raises(ResolveError):
        _resolve()


@pytest.mark.parametrize(
    "constant",
    [
        pytest.param("NaN", id="nan"),
        pytest.param("Infinity", id="positive-infinity"),
        pytest.param("-Infinity", id="negative-infinity"),
    ],
)
def test_non_finite_adapter_config_constants_are_not_readable_json(
    tmp_path: Path, constant: str
) -> None:
    config_path = tmp_path / ADAPTER_CONFIG
    config_path.write_text(
        f'{{"peft_type":"LORA","r":32,"lora_alpha":{constant},"base_model_name_or_path":"{BASE}"}}',
        encoding="utf-8",
    )

    with pytest.raises(ResolveError, match=r"adapter_config\.json is not readable json"):
        resolve_module._declared_provenance(str(config_path))


def test_a_duplicate_key_is_rejected_before_provider_resources_exist(monkeypatch, tmp_path) -> None:
    """the control plane must read this file exactly as the gpu container will.

    `json.load` keeps the last value, so a config declaring `r` twice resolves cleanly here against
    one rank while the materializer's `_reject_duplicate_keys` refuses the identical bytes. that
    split means the artifact is only rejected after the modal app or runpod pod exists -- billing
    the user for a deployment that could never have started. this function exists specifically to
    catch provenance problems before provisioning, so the rule has to match on both sides.
    """

    (tmp_path / ADAPTER_CONFIG).write_text(
        f'{{"peft_type":"LORA","r":16,"r":32,"base_model_name_or_path":"{BASE}"}}',
        encoding="utf-8",
    )
    (tmp_path / ADAPTER_WEIGHTS).write_bytes(b"weights-bytes")

    class _Info:
        sha = ARTIFACT_REVISION

    class _Api:
        def repo_info(self, **_kwargs):
            return _Info()

    monkeypatch.setattr(resolve_module, "_hub_api", lambda: _Api())
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda *, filename, **_kwargs: str(tmp_path / Path(filename).name),
    )

    with pytest.raises(ResolveError, match="duplicate key"):
        _resolve()


@pytest.mark.parametrize("unset", ["", "   ", None])
def test_absent_hub_token_is_none_rather_than_an_empty_bearer(monkeypatch, unset) -> None:
    """No token must mean *no credential*, not an empty one.

    `huggingface_hub` sends any non-None token, building the literal header `Bearer `, which httpx
    rejects as an illegal header value. The request therefore dies with `LocalProtocolError` before
    it leaves the process, which surfaces as the same "could not resolve the commit" message a
    genuinely private repo produces. That made every PUBLIC serving checkpoint unreadable to a
    self-hoster who has no `HF_TOKEN` at all -- the exact user this path exists for.

    Asserted on the value handed to `HfApi`, because that is where the malformed header is built;
    a test that only checked `_token()` would keep passing if a caller reintroduced `or ""`.
    """
    if unset is None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HF_TOKEN", unset)

    seen: list[object] = []

    class _Api:
        def __init__(self, token=None):
            seen.append(token)

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)

    resolve_module._hub_api()

    assert seen == [None], seen


def test_a_real_hub_token_is_still_forwarded(monkeypatch) -> None:
    """The fix must not stop sending a credential that exists -- private repos still need it."""
    monkeypatch.setenv("HF_TOKEN", "  hf_realtoken  ")

    seen: list[object] = []

    class _Api:
        def __init__(self, token=None):
            seen.append(token)

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)

    resolve_module._hub_api()

    assert seen == ["hf_realtoken"], seen


@pytest.mark.parametrize(
    ("label", "prefix", "encoding"),
    [
        pytest.param("utf-8 bom", b"\xef\xbb\xbf", "utf-8", id="utf-8-bom"),
        pytest.param("utf-16-le", b"\xff\xfe", "utf-16-le", id="utf-16-le"),
        pytest.param("utf-16-be", b"\xfe\xff", "utf-16-be", id="utf-16-be"),
    ],
)
def test_a_config_the_container_cannot_decode_is_rejected_here(
    monkeypatch, tmp_path, label: str, prefix: bytes, encoding: str
) -> None:
    """The control plane must decode these bytes exactly as the gpu container will.

    `_load_strict_config` decodes strict utf-8 and refuses a BOM. Passing the raw handle to
    `json.load` instead lets it auto-detect utf-16 and skip a BOM (RFC 4627), so a config the
    container rejects outright resolved cleanly, provisioned, and started billing before failing
    -- the same split this function already closes for duplicate keys.
    """
    body = json.dumps({"peft_type": "LORA", "r": 32, "base_model_name_or_path": BASE})
    (tmp_path / ADAPTER_CONFIG).write_bytes(prefix + body.encode(encoding))
    (tmp_path / ADAPTER_WEIGHTS).write_bytes(b"weights-bytes")

    class _Info:
        sha = ARTIFACT_REVISION

    class _Api:
        def repo_info(self, **_kwargs):
            return _Info()

    monkeypatch.setattr(resolve_module, "_hub_api", lambda: _Api())
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda *, filename, **_kwargs: str(tmp_path / Path(filename).name),
    )

    with pytest.raises(ResolveError, match="not readable json"):
        _resolve()
