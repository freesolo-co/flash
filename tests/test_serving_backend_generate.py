"""Rendering the self-hosted Modal serving app.

The generated file is deployed to a real GPU, where a mistake costs a cold start and real money to
discover. So these assertions cover the things that would otherwise only fail on hardware:
that it parses, that its engine values are the catalog's validated ones, and that the contract
details flash's client depends on are actually present.
"""

from __future__ import annotations

import ast
import re

import pytest

from flash.core.catalog import MODELS
from flash.serve.backend.generate import (
    VLLM_VERSION,
    app_name_for,
    render_app,
    write_app,
)
from flash.serve.backend.gpus import MODAL_GPUS_BY_NAME

_MODEL_IDS = sorted(MODELS)


def _constants(source: str) -> dict[str, object]:
    """Module-level constants of the generated app, read via ast rather than executed.

    Importing it would need modal, vllm and a GPU; the values are literals, so parsing is enough.
    Non-literal assignments (the image builder, the app object) are simply skipped.
    """
    out: dict[str, object] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        try:
            out[node.targets[0].id] = ast.literal_eval(node.value)
        except ValueError:
            continue
    return out


def _advertised_capabilities(source: str) -> list[str]:
    """The capability strings /healthz actually returns.

    Found by locating the "capabilities" key in the parsed source, so an explanatory comment naming
    a capability the backend deliberately withholds is not mistaken for advertising it.
    """
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=False):
            if (
                isinstance(key, ast.Constant)
                and key.value == "capabilities"
                and isinstance(value, ast.List)
            ):
                return [
                    item.value
                    for item in value.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                ]
    raise AssertionError("the generated app advertises no capabilities")


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_generated_app_is_valid_python(model_id):
    ast.parse(render_app(MODELS[model_id]))


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_every_template_placeholder_is_substituted(model_id):
    """A missed placeholder must not survive as a literal brace in the output.

    The generated app is full of runtime f-strings, so a leftover `{gpu}` still parses -- it would
    just deploy with a broken value. Checking that every remaining brace sits inside an f-string
    catches it here instead of on the GPU.
    """
    source = render_app(MODELS[model_id])
    spans = [
        (node.lineno, node.end_lineno)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.JoinedStr)
    ]
    stray = [
        (lineno, line.strip())
        for lineno, line in enumerate(source.splitlines(), 1)
        if re.search(r"\{[a-z_]+\}", line)
        and not any(start <= lineno <= end for start, end in spans)
    ]
    assert stray == []


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_engine_values_come_from_the_catalog(model_id):
    """The app must deploy the configuration Freesolo validated, not a computed guess."""
    info = MODELS[model_id]
    values = _constants(render_app(info))
    assert values["BASE_MODEL"] == info.id
    assert values["GPU"] == info.serving.gpu
    assert values["MAX_MODEL_LEN"] == info.serving.max_model_len
    assert values["MAX_LORAS"] == info.serving.max_loras
    assert values["MAX_LORA_RANK"] == info.serving.max_lora_rank
    assert values["VLLM_VERSION"] == VLLM_VERSION


def test_moe_renders_bf16_and_dense_renders_fp8():
    """Quantization is per model and getting it wrong is not a performance detail.

    The 35B-A3B MoE will not compile its fused-expert LoRA path on fp8, and its LoRA pool only fits
    the H200 at six hot adapters. `None` is what tells vLLM to load the checkpoint as-is; a dtype
    string here would request online quantization instead.
    """
    moe = _constants(render_app(MODELS["Qwen/Qwen3.6-35B-A3B"]))
    assert moe["QUANTIZATION"] is None
    assert moe["MAX_LORAS"] == 6
    dense = _constants(render_app(MODELS["Qwen/Qwen3.5-4B"]))
    assert dense["QUANTIZATION"] == "fp8"


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_no_private_repo_or_platform_coupling(model_id):
    """A self-hoster has no access to Freesolo's private FP8 repos or its database.

    fp8 comes from vLLM online-quantizing the PUBLIC checkpoint at load, so the generated app must
    reference neither. Supabase likewise: durable state here is a modal.Dict.
    """
    source = render_app(MODELS[model_id])
    for forbidden in ("Freesolo-Co/", "supabase", "SUPABASE", "postgrest"):
        assert forbidden not in source


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_advertises_exactly_the_capabilities_it_implements(model_id):
    """Advertising a capability flash requires is a promise the backend has to keep.

    thinking_structured_outputs_deferred_v1 must be ABSENT: deferring a grammar until after the
    reasoning block is not implemented here, and advertising it would let flash deploy a
    thinking+structured-output adapter that then serves invalid output.

    Read from the parsed capabilities LIST rather than by substring: the source also explains in a
    comment why that capability is withheld, and a substring check cannot tell a comment from a
    promise.
    """
    advertised = _advertised_capabilities(render_app(MODELS[model_id]))
    assert "immutable_adapter_revisions" in advertised
    assert "alias_compare_and_swap" in advertised
    assert "thinking_structured_outputs_deferred_v1" not in advertised


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_implements_the_endpoints_the_client_calls(model_id):
    source = render_app(MODELS[model_id])
    for route in (
        '"/healthz"',
        '"/adapters"',
        '"/adapters/{adapter_id:path}"',
        '"/adapters/{revision_id:path}/activate"',
        '"/v1/chat/completions"',
    ):
        assert route in source


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_retryable_unavailable_envelope_is_returned_not_raised(model_id):
    """flash matches a TOP-LEVEL "error" object to decide a 503 is worth retrying.

    FastAPI renders a raised HTTPException as {"detail": ...}, which buries the envelope one level
    down and makes the client treat a cold start as a hard deploy failure. So this path must return
    a JSONResponse.
    """
    source = render_app(MODELS[model_id])
    assert "JSONResponse(" in source
    assert '"adapter_unavailable"' in source
    assert '"adapter_loading"' in source


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_serving_key_is_compared_in_constant_time(model_id):
    source = render_app(MODELS[model_id])
    assert "hmac.compare_digest" in source
    # hashlib has no compare_digest; that spelling raises AttributeError on every authed request.
    assert "hashlib.compare_digest" not in source


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_alias_swap_holds_a_lock(model_id):
    """`alias_compare_and_swap` is advertised, so the swap has to be genuinely atomic.

    Reading the current target, checking the caller's expectation, then writing is a lost-update
    race: two concurrent deploys of one run both observe the same previous revision, both pass the
    check, and the second silently discards the first.
    """
    source = render_app(MODELS[model_id])
    assert "_run_lock(" in source
    assert "skip_if_exists=True" in source


def test_app_name_is_a_valid_modal_identifier():
    for model_id in _MODEL_IDS:
        name = app_name_for(model_id)
        assert re.fullmatch(r"[a-z0-9-]+", name), name
    assert app_name_for("Qwen/Qwen3.5-4B") == "flash-serve-qwen3-5-4b"


def test_a_non_default_gpu_keeps_the_catalog_engine_values():
    """Choosing a bigger card is supported; silently re-tuning the engine for it is not.

    max_loras and max_lora_rank were validated per model, and raising them changes what vLLM
    pre-allocates. The card is the user's choice; the engine config stays the validated one.
    """
    info = MODELS["Qwen/Qwen3.5-4B"]
    values = _constants(render_app(info, gpu=MODAL_GPUS_BY_NAME["H100"]))
    assert values["GPU"] == "H100"
    assert values["MAX_LORAS"] == info.serving.max_loras
    assert values["MAX_LORA_RANK"] == info.serving.max_lora_rank


def test_write_app_refuses_to_clobber_an_edited_file(tmp_path):
    """The generated app is meant to be edited, so overwriting it silently would destroy work."""
    destination = tmp_path / "flash_serving_app.py"
    write_app(MODELS["Qwen/Qwen3.5-4B"], destination)
    destination.write_text("# my local edits\n")
    with pytest.raises(FileExistsError):
        write_app(MODELS["Qwen/Qwen3.5-4B"], destination)
    assert destination.read_text() == "# my local edits\n"
    write_app(MODELS["Qwen/Qwen3.5-4B"], destination, overwrite=True)
    assert "flash-serve-qwen3-5-4b" in destination.read_text()


def test_generated_app_names_itself_in_its_deploy_instructions(tmp_path):
    """The header tells the user what to run; a wrong filename there sends them to a missing file."""
    destination = tmp_path / "my_serving_app.py"
    write_app(MODELS["Qwen/Qwen3.5-4B"], destination)
    assert "modal deploy my_serving_app.py" in destination.read_text()
