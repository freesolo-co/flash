"""Rendering the self-hosted Modal serving app.

The generated file is deployed to a real GPU, where a mistake costs a cold start and real money to
discover. So these assertions cover the things that would otherwise only fail on hardware:
that it parses, that its engine values are the catalog's validated ones, and that the contract
details flash's client depends on are actually present.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

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


def test_the_serve_modal_extra_installs_what_the_generated_app_imports_locally():
    """`modal deploy` imports the generated module in the LOCAL interpreter to discover the app.

    So every module-scope import in the template has to be satisfied by `[serve-modal]` itself.
    modal does not depend on fastapi, so a fresh `pip install 'freesolo-flash[serve-modal]'` that
    omits it fails at the documented first deploy with ModuleNotFoundError -- before any remote
    image is ever built, and with nothing in the error naming the missing extra.

    vLLM is deliberately NOT here: it is imported inside the Engine methods, which only ever run
    in the Modal image on the GPU.
    """
    root = Path(__file__).resolve().parents[1]
    extra = tomllib.loads((root / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    packages = {
        re.split(r"[<>=!\[ ]", spec, maxsplit=1)[0].lower() for spec in extra["serve-modal"]
    }

    source = render_app(MODELS["Qwen/Qwen3.5-4B"])
    imported = set()
    for node in ast.parse(source).body:  # module scope only: nested imports run on the GPU
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])

    stdlib = {
        "__future__",
        "asyncio",
        "contextlib",
        "hashlib",
        "hmac",
        "os",
        "time",
        "uuid",
        "typing",
        "json",
        "datetime",
        "dataclasses",
    }
    missing = sorted(imported - stdlib - packages)
    assert not missing, f"[serve-modal] does not install {missing}, imported at module scope"


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_a_catalog_prefill_budget_reaches_the_engine_args(model_id):
    """`max_num_batched_tokens` is a memory bound, not a tuning hint.

    The 35B-A3B is the one model that sets it, and it is what keeps the prefill activation peak off
    a card whose LoRA pool is already at its limit. Rendered as a constant but never passed to
    AsyncEngineArgs, the app deploys with vLLM's much larger default and OOMs during prefill --
    after a multi-minute cold start, on the largest and most expensive GPU in the table.
    """
    info = MODELS[model_id]
    source = render_app(info)
    budget = info.serving.max_num_batched_tokens or 0
    assert _constants(source)["MAX_NUM_BATCHED_TOKENS"] == budget
    passed = re.search(r'"max_num_batched_tokens":\s*MAX_NUM_BATCHED_TOKENS', source)
    assert passed, "the rendered prefill budget never reaches AsyncEngineArgs"
    # Passed only when the catalog sets one: vLLM computes its own default otherwise, and pinning
    # it to 0 would be rejected rather than meaning "no limit".
    assert "if MAX_NUM_BATCHED_TOKENS" in source


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


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_registration_settles_through_a_spawned_function(model_id):
    """Registration must not settle in a background task on the web container's event loop.

    Caught on a real Modal deploy: `asyncio.create_task(_load())` returns 202, then the task dies
    with the container when the web function scales down -- silently, and because the next poll is
    load balanced to a different replica, nothing anywhere notices. Every adapter stays
    `registered` until the deploy times out, with the GPU healthy and idle the whole time.

    Asserted on the AST, not the text, so the comment explaining why `create_task` is wrong does
    not read as the defect itself.

    Scoped to the REQUEST HANDLERS rather than the whole module. A blanket `create_task` ban would
    read as this rule while actually enforcing a stricter one, and would fail on work that is
    legitimately loop-local -- the run lock's lease heartbeat only has to outlive the critical
    section it is renewing, and dying with its container is correct there.
    """
    source = render_app(MODELS[model_id])
    tree = ast.parse(source)
    api = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "api"
    )
    created = [
        node
        for node in ast.walk(api)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_task"
    ]
    assert not created, "registration must outlive the request that scheduled it"
    assert "settle_adapter.spawn(record)" in source


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_durable_state_is_read_and_written_without_blocking_the_event_loop(model_id):
    """Modal Dict calls in an async handler must use `.aio`.

    The blocking form stalls every other request the container is serving -- up to 100 of them --
    while one round trip completes. Modal logs a warning per call; the deployed app logged 49.
    """
    source = render_app(MODELS[model_id])
    blocking = [
        line.strip()
        for line in source.splitlines()
        if "adapter_records." in line and not line.lstrip().startswith("#") and ".aio" not in line
    ]
    assert not blocking, f"blocking Dict calls: {blocking}"


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_image_carries_the_cuda_toolchain_the_engine_jits_against(model_id):
    """vLLM's FlashInfer sampler JIT-compiles a kernel during engine init, so the RUNTIME container
    needs nvcc and a toolchain.

    Caught on a real Modal deploy: on `debian_slim` the weights loaded, the model compiled, and then
    init died with "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist" --
    late enough to read as a mid-boot crash rather than a missing dependency, and invisible to any
    test that stubs the GPU.
    """
    source = render_app(MODELS[model_id])
    # on the AST, so the comment explaining why debian_slim is wrong is not read as the defect
    called = {
        node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "debian_slim" not in called
    assert "from_registry" in called
    assert "nvidia/cuda:12.8.0-devel-ubuntu22.04" in source
    for tool in ("build-essential", "ninja-build"):
        assert tool in source, tool
