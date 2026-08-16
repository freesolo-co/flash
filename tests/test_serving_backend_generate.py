"""Rendering the self-hosted Modal serving app.

The generated file is deployed to a real GPU, where a mistake costs a cold start and real money to
discover. So these assertions cover the things that would otherwise only fail on hardware:
that it parses, that its engine values are the catalog's validated ones, and that the contract
details flash's client depends on are actually present.
"""

from __future__ import annotations

import ast
import inspect
import re
import sys
import tomllib
from importlib import resources
from pathlib import Path

import pytest

import flash.serve.backend.generate as generate_module
from flash._internal.channel import DIST_NAME
from flash.content.multimodal import _IMAGE_BLOCK_TYPES
from flash.core.catalog import MODELS
from flash.serve.backend.generate import (
    app_name_for,
    flash_requirement,
    render_app,
    write_app,
)
from flash.serve.backend.gpus import MODAL_GPUS_BY_NAME
from flash.serve.modal import APT_PACKAGES, CUDA_IMAGE

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
    return list(_constants(source)["SERVING_CAPABILITIES"])


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_generated_app_is_valid_python(model_id):
    ast.parse(render_app(MODELS[model_id]))


def test_raw_resource_is_ordinary_python_with_one_config_marker():
    raw = resources.files("flash.serve.backend.templates").joinpath("modal_app.py.tmpl").read_text()
    ast.parse(raw)
    assert raw.count("# flash generated config") == 1
    assert "{{" not in raw
    assert "}}" not in raw
    assert ".format(" not in raw


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_generated_config_marker_is_fully_replaced(model_id):
    source = render_app(MODELS[model_id])
    ast.parse(source)
    assert "# flash generated config" not in source


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
    # the gpu image installs this exact flash build, so vllm is pinned by the distribution rather
    # than by a second constant that could drift from the runtime the app imports.
    assert values["FLASH_REQUIREMENT"] == flash_requirement()
    assert "[serve-runtime]==" in values["FLASH_REQUIREMENT"]


def test_the_pinned_distribution_follows_the_release_channel():
    """the two channels ship under different distribution names.

    reading the production name unconditionally would make `flash serve setup` unable to resolve
    its own version on the dev channel, and -- if it did resolve -- would pin the GPU container to
    the production build a dev-channel user is deliberately not running.

    `CHANNEL` is a source constant that scripts/build_dev_dist.py rewrites at build time, so
    asserting against the imported `DIST_NAME` on this prod checkout would hold for a hardcoded
    production name too. Reading the source is what actually distinguishes the two.
    """
    source = inspect.getsource(generate_module.flash_requirement)
    assert "DIST_NAME" in source, (
        "flash_requirement must resolve the distribution through channel.DIST_NAME; a hardcoded "
        "name fails to resolve on the dev channel, where the package is freesolo-flash-dev"
    )
    assert '"freesolo-flash"' not in source
    assert "'freesolo-flash'" not in source
    # and it still produces a usable requirement on this (prod) checkout.
    assert flash_requirement().startswith(f"{DIST_NAME}[serve-runtime]==")


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

    # Asked of the running interpreter rather than hand-listed. A literal set has to be edited
    # every time the template picks up another stdlib module, and forgetting turns an ordinary
    # `import shutil` into "[serve-modal] does not install ['shutil']" -- a failure that names the
    # wrong problem and cannot be fixed by adding the package it asks for. The names are
    # version-specific, so on 3.11 this still rejects a module that is only stdlib in 3.12, which
    # is what `requires-python = ">=3.11"` means.
    # `flash` itself is the distribution providing this extra, so it is always importable. The
    # runtime and modal helpers it reaches at module scope are import-light by design and pull in
    # neither modal nor the gpu stack.
    missing = sorted(imported - sys.stdlib_module_names - packages - {"flash"})
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
def test_alias_swap_uses_one_api_container_and_a_process_local_lock(model_id):
    """`alias_compare_and_swap` stays atomic without a distributed lease.

    The generated api is pinned to one container, so every activation shares the same per-run
    asyncio lock. This keeps the client-facing compare-and-swap contract while avoiding lock records
    in the durable adapter store.
    """
    source = render_app(MODELS[model_id])
    assert "def _lock_for_run(" in source
    assert "async with _lock_for_run(run_id):" in source
    assert "max_containers=1" in source
    assert "_run_lock" not in source
    assert "skip_if_exists=True" not in source


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
    # the generated app builds on the shared image, so assert the toolchain where it now lives
    # rather than re-declaring the registry and apt list in every generated file.
    assert "base_image" in {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert CUDA_IMAGE == "nvidia/cuda:12.8.0-devel-ubuntu22.04"
    for tool in ("build-essential", "ninja-build"):
        assert tool in APT_PACKAGES, tool


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_text_only_guard_rejects_every_image_block_the_renderer_accepts(model_id):
    """The 400 must name the same block types `flash/content/multimodal.py` decodes.

    The guard once hardcoded ("image", "image_url") while the renderer had accepted `input_image`
    for as long as it had existed. That block form went straight past a check whose only job is to
    refuse it, reaching the tokenizer with no decoded pixels and no `multi_modal_data`: the model
    answered having never seen the image, which is the exact outcome the guard promises not to
    produce. Asserted against the renderer's own set so a fourth block type cannot reintroduce it.
    """
    source = render_app(MODELS[model_id])
    rejected = set(_constants(source)["IMAGE_BLOCK_TYPES"])
    assert 'block.get("type") in IMAGE_BLOCK_TYPES' in source
    assert rejected == set(_IMAGE_BLOCK_TYPES), (
        f"text-only guard rejects {sorted(rejected)} but the renderer decodes "
        f"{sorted(_IMAGE_BLOCK_TYPES)}"
    )
