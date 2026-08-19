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
import shlex
import sys
import tomllib
from dataclasses import replace
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
from flash.serve.modal import APT_PACKAGES, CUDA_IMAGE

_MODEL_IDS = sorted(MODELS)


def _constants(source: str) -> dict[str, object]:
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
def test_engine_values_come_from_the_catalog(model_id):
    info = MODELS[model_id]
    values = _constants(render_app(info))
    assert values["BASE_MODEL"] == info.id
    assert values["GPU"] == info.serving.gpu
    assert values["MAX_MODEL_LEN"] == info.serving.max_model_len
    assert values["MAX_LORAS"] == info.serving.max_loras
    assert values["MAX_LORA_RANK"] == info.serving.max_lora_rank
    assert values["FLASH_REQUIREMENT"] == flash_requirement()
    assert "[serve-runtime]==" in values["FLASH_REQUIREMENT"]


def test_generation_refuses_a_model_without_an_exact_serving_gpu():
    info = MODELS["Qwen/Qwen3.5-4B"]
    with pytest.raises(ValueError, match="no serving configuration"):
        render_app(replace(info, serving=None))
    with pytest.raises(ValueError, match="no validated serving GPU"):
        render_app(replace(info, serving=replace(info.serving, gpu="   ")))


def test_the_pinned_distribution_follows_the_release_channel():
    source = inspect.getsource(generate_module.flash_requirement)
    assert "DIST_NAME" in source, (
        "flash_requirement must resolve the distribution through channel.DIST_NAME; a hardcoded "
        "name fails to resolve on the dev channel, where the package is freesolo-flash-dev"
    )
    assert '"freesolo-flash"' not in source
    assert "'freesolo-flash'" not in source
    assert flash_requirement().startswith(f"{DIST_NAME}[serve-runtime]==")


def test_the_serve_modal_extra_installs_what_the_generated_app_imports_locally():
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
    missing = sorted(imported - sys.stdlib_module_names - packages - {"flash"})
    assert not missing, f"[serve-modal] does not install {missing}, imported at module scope"
    assert packages == {"modal", "fastapi"}


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_a_catalog_prefill_budget_reaches_the_engine_args(model_id):
    info = MODELS[model_id]
    source = render_app(info)
    budget = info.serving.max_num_batched_tokens or 0
    assert _constants(source)["MAX_NUM_BATCHED_TOKENS"] == budget
    passed = re.search(r'"max_num_batched_tokens":\s*MAX_NUM_BATCHED_TOKENS', source)
    assert passed, "the rendered prefill budget never reaches AsyncEngineArgs"
    assert "if MAX_NUM_BATCHED_TOKENS" in source


def test_moe_renders_bf16_and_keeps_its_lora_budget():
    moe = _constants(render_app(MODELS["Qwen/Qwen3.6-35B-A3B"]))
    assert moe["QUANTIZATION"] is None
    assert moe["MAX_LORAS"] == 6


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_the_generated_app_never_quantizes_the_public_checkpoint(model_id):
    """the catalog's fp8 describes `serve_model_id`, not the weights this app actually loads.

    those pre-quantized repos are private, and this app is forbidden from naming them
    (test_no_private_repo_or_platform_coupling), so it serves the public `info.id` instead.
    passing the catalog quantization through would tell vllm to quantize a different checkpoint
    on the fly, which is an engine shape nobody validated on the listed gpu.
    """

    rendered = _constants(render_app(MODELS[model_id]))
    assert rendered["BASE_MODEL"] == model_id
    assert rendered["QUANTIZATION"] is None
    assert rendered["KV_CACHE_DTYPE"] is None


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_no_private_repo_or_platform_coupling(model_id):
    source = render_app(MODELS[model_id])
    for forbidden in ("Freesolo-Co/", "supabase", "SUPABASE", "postgrest"):
        assert forbidden not in source


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_advertises_exactly_the_capabilities_it_implements(model_id):
    advertised = _advertised_capabilities(render_app(MODELS[model_id]))
    assert "immutable_adapter_revisions" in advertised
    assert "alias_compare_and_swap" in advertised
    assert "thinking_structured_outputs_deferred_v1" not in advertised


def test_app_name_is_a_valid_modal_identifier():
    for model_id in _MODEL_IDS:
        name = app_name_for(model_id)
        assert re.fullmatch(r"[a-z0-9-]+", name), name
    assert app_name_for("Qwen/Qwen3.5-4B") == "flash-serve-qwen3-5-4b"


def test_write_app_refuses_to_clobber_an_edited_file(tmp_path):
    destination = tmp_path / "flash_serving_app.py"
    write_app(MODELS["Qwen/Qwen3.5-4B"], destination)
    destination.write_text("# my local edits\n")
    with pytest.raises(FileExistsError):
        write_app(MODELS["Qwen/Qwen3.5-4B"], destination)
    assert destination.read_text() == "# my local edits\n"
    write_app(MODELS["Qwen/Qwen3.5-4B"], destination, overwrite=True)
    assert "flash-serve-qwen3-5-4b" in destination.read_text()


def test_generated_app_names_its_actual_path_in_deploy_instructions(tmp_path):
    destination = tmp_path / "generated apps" / "my_serving_app.py"
    write_app(MODELS["Qwen/Qwen3.5-4B"], destination)
    expected = f"modal deploy {shlex.quote(str(destination))}"
    assert expected in destination.read_text()


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_registration_settles_through_a_spawned_function(model_id):
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
    assert "lifecycle.settle.spawn(record)" in source


@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_image_carries_the_cuda_toolchain_the_engine_jits_against(model_id):
    source = render_app(MODELS[model_id])
    called = {
        node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "debian_slim" not in called
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
    source = render_app(MODELS[model_id])
    rejected = set(_constants(source)["IMAGE_BLOCK_TYPES"])
    assert 'block.get("type") in IMAGE_BLOCK_TYPES' in source
    assert rejected == set(_IMAGE_BLOCK_TYPES), (
        f"text-only guard rejects {sorted(rejected)} but the renderer decodes "
        f"{sorted(_IMAGE_BLOCK_TYPES)}"
    )
