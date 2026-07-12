from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from flash.engine.worker.model_interpolation import (
    interpolation_required_disk_gb,
    materialize_interpolation_from_paths,
    materialize_model_interpolation,
    validate_materialized_interpolation,
)
from flash.engine.worker.model_source import assert_model_source_parity
from flash.schema import ConfigError, spec_from_dict
from flash.spec import (
    SUPPORTED_MODEL_INTERPOLATION_PAIR,
    JobSpec,
    ModelInterpolationSpec,
)


def _config(*, hidden_size: int = 4) -> dict:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "hidden_size": hidden_size,
        "tie_word_embeddings": True,
        "text_config": {
            "hidden_size": hidden_size,
            "linear_attention_config": {"conv_kernel_size": 4},
        },
        "vision_config": {"hidden_size": 2},
    }


def _repo(
    root: Path,
    name: str,
    tensors: dict[str, np.ndarray],
    *,
    config: dict | None = None,
    shards: int = 1,
) -> Path:
    repo = root / name
    repo.mkdir()
    (repo / "config.json").write_text(json.dumps(config or _config()))
    (repo / "generation_config.json").write_text('{"max_new_tokens":32}')
    (repo / "tokenizer.json").write_text('{"version":"1.0"}')
    (repo / "tokenizer_config.json").write_text('{"chat_template":"qwen"}')
    (repo / "processor_config.json").write_text('{"processor_class":"QwenProcessor"}')
    (repo / "processing_qwen.py").write_text("REMOTE_CODE = True\n")
    (repo / "visual.py").write_text("VISUAL = True\n")
    (repo / "mtp_config.json").write_text('{"mtp":true}')
    (repo / "gdn_config.json").write_text('{"gdn":true}')

    names = sorted(tensors)
    groups = [names[index::shards] for index in range(shards)]
    weight_map: dict[str, str] = {}
    total = 0
    for index, keys in enumerate(groups, start=1):
        filename = (
            "model.safetensors"
            if shards == 1
            else f"model-{index:05d}-of-{shards:05d}.safetensors"
        )
        save_file({key: tensors[key] for key in keys}, repo / filename)
        for key in keys:
            weight_map[key] = filename
            total += tensors[key].nbytes
    if shards > 1:
        (repo / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map})
        )
    return repo


def _spec(alpha: float = 0.5, source: str = "instruct") -> ModelInterpolationSpec:
    base_model, instruct_model = SUPPORTED_MODEL_INTERPOLATION_PAIR
    return ModelInterpolationSpec(
        base_model=base_model,
        instruct_model=instruct_model,
        alpha=alpha,
        tokenizer_config_source=source,
        base_revision="a" * 40,
        instruct_revision="b" * 40,
    )


def _tensors(offset: float = 0.0) -> dict[str, np.ndarray]:
    embedding = np.arange(12, dtype=np.float32).reshape(3, 4) + offset
    return {
        "model.embed_tokens.weight": embedding,
        "lm_head.weight": embedding.copy(),
        "model.layers.0.weight": np.arange(16, dtype=np.float32).reshape(4, 4) + offset,
        "model.layers.0.flag": np.array([1, 2], dtype=np.int64),
    }


def _read_tensor(output: Path, key: str) -> np.ndarray:
    index = json.loads((output / "model.safetensors.index.json").read_text())
    filename = index["weight_map"][key]
    with safe_open(output / filename, framework="np", device="cpu") as handle:
        return handle.get_tensor(key)


def test_interpolation_spec_parses_and_round_trips() -> None:
    raw = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "environment": {"id": "owner/env"},
        "train": {"epochs": 1, "max_examples": 8},
        "model_initialization": {
            "type": "interpolation",
            "base_model": "Qwen/Qwen3.5-4B-Base",
            "instruct_model": "Qwen/Qwen3.5-4B",
            "alpha": 0.25,
            "tokenizer_config_source": "instruct",
            "base_revision": "a" * 40,
            "instruct_revision": "b" * 40,
        },
    }
    spec = spec_from_dict(raw)
    assert spec.model == "Qwen/Qwen3.5-4B"
    assert spec.model_initialization == ModelInterpolationSpec(
        base_model="Qwen/Qwen3.5-4B-Base",
        instruct_model="Qwen/Qwen3.5-4B",
        alpha=0.25,
        tokenizer_config_source="instruct",
        base_revision="a" * 40,
        instruct_revision="b" * 40,
    )
    assert JobSpec.from_json(spec.to_json()) == spec

    for alpha in (0.0, 0.25, 1.0):
        valid = {**raw, "model_initialization": {**raw["model_initialization"], "alpha": alpha}}
        assert spec_from_dict(valid).model_initialization.alpha == alpha
    for alpha in (-0.01, 1.01, float("nan"), float("inf"), float("-inf"), True):
        bad = {**raw, "model_initialization": {**raw["model_initialization"], "alpha": alpha}}
        with pytest.raises((ConfigError, ValueError), match="alpha"):
            spec_from_dict(bad)
    with pytest.raises((ConfigError, ValueError), match="namespace/name"):
        spec_from_dict(
            {
                **raw,
                "model_initialization": {
                    **raw["model_initialization"],
                    "base_model": "Qwen/../private",
                },
            }
        )
    with pytest.raises((ConfigError, ValueError), match="immutable"):
        spec_from_dict(
            {
                **raw,
                "model_initialization": {
                    **raw["model_initialization"],
                    "base_revision": "A" * 40,
                },
            }
        )
    with pytest.raises(ConfigError, match="only supports"):
        spec_from_dict(
            {
                **raw,
                "model_initialization": {
                    **raw["model_initialization"],
                    "base_model": "Qwen/Qwen3-4B-Base",
                },
            }
        )
    with pytest.raises(ConfigError, match="must equal"):
        spec_from_dict({**raw, "model": "Qwen/Qwen3.5-9B"})
    for key, revision in (
        ("base_revision", "main"),
        ("base_revision", "a" * 39),
        ("instruct_revision", "A" * 40),
        ("instruct_revision", "g" * 40),
    ):
        with pytest.raises(ConfigError, match="immutable"):
            spec_from_dict(
                {
                    **raw,
                    "model_initialization": {
                        **raw["model_initialization"],
                        key: revision,
                    },
                }
            )


@pytest.mark.parametrize(("alpha", "expected_offset"), [(0.0, 0.0), (0.5, 1.0), (1.0, 2.0)])
def test_alpha_endpoints_midpoint_shards_and_full_tree(
    tmp_path: Path, alpha: float, expected_offset: float
) -> None:
    base = _repo(tmp_path, "base", _tensors(0.0), shards=2)
    instruct = _repo(tmp_path, "instruct", _tensors(2.0), shards=2)
    output = tmp_path / f"output-{alpha}"
    result = materialize_interpolation_from_paths(
        _spec(alpha),
        base_path=str(base),
        instruct_path=str(instruct),
        output_dir=str(output),
        base_commit="a" * 40,
        instruct_commit="b" * 40,
        max_shard_bytes=80,
    )
    expected = np.arange(16, dtype=np.float32).reshape(4, 4) + expected_offset
    actual = _read_tensor(output, "model.layers.0.weight").astype(np.float32)
    assert np.array_equal(actual, expected)
    assert len(result.manifest["shards"]) > 1
    assert json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "processing_qwen.py",
        "visual.py",
        "mtp_config.json",
        "gdn_config.json",
    ):
        assert (output / name).read_bytes() == (instruct / name).read_bytes()
    assert result.manifest["formula"] == "W=(1-alpha)*W_base+alpha*W_instruct"
    assert result.manifest["parents"]["base"]["commit"] == "a" * 40
    assert result.manifest["parents"]["instruct"]["commit"] == "b" * 40


def test_warm_start_rejected_for_interpolated_child_and_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "environment": {"id": "owner/env"},
        "train": {"epochs": 1, "max_examples": 8, "init_from_adapter": "source-run"},
        "model_initialization": {
            "type": "interpolation",
            "base_model": "Qwen/Qwen3.5-4B-Base",
            "instruct_model": "Qwen/Qwen3.5-4B",
            "alpha": 0.5,
            "base_revision": "a" * 40,
            "instruct_revision": "b" * 40,
        },
    }
    with pytest.raises(ConfigError, match="exact full checkpoint"):
        spec_from_dict(raw)

    import flash.runner as runner

    source = JobSpec.from_dict(
        {
            **spec_from_dict({**raw, "train": {"epochs": 1, "max_examples": 8}}).to_dict(),
            "run_id": "source-run",
            "train": {"hf_repo": "Freesolo-Co/source", "epochs": 1, "max_examples": 8},
        }
    )
    child_raw = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "environment": {"id": "owner/env"},
            "train": {"epochs": 1, "max_examples": 8, "init_from_adapter": "source-run"},
        }
    ).to_dict()
    child = JobSpec.from_dict(child_raw)
    monkeypatch.setattr(
        runner,
        "get_status",
        lambda run_id: SimpleNamespace(spec=source.to_dict(), billing_context={}, platform_context={}),
    )
    with pytest.raises(ValueError, match="exact full checkpoint"):
        runner._resolve_init_from_adapter(child)


def test_nonfloating_missing_extra_shape_and_config_mismatches(tmp_path: Path) -> None:
    base_tensors = _tensors()

    bad_nonfloat = _tensors(1.0)
    bad_nonfloat["model.layers.0.flag"] = np.array([1, 3], dtype=np.int64)
    base = _repo(tmp_path, "base-nonfloat", base_tensors)
    instruct = _repo(tmp_path, "instruct-nonfloat", bad_nonfloat)
    with pytest.raises(ValueError, match="non-floating tensor mismatch"):
        materialize_interpolation_from_paths(
            _spec(), base_path=str(base), instruct_path=str(instruct), output_dir=str(tmp_path / "o1")
        )

    missing = _tensors(1.0)
    missing.pop("lm_head.weight")
    instruct_missing = _repo(tmp_path, "instruct-missing", missing)
    with pytest.raises(ValueError, match="tensor sets differ"):
        materialize_interpolation_from_paths(
            _spec(),
            base_path=str(base),
            instruct_path=str(instruct_missing),
            output_dir=str(tmp_path / "o2"),
        )

    extra = _tensors(1.0)
    extra["extra.weight"] = np.ones(1, dtype=np.float32)
    instruct_extra = _repo(tmp_path, "instruct-extra", extra)
    with pytest.raises(ValueError, match="tensor sets differ"):
        materialize_interpolation_from_paths(
            _spec(),
            base_path=str(base),
            instruct_path=str(instruct_extra),
            output_dir=str(tmp_path / "o3"),
        )

    shaped = _tensors(1.0)
    shaped["model.layers.0.weight"] = np.ones((5, 4), dtype=np.float32)
    instruct_shape = _repo(tmp_path, "instruct-shape", shaped)
    with pytest.raises(ValueError, match="shape mismatch"):
        materialize_interpolation_from_paths(
            _spec(),
            base_path=str(base),
            instruct_path=str(instruct_shape),
            output_dir=str(tmp_path / "o4"),
        )

    instruct_config = _repo(
        tmp_path, "instruct-config", _tensors(1.0), config=_config(hidden_size=8)
    )
    with pytest.raises(ValueError, match="config fingerprints"):
        materialize_interpolation_from_paths(
            _spec(),
            base_path=str(base),
            instruct_path=str(instruct_config),
            output_dir=str(tmp_path / "o5"),
        )


def test_interpolation_disk_uses_both_immutable_hub_totals_and_temp_headroom() -> None:
    class Api:
        def list_repo_tree(self, *, repo_id, revision, **kwargs):
            assert revision in {"a" * 40, "b" * 40}
            sizes = [2_000_000_000, 500_000_000] if repo_id.endswith("-Base") else [3_000_000_000]
            return [SimpleNamespace(size=size) for size in sizes]

    assert interpolation_required_disk_gb(_spec(), api=Api()) == 19


def test_manifest_is_deterministic(tmp_path: Path) -> None:
    base = _repo(tmp_path, "base", _tensors())
    instruct = _repo(tmp_path, "instruct", _tensors(2.0))
    first = materialize_interpolation_from_paths(
        _spec(), base_path=str(base), instruct_path=str(instruct), output_dir=str(tmp_path / "first")
    )
    second = materialize_interpolation_from_paths(
        _spec(), base_path=str(base), instruct_path=str(instruct), output_dir=str(tmp_path / "second")
    )
    assert first.manifest == second.manifest
    assert first.output_fingerprint == second.output_fingerprint


def test_qwen_multimodal_loader_uses_materialized_source(monkeypatch: pytest.MonkeyPatch) -> None:
    from flash.engine.worker import adapter

    calls: list[str] = []

    class Loader:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append(source)
            return SimpleNamespace(source=source)

    monkeypatch.setattr(adapter, "is_vl_checkpoint", lambda model_id: True)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForImageTextToText=Loader),
    )
    result = adapter.prepare_fresh_lora_base(
        "/tmp/interpolated-qwen",
        "Qwen/Qwen3.5-4B",
        {"dtype": "bfloat16"},
        phase="sft",
    )
    assert result.source == "/tmp/interpolated-qwen"
    assert calls == ["/tmp/interpolated-qwen"]


def test_sft_grpo_opd_share_one_materialized_source() -> None:
    root = Path(__file__).parents[1] / "flash" / "engine" / "worker"
    assert "resolve_model_source(model_id)" in (root / "sft.py").read_text()
    assert "resolve_model_source(model_id)" in (root / "rl.py").read_text()
    assert "resolve_model_source(model_id)" in (root / "opd.py").read_text()
    assert "model_source=model_source" in (root / "rl.py").read_text()
    assert "model_source=rollout_model_source" in (root / "opd.py").read_text()
    for name, model_expr in (
        ("sft.py", "trainer.model"),
        ("rl.py", "trainer.model"),
        ("opd.py", "model"),
    ):
        source = (root / name).read_text()
        assert (
            f"interpolated_checkpoint_intent = _w.finalize_interpolated_training({model_expr}, tok)"
            in source
        )
        assert "interpolated_checkpoint_intent=interpolated_checkpoint_intent" in source
    serving_route = Path(__file__).parents[1] / "flash" / "server" / "routes" / "serving.py"
    assert "interpolated-model runs cannot deploy a LoRA" in serving_route.read_text()
    assert assert_model_source_parity("/tmp/model", "/tmp/model") == "/tmp/model"
    with pytest.raises(RuntimeError, match="parity"):
        assert_model_source_parity("/tmp/trainer", "/tmp/rollout")


def test_checkpoint_intent_hash_and_token_are_deterministic() -> None:
    from flash.serve.model_checkpoints import build_checkpoint_outbox

    intent = {
        "schema": "flash.interpolated_checkpoint_intent",
        "version": 1,
        "model_id": "interp-run-30",
        "base_model": "Qwen/Qwen3.5-4B",
        "model_repo_id": "Freesolo-Co/flash-checkpoint-interp-run-30",
        "model_revision": "d" * 40,
        "tokenizer_repo_id": None,
        "tokenizer_revision": None,
        "thinking": False,
        "structured_outputs": None,
        "private": True,
        "metadata": {
            "schema": "flash.model-interpolation",
            "version": 1,
            "canonical_model": "Qwen/Qwen3.5-4B",
            "formula": "W=(1-alpha)*W_base+alpha*W_instruct",
            "alpha": 0.5,
            "parents": {
                "base": {
                    "model": "Qwen/Qwen3.5-4B-Base",
                    "requested_revision": "a" * 40,
                    "commit": "a" * 40,
                    "config_fingerprint": "1" * 64,
                },
                "instruct": {
                    "model": "Qwen/Qwen3.5-4B",
                    "requested_revision": "b" * 40,
                    "commit": "b" * 40,
                    "config_fingerprint": "2" * 64,
                },
            },
            "tokenizer_config_source": "instruct",
            "interpolation_output_fingerprint": "f" * 64,
            "interpolation_tree_fingerprint": "3" * 64,
            "trained_tree_fingerprint": "e" * 64,
        },
        "output_fingerprint": "e" * 64,
        "interpolation_output_fingerprint": "f" * 64,
    }
    first = build_checkpoint_outbox(intent, run_id="interp-run-30", now=1.0)
    second = build_checkpoint_outbox(intent, run_id="interp-run-30", now=2.0)
    assert first["payload_hash"] == second["payload_hash"]
    assert first["deployment_token"] == second["deployment_token"]
    assert first["activation_state"] == "pending"
    with pytest.raises(ValueError, match="model_id"):
        build_checkpoint_outbox(intent, run_id="different-run", now=1.0)


def test_complete_fingerprint_rejects_asset_and_index_corruption(tmp_path: Path) -> None:
    base = _repo(tmp_path, "base", _tensors())
    instruct = _repo(tmp_path, "instruct", _tensors(2.0))
    output = tmp_path / "output"
    result = materialize_interpolation_from_paths(
        _spec(), base_path=str(base), instruct_path=str(instruct), output_dir=str(output)
    )
    validate_materialized_interpolation(output, result.manifest)

    (output / "processing_qwen.py").write_text("CORRUPT = True\n")
    with pytest.raises(ValueError, match="complete source fingerprint"):
        validate_materialized_interpolation(output, result.manifest)
    (output / "processing_qwen.py").write_text("REMOTE_CODE = True\n")

    index = json.loads((output / "model.safetensors.index.json").read_text())
    index["weight_map"]["model.layers.0.weight"] = "missing.safetensors"
    (output / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="complete source fingerprint"):
        validate_materialized_interpolation(output, result.manifest)


def test_cache_lock_race_atomic_replacement_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    base = _repo(tmp_path, "base", _tensors())
    instruct = _repo(tmp_path, "instruct", _tensors(2.0))
    calls: list[tuple[str, str]] = []
    gate = threading.Lock()

    def snapshot_download(*, repo_id, cache_dir, **kwargs):
        with gate:
            calls.append((repo_id, cache_dir))
        time.sleep(0.02)
        return str(base if repo_id.endswith("-Base") else instruct)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    root = tmp_path / "cache"
    results: list[str] = []

    def run() -> None:
        results.append(materialize_model_interpolation(_spec(), output_root=str(root)).source)

    threads = [threading.Thread(target=run), threading.Thread(target=run)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(results)) == 1
    assert len(calls) == 2
    assert not list(root.glob("*.staging-*"))
    assert not list(root.glob("*.hub-cache-*"))

    output = Path(results[0])
    (output / "visual.py").write_text("corrupt\n")
    materialize_model_interpolation(_spec(), output_root=str(root))
    assert (output / "visual.py").read_text() == "VISUAL = True\n"
    assert len(calls) == 4


def test_worker_parent_policy_rejects_an_unsupported_pair() -> None:
    from flash.engine.worker.model_interpolation import validate_parent_policy

    unsupported = SimpleNamespace(
        base_model="private-org/base",
        instruct_model="private-org/instruct",
    )
    with pytest.raises(ValueError, match="not supported"):
        validate_parent_policy(unsupported)


def test_final_merge_publishes_intent_without_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flash.engine import worker
    from flash.engine.worker.finalize import finalize_interpolated_training
    from flash.engine.worker.model_source import ResolvedModelSource

    base = _repo(tmp_path, "base", _tensors())
    instruct = _repo(tmp_path, "instruct", _tensors(2.0))
    materialized = materialize_interpolation_from_paths(
        _spec(),
        base_path=str(base),
        instruct_path=str(instruct),
        output_dir=str(tmp_path / "source"),
        base_commit="a" * 40,
        instruct_commit="b" * 40,
    )
    monkeypatch.setattr(
        worker,
        "RESOLVED_MODEL_SOURCE",
        ResolvedModelSource(
            canonical_model="Qwen/Qwen3.5-4B",
            source=materialized.source,
            setup_seconds=1.0,
            interpolation=materialized.manifest,
        ),
    )
    monkeypatch.setattr(worker, "RUN_ID", "interp-run-30")
    monkeypatch.setattr(worker, "THINKING", False)

    class Merged:
        def save_pretrained(self, output, **kwargs):
            Path(output, "config.json").write_text(json.dumps(_config()))
            save_file(_tensors(3.0), Path(output, "model.safetensors"))

    class Model:
        def merge_and_unload(self):
            return Merged()

    class Tokenizer:
        def save_pretrained(self, output):
            Path(output, "tokenizer.json").write_text('{"trained":true}')

    class Api:
        def __init__(self):
            self.uploaded_files: set[str] = set()
            self.uploaded = False

        def create_repo(self, **kwargs):
            assert kwargs["repo_id"] == "Freesolo-Co/flash-checkpoint-interp-run-30"
            assert kwargs["private"] is True

        def update_repo_settings(self, **kwargs):
            assert kwargs["private"] is True

        def repo_info(self, **kwargs):
            return SimpleNamespace(sha=("d" if self.uploaded else "c") * 40, private=True)

        def upload_folder(self, *, folder_path, **kwargs):
            self.uploaded_files = {
                path.relative_to(folder_path).as_posix()
                for path in Path(folder_path).rglob("*")
                if path.is_file()
            }
            self.uploaded = True
            return SimpleNamespace(oid="d" * 40)

    api = Api()
    monkeypatch.setattr(worker, "hf_api", lambda: api)
    result = finalize_interpolated_training(Model(), Tokenizer())
    assert result is not None
    assert result["model_revision"] == "d" * 40
    assert result["output_fingerprint"] == result["metadata"]["trained_tree_fingerprint"]
    assert result["interpolation_output_fingerprint"] == materialized.output_fingerprint
    assert result["schema"] == "flash.interpolated_checkpoint_intent"
    assert result["model_id"] == "interp-run-30"
    assert "deployment_token" not in result
    for required in (
        "model.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "tokenizer.json",
        "processor_config.json",
        "processing_qwen.py",
        "visual.py",
        "mtp_config.json",
        "gdn_config.json",
        "flash_interpolation_manifest.json",
    ):
        assert required in api.uploaded_files
