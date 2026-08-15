"""`flash export` core: the HF-to-HF adapter copy and the destination-token resolution.

These run offline by faking ``huggingface_hub`` (the copy is a download-then-upload, so we record
the calls instead of touching the Hub).
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest

BASE_MODEL = "Qwen/Qwen3.5-0.8B"


class _FakeHfHubHTTPError(OSError):
    pass


def _install_fake_hub(monkeypatch, *, download, hf_api):
    """Inject a fake ``huggingface_hub`` module exposing HfApi + snapshot_download."""
    fake = types.ModuleType("huggingface_hub")
    fake_errors = types.ModuleType("huggingface_hub.errors")
    fake_errors.HfHubHTTPError = _FakeHfHubHTTPError
    fake.HfApi = hf_api
    fake.snapshot_download = download
    fake.errors = fake_errors
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", fake_errors)


def _safetensors_bytes(header: dict, data: bytes) -> bytes:
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    return struct.pack("<Q", len(header_bytes)) + header_bytes + data


def _parse_safetensors_bytes(contents: bytes) -> tuple[dict, bytes]:
    (header_length,) = struct.unpack("<Q", contents[:8])
    data_start = 8 + header_length
    return json.loads(contents[8:data_start]), contents[data_start:]


# Export refuses to ship weights whose key namespace it could not read, so a fixture standing in
# for "some adapter weights" has to be a parseable safetensors file rather than an opaque blob.
_PLAIN_WEIGHTS = _safetensors_bytes(
    {
        "base_model.model.model.layers.0.mlp.up_proj.lora_A.default.weight": {
            "dtype": "F16",
            "shape": [1],
            "data_offsets": [0, 2],
        }
    },
    b"\x01\x02",
)


def test_export_import_does_not_initialize_worker_package(tmp_path):
    malformed_spec = tmp_path / "job-spec.json"
    malformed_spec.write_text("not-json", encoding="utf-8")
    env = os.environ.copy()
    env["FLASH_JOB_SPEC_JSON"] = "not-json"
    env["FLASH_JOB_SPEC_PATH"] = str(malformed_spec)
    env["PYTHONPATH"] = str(Path(__file__).parents[1])

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import flash.serve.export; "
                "assert 'flash.engine.worker' not in sys.modules"
            ),
        ],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_export_adapter_reads_source_with_operator_token_writes_dest_with_user_token(monkeypatch):
    calls: dict = {}

    def fake_snapshot_download(*, repo_id, repo_type, allow_patterns, local_dir, token):
        calls["download"] = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "allow_patterns": allow_patterns,
            "token": token,
        }
        # Materialize the adapter folder exactly where export_adapter looks for it.
        adapter = Path(local_dir) / "rl/run-x/seed0/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(_PLAIN_WEIGHTS)
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            calls["dest_token"] = token

        def create_repo(self, *, repo_id, repo_type, private, exist_ok):
            calls["create_repo"] = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "private": private,
                "exist_ok": exist_ok,
            }

        def list_repo_files(self, *, repo_id, repo_type):
            return []  # brand-new repo -> no orphans to clean

        def update_repo_settings(self, *, repo_id, repo_type, private):
            calls["update_settings"] = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "private": private,
            }

        def repo_info(self, **kw):
            return types.SimpleNamespace(sha="parent-sha")

        def upload_folder(
            self,
            *,
            repo_id,
            repo_type,
            folder_path,
            commit_message,
            delete_patterns=None,
            parent_commit=None,
        ):
            calls["upload"] = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "files": sorted(p.name for p in Path(folder_path).iterdir()),
                "delete_patterns": delete_patterns,
                "parent_commit": parent_commit,
            }

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)

    from flash.serve.export import export_adapter

    url = export_adapter(
        source_repo="org/test-runs",
        source_subfolder="rl/run-x/seed0/adapter",
        dest_repo="me/adapters",
        dest_token="hf_user",
        base_model=BASE_MODEL,
        source_token="hf_operator",
        private=True,
    )
    assert url == "https://huggingface.co/me/adapters"
    # Read the PRIVATE source dataset repo with the operator token, only the adapter subfolder.
    assert calls["download"]["repo_id"] == "org/test-runs"
    assert calls["download"]["repo_type"] == "dataset"
    assert calls["download"]["allow_patterns"] == ["rl/run-x/seed0/adapter/*"]
    assert calls["download"]["token"] == "hf_operator"
    # Write the user's MODEL repo with the user's token; create it (private) if missing.
    assert calls["dest_token"] == "hf_user"
    assert calls["create_repo"] == {
        "repo_id": "me/adapters",
        "repo_type": "model",
        "private": True,
        "exist_ok": True,
    }
    assert calls["upload"]["repo_id"] == "me/adapters"
    assert calls["upload"]["repo_type"] == "model"
    assert set(calls["upload"]["files"]) == {"adapter_config.json", "adapter_model.safetensors"}
    # The repo is created PRIVATE, then visibility is enforced after the upload commits (create_repo(
    # exist_ok=True) won't change an existing repo's visibility).
    assert calls["create_repo"]["private"] is True
    assert calls["update_settings"] == {
        "repo_id": "me/adapters",
        "repo_type": "model",
        "private": True,
    }
    # Stale adapter artifacts are cleared by STATIC, adapter-scoped patterns (never the user's
    # unrelated files), so a re-export of the same format is a no-op delete.
    assert calls["upload"]["delete_patterns"] == ["adapter_model*", "adapter_config.json"]
    # The commit is pinned to the repo's current head (concurrent-export guard).
    assert calls["upload"]["parent_commit"] == "parent-sha"


def test_export_adapter_normalizes_safetensors_keys_for_vanilla_peft(monkeypatch):
    uploaded: dict = {}
    infixed_a = "base_model.model.model.language_model.layers.0.mlp.gate_proj.lora_A.default.weight"
    infixed_b = "base_model.model.model.language_model.layers.0.mlp.gate_proj.lora_B.default.weight"
    plain = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    double_infix = (
        "base_model.model.model.language_model.layers.0.language_model.mlp.up_proj."
        "lora_A.default.weight"
    )
    header = {
        "__metadata__": {"format": "pt", "note": "preserve exactly"},
        infixed_a: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
        plain: {"dtype": "F16", "shape": [1], "data_offsets": [2, 4]},
        infixed_b: {"dtype": "F16", "shape": [1], "data_offsets": [4, 6]},
        double_infix: {"dtype": "F16", "shape": [1], "data_offsets": [6, 8]},
    }
    data = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    source_bytes = _safetensors_bytes(header, data)

    def fake_snapshot_download(*, local_dir, **kw):
        adapter = Path(local_dir) / "rl/run-x/seed0/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(source_bytes)
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kw):
            pass

        def update_repo_settings(self, **kw):
            pass

        def repo_info(self, **kw):
            return types.SimpleNamespace(sha="parent-sha")

        def upload_folder(self, *, folder_path, **kw):
            uploaded["weights"] = (Path(folder_path) / "adapter_model.safetensors").read_bytes()

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    export_adapter(
        source_repo="org/test-runs",
        source_subfolder="rl/run-x/seed0/adapter",
        dest_repo="me/adapters",
        dest_token="hf_user",
        base_model=BASE_MODEL,
        source_token="hf_operator",
    )

    uploaded_header, uploaded_data = _parse_safetensors_bytes(uploaded["weights"])
    expected_a = infixed_a.replace(".language_model.", ".", 1)
    expected_b = infixed_b.replace(".language_model.", ".", 1)
    expected_double = double_infix.replace(".language_model.", ".", 1)
    assert set(uploaded_header) == {
        "__metadata__",
        expected_a,
        plain,
        expected_b,
        expected_double,
    }
    assert uploaded_header["__metadata__"] == header["__metadata__"]
    assert uploaded_header[expected_a] == header[infixed_a]
    assert uploaded_header[plain] == header[plain]
    assert uploaded_header[expected_b] == header[infixed_b]
    assert uploaded_header[expected_double] == header[double_infix]
    assert ".language_model." in expected_double
    assert uploaded_data == data


def test_export_adapter_key_collision_fails_the_export(monkeypatch):
    """A rename that would shadow an existing key cannot be applied, so nothing is uploaded.

    Normalization is needed here (the infix is present) and cannot be performed, which is exactly
    the case where shipping the weights hands the user something peft loads as a no-op."""
    uploaded: dict = {}
    infixed = "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight"
    plain = infixed.replace(".language_model.", ".", 1)
    source_bytes = _safetensors_bytes(
        {
            infixed: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
            plain: {"dtype": "F16", "shape": [1], "data_offsets": [2, 4]},
        },
        b"\x01\x02\x03\x04",
    )

    def fake_snapshot_download(*, local_dir, **kw):
        adapter = Path(local_dir) / "rl/run-x/seed0/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(source_bytes)
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kw):
            pass

        def update_repo_settings(self, **kw):
            pass

        def repo_info(self, **kw):
            return types.SimpleNamespace(sha="parent-sha")

        def upload_folder(self, *, folder_path, **kw):
            uploaded["weights"] = (Path(folder_path) / "adapter_model.safetensors").read_bytes()

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.errors import ServingError
    from flash.serve.export import export_adapter

    with pytest.raises(ServingError, match="collides"):
        export_adapter(
            source_repo="org/test-runs",
            source_subfolder="rl/run-x/seed0/adapter",
            dest_repo="me/adapters",
            dest_token="hf_user",
            base_model=BASE_MODEL,
            source_token="hf_operator",
        )

    assert "weights" not in uploaded


def test_export_adapter_normalizes_lm_keys_with_inert_vision_weights(tmp_path):
    from flash.serve import export

    lm_a = "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight"
    lm_b = "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_B.default.weight"
    vision_a = "base_model.model.model.visual.blocks.0.attn.qkv.lora_A.default.weight"
    vision_b = "base_model.model.model.visual.blocks.0.attn.qkv.lora_B.default.weight"
    header = {
        lm_a: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
        lm_b: {"dtype": "F16", "shape": [1], "data_offsets": [2, 4]},
        vision_a: {"dtype": "F16", "shape": [1], "data_offsets": [4, 6]},
        vision_b: {"dtype": "F16", "shape": [1], "data_offsets": [6, 8]},
    }
    data = b"\x01\x02\x03\x04\x05\x06\x00\x00"
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(_safetensors_bytes(header, data))

    assert export._normalize_export_adapter_keys(tmp_path) == "text_only"

    normalized_header, normalized_data = _parse_safetensors_bytes(path.read_bytes())
    normalized_lm_a = lm_a.replace(".language_model.", ".", 1)
    normalized_lm_b = lm_b.replace(".language_model.", ".", 1)
    assert set(normalized_header) == {normalized_lm_a, normalized_lm_b, vision_a, vision_b}
    assert normalized_header[normalized_lm_a] == header[lm_a]
    assert normalized_header[normalized_lm_b] == header[lm_b]
    assert normalized_header[vision_a] == header[vision_a]
    assert normalized_header[vision_b] == header[vision_b]
    assert normalized_data == data


def test_export_adapter_with_vision_keys_leaves_safetensors_unchanged(monkeypatch):
    # a genuinely multimodal adapter with nonzero vision lora_b must not be normalized:
    # stripping only its lm keys would leave a mixed namespace no transformers class loads.
    uploaded: dict = {}
    lm_a = "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight"
    lm_b = "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_B.default.weight"
    vision_a = "base_model.model.model.visual.blocks.0.attn.proj.lora_A.default.weight"
    vision_b = "base_model.model.model.visual.blocks.0.attn.proj.lora_B.default.weight"
    source_bytes = _safetensors_bytes(
        {
            lm_a: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
            lm_b: {"dtype": "F16", "shape": [1], "data_offsets": [2, 4]},
            vision_a: {"dtype": "F16", "shape": [1], "data_offsets": [4, 6]},
            vision_b: {"dtype": "F16", "shape": [1], "data_offsets": [6, 8]},
        },
        b"\x01\x02\x03\x04\x05\x06\x07\x08",
    )

    def fake_snapshot_download(*, local_dir, **kw):
        adapter = Path(local_dir) / "rl/run-x/seed0/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(source_bytes)
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kw):
            pass

        def update_repo_settings(self, **kw):
            pass

        def repo_info(self, **kw):
            return types.SimpleNamespace(sha="parent-sha")

        def upload_folder(self, *, folder_path, **kw):
            uploaded["weights"] = (Path(folder_path) / "adapter_model.safetensors").read_bytes()

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    export_adapter(
        source_repo="org/test-runs",
        source_subfolder="rl/run-x/seed0/adapter",
        dest_repo="me/adapters",
        dest_token="hf_user",
        base_model=BASE_MODEL,
        source_token="hf_operator",
    )

    assert uploaded["weights"] == source_bytes


class _FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def any(self):
        return any(self.values)


def _install_fake_torch(monkeypatch):
    """Inject a minimal ``torch``: load/save/Tensor, the whole surface the .bin path touches.

    torch is a ``gpu`` extra and is not installed alongside the control plane, so faking it is the
    only way this path runs in CI at all. The byte scan that decides whether to import torch still
    reads the real file, so the round trip below is a real one.
    """
    fake = types.ModuleType("torch")
    fake.Tensor = _FakeTensor

    def load(path, map_location=None, weights_only=False):
        assert weights_only is True, "adapter state dicts must be loaded weights-only"
        return {
            key: _FakeTensor(values) for key, values in json.loads(Path(path).read_bytes()).items()
        }

    def save(state, destination):
        destination.write(
            json.dumps({key: tensor.values for key, tensor in state.items()}).encode("utf-8")
        )

    fake.load = load
    fake.save = save
    monkeypatch.setitem(sys.modules, "torch", fake)


def _write_fake_bin(path: Path, state: dict) -> None:
    path.write_bytes(json.dumps(state).encode("utf-8"))


def test_export_adapter_normalizes_bin_keys_for_vanilla_peft(tmp_path, monkeypatch):
    """`adapter_model.bin` is a shape the rest of the pipeline accepts, so it gets normalized too.

    Skipping it exports a VL adapter whose keys peft binds to nothing."""
    from flash.serve import export

    infixed = "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight"
    plain = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
    _install_fake_torch(monkeypatch)
    _write_fake_bin(tmp_path / "adapter_model.bin", {infixed: [1.0], plain: [2.0]})

    assert export._normalize_export_adapter_keys(tmp_path) == "text_only"

    stripped = infixed.replace(".language_model.", ".", 1)
    state = json.loads((tmp_path / "adapter_model.bin").read_bytes())
    assert set(state) == {stripped, plain}
    assert state[stripped] == [1.0]
    assert state[plain] == [2.0]


def test_export_adapter_bin_with_live_vision_weights_keeps_multimodal_namespace(
    tmp_path, monkeypatch
):
    from flash.serve import export

    infixed = "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight"
    vision_b = "base_model.model.model.visual.blocks.0.attn.proj.lora_B.default.weight"
    _install_fake_torch(monkeypatch)
    path = tmp_path / "adapter_model.bin"
    _write_fake_bin(path, {infixed: [1.0], vision_b: [0.5]})
    before = path.read_bytes()

    assert export._normalize_export_adapter_keys(tmp_path) == "multimodal"
    assert path.read_bytes() == before


def test_export_adapter_bin_needing_normalization_without_torch_is_refused(tmp_path, monkeypatch):
    """Normalization is needed here and cannot be applied, which fails invisibly if allowed.

    peft would warn and apply nothing, so the user benchmarks the base model believing it is their
    adapter. Refuse the export instead of logging a warning nobody sees."""
    from flash.serve import export
    from flash.serve.errors import ServingError

    infixed = "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight"
    path = tmp_path / "adapter_model.bin"
    _write_fake_bin(path, {infixed: [1.0]})
    before = path.read_bytes()
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(ServingError, match="torch is not installed"):
        export._normalize_export_adapter_keys(tmp_path)
    assert path.read_bytes() == before


def test_export_adapter_bin_without_the_infix_never_imports_torch(tmp_path, monkeypatch):
    """The control plane installs no torch (it is a `gpu` extra), so a `.bin` that cannot need a
    rename must not reach for it. Importing torch here would fail every legacy `.bin` export."""
    from flash.serve import export

    plain = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
    _write_fake_bin(tmp_path / "adapter_model.bin", {plain: [1.0]})
    monkeypatch.setattr(
        export, "_torch_for", lambda path: pytest.fail(f"torch imported for {path.name}")
    )

    assert export._normalize_export_adapter_keys(tmp_path) == "text_only"


def test_export_adapter_bin_shard_without_the_infix_still_pins_the_namespace(tmp_path, monkeypatch):
    """A shard that never mentions the infix still carries keys the global plan has to see.

    Sharding splits ONE key namespace across files, so liveness is a property of their union: a live
    vision LoRA-B in the second shard means the first shard's language-model keys must stay put.
    Reading only the shard that mentions the infix strips them anyway and ships a mixed namespace
    peft binds to nothing."""
    from flash.serve import export

    infixed = "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight"
    vision_b = "base_model.model.model.visual.blocks.0.attn.proj.lora_B.default.weight"
    _install_fake_torch(monkeypatch)
    first = tmp_path / "adapter_model-00001-of-00002.bin"
    second = tmp_path / "adapter_model-00002-of-00002.bin"
    _write_fake_bin(first, {infixed: [1.0]})
    _write_fake_bin(second, {vision_b: [0.5]})
    # peft writes the index whenever it shards, and it is what makes the shards a representation
    # peft can discover at all -- without it there is nothing to export.
    (tmp_path / "adapter_model.bin.index.json").write_text(
        json.dumps({"weight_map": {infixed: first.name, vision_b: second.name}})
    )
    before = (first.read_bytes(), second.read_bytes())

    assert export._normalize_export_adapter_keys(tmp_path) == "multimodal"
    assert (first.read_bytes(), second.read_bytes()) == before


def test_export_adapter_bin_shard_collision_across_shards_is_refused(tmp_path, monkeypatch):
    """The rename is planned over the union of the shards, so the collision is only visible there.

    Missing it rewrites the infixed key onto a name the other shard already uses, and the index then
    maps one key to two shards -- entries silently collapse instead of the export failing."""
    from flash.serve import export
    from flash.serve.errors import ServingError

    infixed = "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight"
    plain = infixed.replace(".language_model.", ".", 1)
    _install_fake_torch(monkeypatch)
    first = tmp_path / "adapter_model-00001-of-00002.bin"
    second = tmp_path / "adapter_model-00002-of-00002.bin"
    _write_fake_bin(first, {infixed: [1.0]})
    _write_fake_bin(second, {plain: [2.0]})
    (tmp_path / "adapter_model.bin.index.json").write_text(
        json.dumps({"weight_map": {infixed: first.name, plain: second.name}})
    )
    before = (first.read_bytes(), second.read_bytes())

    with pytest.raises(ServingError, match="collides"):
        export._normalize_export_adapter_keys(tmp_path)
    assert (first.read_bytes(), second.read_bytes()) == before


def test_export_adapter_normalizes_only_the_representation_peft_loads(tmp_path, monkeypatch):
    """A retry's adapter lands on the previous attempt's path without deleting what it did not
    write, so a stale `.bin` can outlive the safetensors peft actually binds.

    Scanning both reads one namespace out of two: every shared key collides with itself and a valid
    export fails. peft takes safetensors over `.bin`, so the exporter normalizes exactly that."""
    from flash.serve import export

    infixed = "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight"
    stale = tmp_path / "adapter_model.bin"
    _write_fake_bin(stale, {infixed: [9.0]})
    before = stale.read_bytes()
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(
        _safetensors_bytes(
            {infixed: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}, b"\x01\x02"
        )
    )
    monkeypatch.setattr(
        export, "_torch_for", lambda p: pytest.fail(f"read the stale representation {p.name}")
    )

    assert export._normalize_export_adapter_keys(tmp_path) == "text_only"

    assert set(_parse_safetensors_bytes(path.read_bytes())[0]) == {
        infixed.replace(".language_model.", ".", 1)
    }
    assert stale.read_bytes() == before


def test_export_adapter_normalizes_sharded_safetensors_and_index(monkeypatch):
    """Sharded weights are a shape serving validation accepts, so the export must remap them all.

    Every shard is rewritten and the index that maps keys to shards is rewritten with them, so the
    two never disagree about which keys exist."""
    uploaded: dict = {}
    infixed_a = "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight"
    infixed_b = "base_model.model.model.language_model.layers.1.mlp.up_proj.lora_B.default.weight"
    shard_names = (
        "adapter_model-00001-of-00002.safetensors",
        "adapter_model-00002-of-00002.safetensors",
    )
    shard_bytes = (
        _safetensors_bytes(
            {infixed_a: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}, b"\x01\x02"
        ),
        _safetensors_bytes(
            {infixed_b: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}, b"\x03\x04"
        ),
    )

    def fake_snapshot_download(*, local_dir, **kw):
        adapter = Path(local_dir) / "rl/run-x/seed0/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")
        for name, contents in zip(shard_names, shard_bytes, strict=True):
            (adapter / name).write_bytes(contents)
        (adapter / "adapter_model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 4},
                    "weight_map": {infixed_a: shard_names[0], infixed_b: shard_names[1]},
                }
            )
        )
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kw):
            pass

        def update_repo_settings(self, **kw):
            pass

        def repo_info(self, **kw):
            return types.SimpleNamespace(sha="parent-sha")

        def upload_folder(self, *, folder_path, **kw):
            folder = Path(folder_path)
            uploaded["shards"] = [(folder / name).read_bytes() for name in shard_names]
            uploaded["index"] = json.loads(
                (folder / "adapter_model.safetensors.index.json").read_text()
            )

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    export_adapter(
        source_repo="org/test-runs",
        source_subfolder="rl/run-x/seed0/adapter",
        dest_repo="me/adapters",
        dest_token="hf_user",
        base_model=BASE_MODEL,
        source_token="hf_operator",
    )

    expected_a = infixed_a.replace(".language_model.", ".", 1)
    expected_b = infixed_b.replace(".language_model.", ".", 1)
    header_a, data_a = _parse_safetensors_bytes(uploaded["shards"][0])
    header_b, data_b = _parse_safetensors_bytes(uploaded["shards"][1])
    assert set(header_a) == {expected_a}
    assert set(header_b) == {expected_b}
    # tensor payloads are copied byte-for-byte; only the header key names move.
    assert (data_a, data_b) == (b"\x01\x02", b"\x03\x04")
    assert uploaded["index"]["weight_map"] == {
        expected_a: shard_names[0],
        expected_b: shard_names[1],
    }
    assert uploaded["index"]["metadata"] == {"total_size": 4}
    # the index and the shards agree on the key set, which is the whole point of rewriting both.
    assert set(uploaded["index"]["weight_map"]) == set(header_a) | set(header_b)


def test_stale_shards_left_by_a_shorter_retry_are_not_scanned(tmp_path):
    """The index names the live shard set; same-suffix leftovers from a longer attempt do not.

    hf_upload_folder writes a retry's adapter over the previous attempt without deleting what it no
    longer writes, so a 3-shard attempt followed by a 2-shard one leaves shard 3 on disk. Scanning
    it would let its dead visual tensor pin the live text-only shards to the multimodal namespace,
    exporting keys peft loads as a no-op.
    """
    from flash.serve import export

    live_key = "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight"
    stale_visual = "base_model.model.model.visual.blocks.0.attn.proj.lora_B.default.weight"
    live_names = ("adapter_model-00001-of-00002.safetensors",)
    (tmp_path / live_names[0]).write_bytes(
        _safetensors_bytes(
            {live_key: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}, b"\x01\x02"
        )
    )
    # the orphan from the previous, longer attempt: still on disk, absent from the current index.
    (tmp_path / "adapter_model-00003-of-00003.safetensors").write_bytes(
        _safetensors_bytes(
            {stale_visual: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}, b"\x09\x09"
        )
    )
    (tmp_path / "adapter_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {live_key: live_names[0]}})
    )

    assert [p.name for p in export._adapter_weight_paths(tmp_path)] == list(live_names)
    # the stale visual tensor must not pin the live text-only weights to the multimodal namespace
    assert export._normalize_export_adapter_keys(tmp_path) == "text_only"


def test_index_rewrite_leaves_the_unselected_representation_alone(tmp_path):
    """Only the selected representation's shards are rewritten, so only its index may be remapped.

    Remapping the other suffix's index would leave it naming keys its own untouched shards do not
    contain -- the index/payload disagreement the rewrite exists to prevent.
    """
    from flash.serve import export

    infixed = "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight"
    shard = "adapter_model-00001-of-00001.safetensors"
    (tmp_path / shard).write_bytes(
        _safetensors_bytes(
            {infixed: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}, b"\x01\x02"
        )
    )
    (tmp_path / "adapter_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {infixed: shard}})
    )
    # a .bin index for shards that were NOT selected and therefore never rewritten
    bin_index = tmp_path / "adapter_model.bin.index.json"
    bin_index.write_text(json.dumps({"weight_map": {infixed: "adapter_model-00001-of-00001.bin"}}))

    assert export._normalize_export_adapter_keys(tmp_path) == "text_only"

    expected = infixed.replace(".language_model.", ".", 1)
    safetensors_map = json.loads((tmp_path / "adapter_model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    assert set(safetensors_map) == {expected}  # the selected index tracks its rewritten shard
    # the untouched representation's index still names the keys ITS shards actually contain
    assert set(json.loads(bin_index.read_text())["weight_map"]) == {infixed}


def test_export_adapter_with_out_of_bounds_non_lm_offsets_is_refused(tmp_path):
    """A header we cannot read is a namespace we cannot vouch for, so the export fails.

    The weights stay untouched on disk, but the caller gets an error instead of a silent copy of an
    adapter whose keys may never bind to anything."""
    from flash.serve import export
    from flash.serve.errors import ServingError

    lm_key = "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight"
    vision_b = "base_model.model.model.visual.blocks.0.attn.proj.lora_B.default.weight"
    source_bytes = _safetensors_bytes(
        {
            lm_key: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
            vision_b: {"dtype": "F16", "shape": [1], "data_offsets": [2, 5]},
        },
        b"\x01\x02\x00\x00",
    )
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(source_bytes)

    with pytest.raises(ServingError, match="could not normalize exported adapter keys"):
        export._normalize_export_adapter_keys(tmp_path)
    assert path.read_bytes() == source_bytes


def test_export_adapter_with_unrecognized_non_lm_tensor_is_unchanged(tmp_path):
    from flash.serve import export

    lm_key = "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight"
    vision_saved = "base_model.model.model.visual.proj.modules_to_save.default.weight"
    source_bytes = _safetensors_bytes(
        {
            lm_key: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
            vision_saved: {"dtype": "F16", "shape": [1], "data_offsets": [2, 4]},
        },
        b"\x01\x02\x00\x00",
    )
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(source_bytes)

    assert export._normalize_export_adapter_keys(tmp_path) == "multimodal"
    assert path.read_bytes() == source_bytes


def test_export_adapter_without_safetensors_still_exports(monkeypatch):
    uploaded: dict = {}
    bin_weights = b"legacy-adapter-weights"

    def fake_snapshot_download(*, local_dir, **kw):
        adapter = Path(local_dir) / "rl/run-x/seed0/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.bin").write_bytes(bin_weights)
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kw):
            pass

        def update_repo_settings(self, **kw):
            pass

        def repo_info(self, **kw):
            return types.SimpleNamespace(sha="parent-sha")

        def upload_folder(self, *, folder_path, **kw):
            folder = Path(folder_path)
            uploaded["files"] = sorted(path.name for path in folder.iterdir())
            uploaded["weights"] = (folder / "adapter_model.bin").read_bytes()

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    url = export_adapter(
        source_repo="org/test-runs",
        source_subfolder="rl/run-x/seed0/adapter",
        dest_repo="me/adapters",
        dest_token="hf_user",
        base_model=BASE_MODEL,
        source_token="hf_operator",
    )

    assert url == "https://huggingface.co/me/adapters"
    assert uploaded["files"] == ["adapter_config.json", "adapter_model.bin"]
    assert uploaded["weights"] == bin_weights


def test_export_adapter_rewrites_temp_merged_base_model_metadata(monkeypatch):
    """Warm-start GRPO can save PEFT/HF metadata pointing at a temporary merged SFT path.

    Export must publish the real catalog base model from the run spec so downstream Hub loaders and
    model cards do not point at a deleted `/tmp/flash_sft_merged_*` directory.
    """
    uploaded: dict = {}
    temp_base = "/tmp/flash_sft_merged_abcd1234"

    def fake_snapshot_download(*, local_dir, **kw):
        adapter = Path(local_dir) / "rl/run-x/seed0/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": temp_base,
                    "r": 32,
                    "target_modules": "all-linear",
                }
            )
            + "\n"
        )
        (adapter / "README.md").write_text(
            f"---\nbase_model:\n- {temp_base}\nlibrary_name: peft\n---\n# Adapter\n"
        )
        (adapter / "adapter_model.safetensors").write_bytes(_PLAIN_WEIGHTS)
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kw):
            pass

        def update_repo_settings(self, **kw):
            pass

        def repo_info(self, **kw):
            return types.SimpleNamespace(sha="parent-sha")

        def upload_folder(self, *, folder_path, **kw):
            folder = Path(folder_path)
            uploaded["adapter_config"] = json.loads((folder / "adapter_config.json").read_text())
            uploaded["readme"] = (folder / "README.md").read_text()

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    export_adapter(
        source_repo="org/test-runs",
        source_subfolder="rl/run-x/seed0/adapter",
        dest_repo="me/adapters",
        dest_token="hf_user",
        source_token="hf_operator",
        base_model=BASE_MODEL,
    )

    assert uploaded["adapter_config"]["base_model_name_or_path"] == BASE_MODEL
    assert uploaded["adapter_config"]["revision"] is None
    assert BASE_MODEL in uploaded["readme"]
    assert temp_base not in uploaded["readme"]


def test_export_adapter_config_validates_and_preserves_revision(tmp_path):
    from flash.serve import export

    revision = "a" * 40
    path = tmp_path / "adapter_config.json"
    path.write_text(
        json.dumps({"base_model_name_or_path": BASE_MODEL, "revision": revision}),
        encoding="utf-8",
    )

    assert export._rewrite_adapter_config_base_model(tmp_path, BASE_MODEL, revision) is False

    path.write_text(
        json.dumps({"base_model_name_or_path": "other/model", "revision": revision}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match run model"):
        export._rewrite_adapter_config_base_model(tmp_path, BASE_MODEL, revision)

    path.write_text(
        json.dumps({"base_model_name_or_path": BASE_MODEL, "revision": "b" * 40}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="revision does not match"):
        export._rewrite_adapter_config_base_model(tmp_path, BASE_MODEL, revision)


def test_export_metadata_repair_skips_non_utf8_files(tmp_path):
    from flash.serve import export

    (tmp_path / "adapter_config.json").write_bytes(b"\xff")
    (tmp_path / "README.md").write_bytes(b"\xfe")

    assert export._rewrite_adapter_config_base_model(tmp_path, BASE_MODEL) is False
    assert export._rewrite_readme_temp_base_model(tmp_path, BASE_MODEL) is False


def test_export_readme_base_model_replacement_is_literal(tmp_path):
    from flash.serve import export

    temp_base = "/tmp/flash_sft_merged_abcd1234"
    literal_base_model = r"org\1/model"
    (tmp_path / "README.md").write_text(
        f"---\nbase_model:\n- {temp_base}\nlibrary_name: peft\n---\n",
        encoding="utf-8",
    )

    assert export._rewrite_readme_temp_base_model(tmp_path, literal_base_model) is True
    assert literal_base_model in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_export_clears_stale_adapter_weights_without_touching_user_files(monkeypatch):
    """A re-export into a repo holding a previous, differently-serialized adapter must clear the stale
    adapter weights (so a leftover ``.bin`` can't be loaded next to the new ``.safetensors``) WITHOUT
    deleting the user's unrelated files. The deletion is by STATIC adapter-scoped patterns, not a
    ``existing - uploaded`` mirror, and needs no ``list_repo_files`` call (so a listing failure can't
    silently skip cleanup either)."""
    calls: dict = {}
    listed = {"called": False}

    def fake_snapshot_download(*, local_dir, **kw):
        adapter = Path(local_dir) / "rl/run-x/seed0/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(_PLAIN_WEIGHTS)
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kw):
            pass

        def list_repo_files(self, *, repo_id, repo_type):
            listed["called"] = (
                True  # must NOT be called: cleanup is pattern-based, not listing-based
            )
            return []

        def update_repo_settings(self, **kw):
            pass

        def repo_info(self, **kw):
            return types.SimpleNamespace(sha="parent-sha")

        def upload_folder(self, *, folder_path, delete_patterns, **kw):
            calls["files"] = sorted(p.name for p in Path(folder_path).iterdir())
            calls["delete_patterns"] = delete_patterns

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    export_adapter(
        source_repo="org/test-runs",
        source_subfolder="rl/run-x/seed0/adapter",
        dest_repo="me/adapters",
        dest_token="hf_user",
        base_model=BASE_MODEL,
        source_token="hf_operator",
    )
    # Static, adapter-scoped delete patterns: ``adapter_model*`` clears a stale ``.bin`` (and sharded /
    # index variants); ``adapter_config.json`` clears the old config. A hypothetical ``extra_weights.pt``
    # or any other user file matches NEITHER pattern, so it is never deleted — the data-loss the old
    # ``existing - uploaded`` mirror caused.
    assert calls["delete_patterns"] == ["adapter_model*", "adapter_config.json"]
    assert "extra_weights.pt" not in calls["delete_patterns"]
    assert not listed["called"], "stale cleanup must not depend on list_repo_files"


def test_export_public_visibility_is_deferred_until_after_upload(monkeypatch):
    """``private=False`` must not expose an empty/partial PUBLIC repo: the repo is created/ensured
    private, the adapter is uploaded, and only THEN is it flipped public — so a failed upload never
    leaves an empty public repo behind."""
    order: list = []

    def fake_snapshot_download(*, local_dir, **kw):
        adapter = Path(local_dir) / "rl/run-x/seed0/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(_PLAIN_WEIGHTS)
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, *, repo_id, repo_type, private, exist_ok):
            order.append(("create_repo", private))

        def update_repo_settings(self, *, repo_id, repo_type, private):
            order.append(("update_settings", private))

        def repo_info(self, **kw):
            return types.SimpleNamespace(sha="parent-sha")

        def upload_folder(self, **kw):
            order.append(("upload", None))

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    export_adapter(
        source_repo="org/test-runs",
        source_subfolder="rl/run-x/seed0/adapter",
        dest_repo="me/adapters",
        dest_token="hf_user",
        base_model=BASE_MODEL,
        source_token="hf_operator",
        private=False,
    )
    # Created private, uploaded, THEN made public — never public before the content lands.
    assert order == [("create_repo", True), ("upload", None), ("update_settings", False)]


def test_export_private_is_enforced_before_upload(monkeypatch):
    """``private=True`` (default) into a PRE-EXISTING public repo must lock it down BEFORE the upload:
    create_repo(exist_ok=True) won't change an existing repo's visibility, so the weights would
    otherwise commit while the repo is still public. Visibility private must be set before upload."""
    order: list = []

    def fake_snapshot_download(*, local_dir, **kw):
        adapter = Path(local_dir) / "rl/run-x/seed0/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(_PLAIN_WEIGHTS)
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, *, repo_id, repo_type, private, exist_ok):
            order.append(("create_repo", private))

        def update_repo_settings(self, *, repo_id, repo_type, private):
            order.append(("update_settings", private))

        def repo_info(self, **kw):
            return types.SimpleNamespace(sha="parent-sha")

        def upload_folder(self, **kw):
            order.append(("upload", None))

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    export_adapter(
        source_repo="org/test-runs",
        source_subfolder="rl/run-x/seed0/adapter",
        dest_repo="me/adapters",
        dest_token="hf_user",
        base_model=BASE_MODEL,
        source_token="hf_operator",
        private=True,
    )
    # Locked private BEFORE the upload (no post-upload visibility flip needed for a private export).
    assert order == [("create_repo", True), ("update_settings", True), ("upload", None)]


def test_export_adapter_falls_back_to_hf_token_env_for_source(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    seen: dict = {}

    def fake_snapshot_download(*, repo_id, repo_type, allow_patterns, local_dir, token):
        seen["token"] = token
        adapter = Path(local_dir) / "rl/run-x/seed0/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(_PLAIN_WEIGHTS)
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kw):
            pass

        def update_repo_settings(self, **kw):
            pass

        def repo_info(self, **kw):
            return types.SimpleNamespace(sha="parent-sha")

        def upload_folder(self, **kw):
            pass

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    export_adapter(
        source_repo="org/test-runs",
        source_subfolder="rl/run-x/seed0/adapter",
        dest_repo="me/adapters",
        dest_token="hf_user",
        base_model=BASE_MODEL,
    )
    assert seen["token"] == "hf_from_env"  # no source_token -> HF_TOKEN


def test_export_adapter_raises_value_error_when_source_is_empty(monkeypatch):
    def fake_snapshot_download(*, local_dir, **kw):
        # Download succeeds but the adapter subfolder has no files (nothing matched).
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    with pytest.raises(ValueError, match="no loadable LoRA adapter"):
        export_adapter(
            source_repo="org/test-runs",
            source_subfolder="rl/run-x/seed0/adapter",
            dest_repo="me/adapters",
            dest_token="hf_user",
            base_model=BASE_MODEL,
            source_token="hf_operator",
        )


def test_export_rejects_source_with_config_but_no_adapter_weight(monkeypatch):
    """A source folder with adapter_config.json but NO adapter_model* weight is not a loadable
    adapter (the non-deployable shape checkpoint-publishing filters out). Exporting it would clear
    the destination's prior weights via delete_patterns and commit a repo that can't load while
    reporting success — reject up front with a ValueError before any create/upload happens."""
    uploaded = {"called": False}

    def fake_snapshot_download(*, local_dir, **kw):
        adapter = Path(local_dir) / "rl/run-x/seed0/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")  # config only, no weight
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kw):
            uploaded["called"] = True

        def upload_folder(self, **kw):
            uploaded["called"] = True

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    with pytest.raises(ValueError, match="no loadable LoRA adapter"):
        export_adapter(
            source_repo="org/test-runs",
            source_subfolder="rl/run-x/seed0/adapter",
            dest_repo="me/adapters",
            dest_token="hf_user",
            base_model=BASE_MODEL,
            source_token="hf_operator",
        )
    assert not uploaded["called"], "must reject before touching the destination repo"


def _lora_pair_header(rank_a: int, rank_b: int, *, prefix: str, offset: int = 0) -> dict:
    """A `lora_A`/`lora_B` pair in PEFT's real 2-D layout: A is [r, in], B is [out, r]."""
    return {
        f"{prefix}.lora_A.default.weight": {
            "dtype": "F16",
            "shape": [rank_a, 64],
            "data_offsets": [offset, offset + 2],
        },
        f"{prefix}.lora_B.default.weight": {
            "dtype": "F16",
            "shape": [64, rank_b],
            "data_offsets": [offset + 2, offset + 4],
        },
    }


def test_export_refuses_adapter_whose_tensors_contradict_its_declared_rank(tmp_path):
    """A rank axis no expert count can explain: `r: 32` beside a 100-long axis.

    Export used to check only that the weight files were present and their key namespace readable,
    so an unloadable adapter published a Hub URL while the identical artifact failed to deploy. A
    successful export then read as evidence the artifact worked.
    """
    from flash.serve import export

    header = _lora_pair_header(32, 32, prefix="base_model.model.model.layers.0.self_attn.q_proj")
    header.update(
        _lora_pair_header(100, 100, prefix="base_model.model.model.layers.0.mlp.up_proj", offset=4)
    )
    (tmp_path / "adapter_model.safetensors").write_bytes(_safetensors_bytes(header, b"\x01" * 8))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 32, "lora_alpha": 64}))

    with pytest.raises(ValueError, match=r"declares r=32.*not a multiple of it"):
        export._normalize_export_adapter_keys(tmp_path)


def test_export_accepts_fused_moe_experts_stacked_on_the_rank_axis(tmp_path):
    """The exact LMR-030 shapes, which are LEGITIMATE: `lora.ParamWrapper.update_layer` builds
    `nn.Linear(in_features, r * num_experts)`, so an r=32 adapter over 256 experts really does
    carry an 8192-long rank axis. Demanding equality here would refuse adapters that load, which
    is a worse failure than the one being fixed -- export is the user's escape hatch."""
    from flash.serve import export

    header = _lora_pair_header(32, 32, prefix="base_model.model.model.layers.0.self_attn.q_proj")
    header.update(
        _lora_pair_header(
            8192, 8192, prefix="base_model.model.model.layers.0.mlp.experts", offset=4
        )
    )
    (tmp_path / "adapter_model.safetensors").write_bytes(_safetensors_bytes(header, b"\x01" * 8))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 32, "lora_alpha": 64}))

    assert export._normalize_export_adapter_keys(tmp_path) == "text_only"


def test_export_rank_mismatch_reaches_the_caller_as_itself(tmp_path):
    """Not reported as an unnormalizable key namespace: that is a different defect and remedy."""
    from flash.serve import export

    header = _lora_pair_header(24, 24, prefix="base_model.model.model.layers.0.mlp.up_proj")
    (tmp_path / "adapter_model.safetensors").write_bytes(_safetensors_bytes(header, b"\x01" * 4))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16}))

    with pytest.raises(ValueError, match="not a multiple of it") as excinfo:
        export._normalize_export_adapter_keys(tmp_path)
    assert "could not normalize exported adapter keys" not in str(excinfo.value)


def test_export_refuses_rank_mismatch_before_touching_the_destination_repo(monkeypatch):
    """The refusal has to land before create_repo/upload_folder: `delete_patterns` clears the
    destination's prior adapter, so failing mid-upload would destroy a good published adapter and
    replace it with an unloadable one."""
    touched = {"called": False}

    def fake_snapshot_download(*, local_dir, **kw):
        adapter = Path(local_dir) / "sft/run-x/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        header = _lora_pair_header(48, 48, prefix="base_model.model.model.layers.0.mlp.gate_proj")
        (adapter / "adapter_model.safetensors").write_bytes(_safetensors_bytes(header, b"\x01" * 4))
        (adapter / "adapter_config.json").write_text(json.dumps({"r": 32}))
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kw):
            touched["called"] = True

        def upload_folder(self, **kw):
            touched["called"] = True

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    with pytest.raises(ValueError, match="not a multiple of it"):
        export_adapter(
            source_repo="org/test-runs",
            source_subfolder="sft/run-x/adapter",
            dest_repo="me/adapters",
            dest_token="hf_user",
            base_model=BASE_MODEL,
            source_token="hf_operator",
        )
    assert not touched["called"], "must refuse before touching the destination repo"


def test_export_accepts_an_adapter_whose_tensors_match_its_declared_rank(tmp_path):
    """The check must not cost a good export: matching shapes pass straight through."""
    from flash.serve import export

    header = _lora_pair_header(16, 16, prefix="base_model.model.model.layers.0.mlp.up_proj")
    (tmp_path / "adapter_model.safetensors").write_bytes(_safetensors_bytes(header, b"\x01" * 4))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))

    assert export._normalize_export_adapter_keys(tmp_path) == "text_only"


def test_export_reads_rank_pattern_overrides_rather_than_refusing_them(tmp_path):
    """PEFT records per-module ranks in `rank_pattern`, and `rank_from_adapter_config` returns the
    max across both places. A module legitimately trained at the higher rank must still export."""
    from flash.serve import export

    header = _lora_pair_header(80, 80, prefix="base_model.model.model.layers.0.mlp.up_proj")
    (tmp_path / "adapter_model.safetensors").write_bytes(_safetensors_bytes(header, b"\x01" * 4))
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"r": 12, "rank_pattern": {"layers.0.mlp.up_proj": 80}})
    )

    assert export._normalize_export_adapter_keys(tmp_path) == "text_only"


def test_export_ignores_shapes_that_cannot_report_a_rank(tmp_path):
    """`modules_to_save` copies, biases and stacked 3-D fused layouts are not [r, in] / [out, r].
    None of them can answer the question, and reading a rank out of them anyway would refuse
    adapters that are fine -- so they are skipped rather than guessed at."""
    from flash.serve import export

    header = {
        "base_model.model.model.layers.0.mlp.up_proj.lora_A.default.weight": {
            "dtype": "F16",
            "shape": [16, 64],
            "data_offsets": [0, 2],
        },
        # 3-D stacked expert weights: no [r, in] / [out, r] reading exists
        "base_model.model.model.layers.0.mlp.experts.lora_A.default.weight": {
            "dtype": "F16",
            "shape": [8, 16, 64],
            "data_offsets": [2, 4],
        },
        # a saved full module copy, which carries no rank at all
        "base_model.model.model.embed_tokens.modules_to_save.default.weight": {
            "dtype": "F16",
            "shape": [1000, 64],
            "data_offsets": [4, 6],
        },
        "base_model.model.model.layers.0.mlp.up_proj.lora_B.default.bias": {
            "dtype": "F16",
            "shape": [64],
            "data_offsets": [6, 8],
        },
    }
    (tmp_path / "adapter_model.safetensors").write_bytes(_safetensors_bytes(header, b"\x01" * 8))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16}))

    assert export._normalize_export_adapter_keys(tmp_path) == "text_only"


def test_export_without_a_readable_declared_rank_still_exports(tmp_path):
    """Every pre-existing fixture and any PEFT save without `r` has no rank to disagree with.
    The check needs both halves, so a config that declares none leaves export exactly as it was."""
    from flash.serve import export

    header = _lora_pair_header(16, 16, prefix="base_model.model.model.layers.0.mlp.up_proj")
    (tmp_path / "adapter_model.safetensors").write_bytes(_safetensors_bytes(header, b"\x01" * 4))
    (tmp_path / "adapter_config.json").write_text("{}")

    assert export._normalize_export_adapter_keys(tmp_path) == "text_only"


def test_export_checks_ranks_across_every_shard_not_just_the_first(tmp_path):
    """Sharding splits one key namespace over several files, so a bad expert tensor can sit in a
    shard the first-file-only reading never opens."""
    from flash.serve import export

    good = _lora_pair_header(32, 32, prefix="base_model.model.model.layers.0.self_attn.q_proj")
    bad = _lora_pair_header(100, 100, prefix="base_model.model.model.layers.1.mlp.up_proj")
    (tmp_path / "adapter_model-00001-of-00002.safetensors").write_bytes(
        _safetensors_bytes(good, b"\x01" * 4)
    )
    (tmp_path / "adapter_model-00002-of-00002.safetensors").write_bytes(
        _safetensors_bytes(bad, b"\x01" * 4)
    )
    (tmp_path / "adapter_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    **dict.fromkeys(good, "adapter_model-00001-of-00002.safetensors"),
                    **dict.fromkeys(bad, "adapter_model-00002-of-00002.safetensors"),
                }
            }
        )
    )
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 32}))

    with pytest.raises(ValueError, match="not a multiple of it"):
        export._normalize_export_adapter_keys(tmp_path)


def test_export_adapter_wraps_download_failure_in_serving_error(monkeypatch):
    from flash.serve.deploy import ServingError

    def fake_snapshot_download(**kw):
        raise RuntimeError("401 Unauthorized")

    class FakeHfApi:
        def __init__(self, token=None):
            pass

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    with pytest.raises(ServingError, match="could not download adapter"):
        export_adapter(
            source_repo="org/test-runs",
            source_subfolder="rl/run-x/seed0/adapter",
            dest_repo="me/adapters",
            dest_token="hf_user",
            base_model=BASE_MODEL,
            source_token="hf_operator",
        )


def test_export_adapter_wraps_hub_download_oserror_in_serving_error(monkeypatch):
    from flash.serve.deploy import ServingError

    message = "You don't have the rights to download this repository"

    def fake_snapshot_download(**kw):
        raise _FakeHfHubHTTPError(message)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    with pytest.raises(ServingError) as exc_info:
        export_adapter(
            source_repo="org/test-runs",
            source_subfolder="rl/run-x/seed0/adapter",
            dest_repo="me/adapters",
            dest_token="hf_user",
            base_model=BASE_MODEL,
            source_token="hf_operator",
        )

    assert message in str(exc_info.value)


def test_export_adapter_propagates_local_download_oserror(monkeypatch):
    from flash.serve.deploy import ServingError

    def fake_snapshot_download(**kw):
        raise PermissionError("local export directory is not writable")

    class FakeHfApi:
        def __init__(self, token=None):
            pass

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    with pytest.raises(PermissionError) as exc_info:
        export_adapter(
            source_repo="org/test-runs",
            source_subfolder="rl/run-x/seed0/adapter",
            dest_repo="me/adapters",
            dest_token="hf_user",
            base_model=BASE_MODEL,
            source_token="hf_operator",
        )

    assert not isinstance(exc_info.value, ServingError)


def test_export_adapter_wraps_hub_create_repo_oserror_in_serving_error(monkeypatch):
    from flash.serve.deploy import ServingError

    message = "You don't have the rights to create a model under the namespace me"

    def fake_snapshot_download(*, local_dir, **kw):
        adapter = Path(local_dir) / "rl/run-x/seed0/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(_PLAIN_WEIGHTS)
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kw):
            raise _FakeHfHubHTTPError(message)

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    with pytest.raises(ServingError) as exc_info:
        export_adapter(
            source_repo="org/test-runs",
            source_subfolder="rl/run-x/seed0/adapter",
            dest_repo="me/adapters",
            dest_token="hf_user",
            base_model=BASE_MODEL,
            source_token="hf_operator",
        )

    assert message in str(exc_info.value)


def test_resolve_hf_token_priority_explicit_then_env_then_dotenv(tmp_path, monkeypatch):
    from flash.client.runtime_secrets import resolve_hf_token

    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)

    # Nothing set anywhere -> None.
    assert resolve_hf_token(None) is None

    # The huggingface_hub aliases are deliberately NOT accepted: only HF_TOKEN is.
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_alias")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_alias2")
    assert resolve_hf_token(None) is None
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    # A local .env supplies the token when the env doesn't.
    (tmp_path / ".env").write_text('HF_TOKEN="hf_from_dotenv"\n')
    assert resolve_hf_token(None) == "hf_from_dotenv"

    # The process environment wins over .env.
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert resolve_hf_token(None) == "hf_from_env"

    # An explicit value (the --api-key flag) wins over everything.
    assert resolve_hf_token("hf_explicit") == "hf_explicit"


def test_hf_api_missing_extra_raises_runtime_error_not_serving_error(monkeypatch):
    # A missing huggingface_hub extra is an internal misconfiguration (-> 500), NOT an upstream
    # gateway/transport failure (ServingError -> 502). _hf_api must raise a plain RuntimeError so the
    # route lets it surface as a 500 rather than a misleading 502.
    import builtins

    from flash.serve import export
    from flash.serve.deploy import ServingError

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ModuleNotFoundError("No module named 'huggingface_hub'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as ei:
        export._hf_api()
    assert not isinstance(ei.value, ServingError)  # NOT the 502 path
    assert "huggingface_hub" in str(ei.value)


def test_export_missing_operator_token_raises_runtime_error_not_serving_error(monkeypatch):
    # No operator read token (neither source_token nor HF_TOKEN) is an internal server
    # misconfiguration (-> 500), NOT an upstream auth failure (ServingError -> 502). export_adapter
    # must raise a plain RuntimeError BEFORE snapshot_download, rather than passing token=None and
    # wrapping the resulting auth error as ServingError.
    from flash.serve import export
    from flash.serve.deploy import ServingError

    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError) as ei:
        export.export_adapter(
            source_repo="op/ds",
            source_subfolder="runs/r/adapter",
            dest_repo="me/out",
            dest_token="hf_dest",
            base_model=BASE_MODEL,
            source_token=None,
        )
    assert not isinstance(ei.value, ServingError)  # NOT the 502 path
    assert "HF_TOKEN" in str(ei.value)
