"""fail-closed build-time repair for the vllm 0.23.0 fp8 moe lora path."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docker" / "patch_vllm_moe_lora.py"
FIXTURES = ROOT / "tests" / "fixtures" / "vllm_023_moe_lora"


def _load_repair() -> Any:
    spec = importlib.util.spec_from_file_location("patch_vllm_moe_lora", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


repair = _load_repair()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _install_tree(tmp_path: Path, *, version: str = "0.23.0") -> Path:
    root = tmp_path / "site-packages"
    for target in repair.TARGETS:
        destination = root / target.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        fixture = FIXTURES / f"{Path(target.relative_path).name}.gz"
        destination.write_bytes(gzip.decompress(fixture.read_bytes()))
    metadata = root / f"vllm-{version}.dist-info" / "METADATA"
    metadata.parent.mkdir()
    metadata.write_text(f"Metadata-Version: 2.4\nName: vllm\nVersion: {version}\n")
    return root


def _distribution(root: Path, version: str = "0.23.0") -> SimpleNamespace:
    return SimpleNamespace(version=version, locate_file=lambda relative: root / relative)


def _target_bytes(root: Path) -> dict[str, bytes]:
    return {
        target.relative_path: (root / target.relative_path).read_bytes()
        for target in repair.TARGETS
    }


def _tree_entries(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*")}


def _assert_failure_without_writes(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    version: str = "0.23.0",
    match: str,
) -> None:
    before_bytes = _target_bytes(root)
    before_entries = _tree_entries(root)
    monkeypatch.setattr(
        repair.importlib.metadata,
        "distribution",
        lambda _name: _distribution(root, version),
    )
    with pytest.raises(repair.RepairError, match=match):
        repair.repair()
    assert _target_bytes(root) == before_bytes
    assert _tree_entries(root) == before_entries


def test_repair_success_postconditions_and_exact_idempotence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _install_tree(tmp_path)
    monkeypatch.setattr(
        repair.importlib.metadata,
        "distribution",
        lambda _name: _distribution(root),
    )
    replacements: list[tuple[Path, Path]] = []
    real_replace = repair.os.replace

    def record_replace(source: Path, destination: Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(repair.os, "replace", record_replace)
    result = repair.repair()
    assert result.startswith("applied and verified")
    assert len(replacements) == len(repair.TARGETS)

    patched = _target_bytes(root)
    for target in repair.TARGETS:
        assert _sha256(patched[target.relative_path]) == target.post_hash
    repair._verify_semantics(patched)

    replacements.clear()
    before = _target_bytes(root)
    assert repair.repair().startswith("verified exact")
    assert replacements == []
    assert _target_bytes(root) == before


def test_semantic_validation_failure_does_not_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _install_tree(tmp_path)
    before_bytes = _target_bytes(root)
    before_entries = _tree_entries(root)
    monkeypatch.setattr(
        repair.importlib.metadata,
        "distribution",
        lambda _name: _distribution(root),
    )
    monkeypatch.setattr(
        repair,
        "_verify_semantics",
        lambda _sources: (_ for _ in ()).throw(repair.RepairError("semantic sabotage")),
    )
    with pytest.raises(repair.RepairError, match="semantic sabotage"):
        repair.repair()
    assert _target_bytes(root) == before_bytes
    assert _tree_entries(root) == before_entries


def test_same_length_byte_drift_fails_before_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _install_tree(tmp_path)
    path = root / repair.TARGETS[0].relative_path
    source = path.read_bytes()
    changed = source.replace(b"Carries", b"carries", 1)
    assert len(changed) == len(source)
    assert changed != source
    path.write_bytes(changed)
    _assert_failure_without_writes(monkeypatch, root, match="unknown")


def test_duplicate_anchor_fails_before_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _install_tree(tmp_path)
    target = repair.TARGETS[0]
    path = root / target.relative_path
    anchor = b"    local_token_lora_mapping: torch.Tensor | None = None\n"
    path.write_bytes(path.read_bytes() + anchor)
    duplicate_hash = _sha256(path.read_bytes())
    monkeypatch.setattr(
        repair,
        "TARGETS",
        (
            repair.Target(target.relative_path, duplicate_hash, target.post_hash),
            *repair.TARGETS[1:],
        ),
    )
    _assert_failure_without_writes(monkeypatch, root, match="expected one anchor, found 2")


def test_mixed_state_fails_before_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = _install_tree(tmp_path)
    pristine = _target_bytes(root)
    transformed = repair._transform_sources(pristine)
    first = repair.TARGETS[0].relative_path
    (root / first).write_bytes(transformed[first])
    _assert_failure_without_writes(monkeypatch, root, match="mixed")


def test_wrong_version_fails_before_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = _install_tree(tmp_path, version="0.23.1")
    _assert_failure_without_writes(
        monkeypatch,
        root,
        version="0.23.1",
        match=r"expected vllm 0\.23\.0, found 0\.23\.1",
    )


def test_semantic_verifier_rejects_raw_cast_sabotage(tmp_path: Path) -> None:
    root = _install_tree(tmp_path)
    transformed = repair._transform_sources(_target_bytes(root))
    path = repair.TARGETS[1].relative_path
    source = transformed[path].decode()
    source = source.replace(
        "            lora_x = lora_unquantized_hidden_states\n",
        "            lora_x = lora_unquantized_hidden_states.to(hidden_states.dtype)\n",
        1,
    )
    transformed[path] = source.encode()
    with pytest.raises(repair.RepairError, match="must not use a raw cast"):
        repair._verify_semantics(transformed)


def test_semantic_verifier_rejects_all_to_all_local_stash_fallback(tmp_path: Path) -> None:
    root = _install_tree(tmp_path)
    transformed = repair._transform_sources(_target_bytes(root))
    path = repair.TARGETS[1].relative_path
    source = transformed[path].decode()
    guard = "            and not self.moe_config.moe_parallel_config.use_all2all_kernels\n"
    assert source.count(guard) == 1
    transformed[path] = source.replace(guard, "", 1).encode()
    with pytest.raises(repair.RepairError, match="disabled for all-to-all"):
        repair._verify_semantics(transformed)


def test_semantic_verifier_rejects_unweighted_router_input_stash(tmp_path: Path) -> None:
    root = _install_tree(tmp_path)
    transformed = repair._transform_sources(_target_bytes(root))
    path = repair.TARGETS[2].relative_path
    source = transformed[path].decode()
    old = """        if lora_ctx is not None and apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                \"apply_router_weight_on_input is only implemented for topk=1\"
            )
            lora_hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
"""
    replacement = ""
    assert source.count(old) == 1
    transformed[path] = source.replace(old, replacement, 1).encode()
    with pytest.raises(repair.RepairError, match="router-weighted unquantized"):
        repair._verify_semantics(transformed)


def test_semantic_verifier_rejects_local_stash_after_prepare(tmp_path: Path) -> None:
    root = _install_tree(tmp_path)
    transformed = repair._transform_sources(_target_bytes(root))
    path = repair.TARGETS[2].relative_path
    source = transformed[path].decode()
    initial = "        lora_hidden_states = hidden_states\n"
    after_prepare = """            apply_router_weight_on_input,
        )

"""
    assert source.count(initial) == 1
    assert source.count(after_prepare) == 1
    source = source.replace(initial, "        pass\n", 1)
    source = source.replace(after_prepare, after_prepare + initial, 1)
    transformed[path] = source.encode()
    with pytest.raises(repair.RepairError, match="stash must precede prepare"):
        repair._verify_semantics(transformed)


def test_semantic_verifier_rejects_missing_requantization(tmp_path: Path) -> None:
    root = _install_tree(tmp_path)
    transformed = repair._transform_sources(_target_bytes(root))
    path = repair.TARGETS[1].relative_path
    source = transformed[path].decode()
    old = """            hidden_states, a1q_scale = moe_kernel_quantize_input(
                hidden_states,
                self.a1_scale,
                self.quant_dtype,
                self.per_act_token_quant,
                self.block_shape,
                quantization_emulation=self.quantization_emulation,
            )
"""
    replacement = "            hidden_states, a1q_scale = hidden_states, a1q_scale\n"
    assert source.count(old) == 1
    transformed[path] = source.replace(old, replacement).encode()
    with pytest.raises(repair.RepairError, match="requantization is missing"):
        repair._verify_semantics(transformed)


def test_semantic_verifier_rejects_clear_moved_outside_finally(tmp_path: Path) -> None:
    root = _install_tree(tmp_path)
    transformed = repair._transform_sources(_target_bytes(root))
    path = repair.TARGETS[2].relative_path
    source = transformed[path].decode()
    old = """        finally:
            if lora_ctx is not None:
                lora_ctx.original_hidden_states = None

        return self._finalize(
"""
    replacement = """        finally:
            pass
        if lora_ctx is not None:
            lora_ctx.original_hidden_states = None

        return self._finalize(
"""
    assert source.count(old) == 1
    transformed[path] = source.replace(old, replacement).encode()
    with pytest.raises(repair.RepairError, match="clear must be in the fused-experts finally"):
        repair._verify_semantics(transformed)


def test_subprocess_repair_does_not_import_runtime_packages(tmp_path: Path) -> None:
    root = _install_tree(tmp_path)
    (root / "vllm" / "__init__.py").write_text('raise AssertionError("vllm imported")\n')
    (root / "torch.py").write_text('raise AssertionError("torch imported")\n')
    (root / "flash.py").write_text('raise AssertionError("flash imported")\n')
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("applied and verified")


def test_subprocess_failure_is_nonzero(tmp_path: Path) -> None:
    root = _install_tree(tmp_path, version="0.23.1")
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "expected vllm 0.23.0, found 0.23.1" in result.stderr


def test_verifier_rejects_a_zero_exit_noop(tmp_path: Path) -> None:
    root = _install_tree(tmp_path)
    noop = subprocess.run([sys.executable, "-c", "pass"], check=False)
    assert noop.returncode == 0

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--verify"],
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "required moe lora backport is absent" in result.stderr
    for target in repair.TARGETS:
        assert _sha256((root / target.relative_path).read_bytes()) == target.pre_hash
