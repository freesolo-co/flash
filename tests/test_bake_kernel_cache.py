import importlib.util
import json
from pathlib import Path

import pytest

from flash.providers.modal.api import call as call_modal


def bake_module():
    spec = importlib.util.spec_from_file_location(
        "bake", Path(__file__).parents[1] / "docker/bake_kernel_cache.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_baked_arch_has_a_modal_gpu():
    from flash.providers._lifecycle.net.worker import BAKED_PER_SM_ARCHES
    from flash.providers.core.base import GPU_CLASSES

    mapping = bake_module().GPU_BY_SM
    assert set(mapping) == BAKED_PER_SM_ARCHES
    for sm, request in mapping.items():
        assert any(
            g.sm == sm and g.modal_name.rstrip("!") == request.rstrip("!") for g in GPU_CLASSES
        )


def test_missing_bake_artifact_fails_verification(tmp_path):
    assert bake_module()._verify(str(tmp_path), "sm90") != 0


@pytest.mark.parametrize(
    "artifact", ["triton/kernel.cubin", "triton/kernel.ptx", "inductor/kernel.so"]
)
def test_verify_accepts_real_raw_artifacts(tmp_path, artifact):
    path = tmp_path / artifact
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"compiled")
    (tmp_path / "mega_cache.json").write_text('{"sm": "sm100"}')
    (tmp_path / "cache_validation.json").write_text(
        '{"first_exit_code": 0, "reload_exit_code": 0, "exported_cache_replayed": true}'
    )
    (tmp_path / "cache_reload.json").write_text(
        json.dumps(
            {
                "cached_groups_compilation_forbidden": True,
                "cached_groups": ["flash_attn", "fla_gdn"],
                "startup_jit_groups": [],
                "compile_attempts": 0,
                "passed": True,
            }
        )
    )
    assert bake_module()._verify(str(tmp_path), "sm100") == 0
    assert bake_module()._verify(str(tmp_path), "sm90") == 1


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "recompiled",
        "unguarded",
        "failed",
        "reload_exit",
        "not_exported",
        "scope",
        "startup",
    ],
)
def test_verification_requires_a_successful_compilation_free_reload(tmp_path, failure):
    (tmp_path / "triton").mkdir()
    (tmp_path / "triton/kernel.cubin").write_bytes(b"compiled")
    (tmp_path / "mega_cache.json").write_text('{"sm": "sm100"}')
    validation = {
        "first_exit_code": 0,
        "reload_exit_code": int(failure == "reload_exit"),
        "exported_cache_replayed": failure != "not_exported",
    }
    (tmp_path / "cache_validation.json").write_text(json.dumps(validation))
    if failure != "missing":
        (tmp_path / "cache_reload.json").write_text(
            json.dumps(
                {
                    "cached_groups_compilation_forbidden": failure != "unguarded",
                    "cached_groups": [] if failure == "scope" else ["flash_attn", "fla_gdn"],
                    "startup_jit_groups": ["flashinfer_gdn"] if failure == "startup" else [],
                    "compile_attempts": int(failure == "recompiled"),
                    "passed": failure != "failed",
                }
            )
        )
    assert bake_module()._verify(str(tmp_path), "sm100") == 1


def test_verify_rejects_metadata_and_lock_files_without_kernels(tmp_path):
    (tmp_path / "triton").mkdir()
    (tmp_path / "triton" / "kernel.lock").write_text("lock")
    (tmp_path / "mega_cache.json").write_text('{"sm": "sm100"}')
    assert bake_module()._verify(str(tmp_path), "sm100") == 1


def test_verify_rejects_legacy_blob_without_raw_kernels(tmp_path):
    (tmp_path / "mega_cache.bin").write_bytes(b"legacy-blob")
    (tmp_path / "mega_cache.json").write_text('{"sm": "sm100"}')
    assert bake_module()._verify(str(tmp_path), "sm100") == 1


@pytest.mark.parametrize("failure", [None, "status", "download", "verify", "exec"])
def test_direct_modal_transfer_checks_completion_and_always_terminates(
    monkeypatch, tmp_path, failure
):
    import asyncio
    import io
    import sys
    import tarfile
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from flash.providers.modal import api

    bake = bake_module()
    process = SimpleNamespace(wait=SimpleNamespace(aio=AsyncMock(return_value=0)))
    status = "rc=0\narch=10.0\n" if failure != "status" else "rc=1\narch=10.0\n"

    async def download(remote, local):
        if failure == "download":
            raise OSError("download interrupted")
        if remote.endswith(".tar.gz"):
            with tarfile.open(local, "w:gz") as archive:
                member = tarfile.TarInfo("bake-warmup.log")
                member.size = len(b"stale log")
                archive.addfile(member, io.BytesIO(b"stale log"))
        else:
            local.write_bytes(b"current bake")

    sandbox = SimpleNamespace(
        exec=SimpleNamespace(aio=AsyncMock(return_value=process)),
        filesystem=SimpleNamespace(
            read_text=SimpleNamespace(aio=AsyncMock(return_value=status)),
            copy_to_local=SimpleNamespace(aio=AsyncMock(side_effect=download)),
        ),
        terminate=SimpleNamespace(aio=AsyncMock()),
    )
    if failure == "exec":
        sandbox.exec.aio.side_effect = RuntimeError("exec unavailable")
    image = Mock()
    image.add_local_dir.return_value = image
    image.add_local_file.return_value = image
    secret = Mock()
    create = AsyncMock(return_value=sandbox)
    modal = SimpleNamespace(
        App=SimpleNamespace(lookup=SimpleNamespace(aio=AsyncMock())),
        Image=SimpleNamespace(from_registry=Mock(return_value=image)),
        Sandbox=SimpleNamespace(create=SimpleNamespace(aio=create)),
        Secret=SimpleNamespace(from_dict=secret),
    )
    monkeypatch.setattr(api, "call", lambda fn, **kw: asyncio.run(fn(modal, None)))
    monkeypatch.setattr(bake, "_verify", lambda *a: int(failure == "verify"))
    monkeypatch.setenv("HF_TOKEN", "model-read-token")
    out = tmp_path / "published"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bake",
            "--arch",
            "10.0",
            "--sm",
            "sm100",
            "--image",
            "test/image",
            "--out",
            str(out),
        ],
    )
    if failure in {"status", "download", "exec"}:
        with pytest.raises((api.ModalApiError, OSError, RuntimeError)):
            bake.main()
    else:
        assert bake.main() == int(failure == "verify")
    assert out.exists() == (failure is None)
    if failure is None:
        assert (out / "warmup.log").read_bytes() == b"current bake"
        assert (out / "bake-warmup.log").read_bytes() == b"current bake"
    if failure in {"status", "exec"}:
        sandbox.filesystem.copy_to_local.aio.assert_not_awaited()
    sandbox.terminate.aio.assert_awaited_once_with(wait=True)
    assert create.call_args.args == ("sleep", "infinity")
    assert secret.call_args.args == ({"BAKE_ARCH": "10.0", "HF_TOKEN": "model-read-token"},)
    assert image.add_local_dir.call_args.args[1] == "/runcode/code/flash"


@pytest.mark.parametrize("returncode", [1, 137])
def test_failed_sandbox_preserves_exit_diagnostic_and_cleans_up(monkeypatch, returncode):
    import sys
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from flash.providers.modal import api

    bake = bake_module()
    monkeypatch.setenv("HF_TOKEN", "test")
    monkeypatch.setattr(api, "credentials", lambda: ("test-id", "test-secret"))
    monkeypatch.setattr(api, "call", call_modal)
    monkeypatch.setattr(
        sys,
        "argv",
        ["bake", "--arch", "9.0", "--sm", "sm90", "--image", "test/image"],
    )
    sandbox = SimpleNamespace(
        object_id="sb-test-h100",
        exec=SimpleNamespace(
            aio=AsyncMock(
                return_value=SimpleNamespace(
                    wait=SimpleNamespace(aio=AsyncMock(return_value=returncode)),
                )
            )
        ),
        terminate=SimpleNamespace(aio=AsyncMock()),
    )
    create = AsyncMock(return_value=sandbox)
    client = SimpleNamespace(__aexit__=AsyncMock())
    image = Mock()
    image.add_local_dir.return_value = image
    image.add_local_file.return_value = image
    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            Client=SimpleNamespace(
                from_credentials=SimpleNamespace(aio=AsyncMock(return_value=client))
            ),
            App=SimpleNamespace(lookup=SimpleNamespace(aio=AsyncMock(return_value="bake-app"))),
            Image=SimpleNamespace(from_registry=Mock(return_value=image)),
            Sandbox=SimpleNamespace(create=SimpleNamespace(aio=create)),
            Secret=SimpleNamespace(from_dict=Mock(return_value="bake-secret")),
        ),
    )

    # Exercise the real provider wrapper: an ordinary RuntimeError here used to
    # become a generic "operation failed or timed out", hiding the exit code.
    with pytest.raises(api.ModalApiError, match=rf"code {returncode}; .*sb-test-h100"):
        bake.main()

    assert create.call_args.kwargs["memory"] == (32768, 131072)
    sandbox.terminate.aio.assert_awaited_once_with(wait=True)
    client.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.parametrize("name", ["../escape", "/absolute"])
def test_cache_archive_rejects_escaping_paths(tmp_path, name):
    import io
    import tarfile

    path = tmp_path / "cache.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo(name)
        entry.size = 1
        archive.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="invalid kernel cache"):
        bake_module()._extract_cache_archive(path, tmp_path / "out")


def test_cache_archive_preserves_hidden_backend_directories(tmp_path):
    import io
    import tarfile

    path = tmp_path / "cache.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("flashinfer/.cache/kernel.so")
        entry.size = 8
        archive.addfile(entry, io.BytesIO(b"compiled"))
    bake_module()._extract_cache_archive(path, tmp_path / "out")
    assert (tmp_path / "out/flashinfer/.cache/kernel.so").read_bytes() == b"compiled"
