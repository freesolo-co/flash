"""Adapter downloads stay confined to one immutable checkpoint, including retries."""

from types import SimpleNamespace

import httpx
import pytest

from flash.artifacts.api import ArtifactFile
from flash.serving.src.engine.hub import download_adapter, download_retry_delay


def test_download_scopes_tree_and_files_to_checkpoint_sha(monkeypatch, tmp_path):
    calls = []
    prefix = "sft/run/checkpoints/step-1/adapter"
    files = [prefix + "/adapter_config.json", prefix + "/shards/adapter_model.safetensors"]

    def tree(self, **kwargs):
        calls.append(("tree", kwargs))
        yield SimpleNamespace(path=prefix + "/shards")
        for filename in files:
            yield ArtifactFile(path=filename, size=7, blob_id="c" * 40, modified=1)

    def download(**kwargs):
        calls.append(("download", kwargs))
        path = tmp_path / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
        return str(path)

    def no_snapshot(**kwargs):
        raise AssertionError("must not enumerate the whole training repository")

    monkeypatch.setattr("flash.artifacts.api.ArtifactStore.list_repo_tree", tree)
    monkeypatch.setattr("flash.artifacts.api.download_file", download)
    monkeypatch.setattr("flash.artifacts.api.download_snapshot", no_snapshot)
    root = download_adapter(
        repo_id="org/data",
        repo_type="dataset",
        revision="a" * 40,
        token="token",
        subfolder=prefix,
        local_dir=str(tmp_path),
    )
    assert root == str(tmp_path)
    assert calls[0][1]["path_in_repo"] == prefix
    assert calls[0][1]["recursive"] is True
    assert [c["filename"] for kind, c in calls if kind == "download"] == files
    assert all(c["revision"] == "a" * 40 and c["token"] is None for _, c in calls)
    assert all((tmp_path / name).read_bytes() == b"weights" for name in files)


@pytest.mark.parametrize("path", ["other/adapter_model.safetensors", "../escape", "/absolute"])
def test_unexpected_tree_path_never_downloads(monkeypatch, tmp_path, path):
    monkeypatch.setattr(
        "flash.artifacts.api.ArtifactStore.list_repo_tree",
        lambda *a, **k: [ArtifactFile(path=path, size=7, blob_id="c" * 40, modified=1)],
    )

    def no_download(**kwargs):
        raise AssertionError("unsafe path must fail before download")

    monkeypatch.setattr("flash.artifacts.api.download_file", no_download)
    with pytest.raises(ValueError, match="outside the adapter directory"):
        download_adapter(
            repo_id="org/data",
            repo_type="dataset",
            revision="a" * 40,
            token=None,
            subfolder="checkpoints/one",
            local_dir=str(tmp_path),
        )


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after": "118"}, 119),
        ({"ratelimit": '"api";r=0;t=118'}, 119),
        ({"retry-after": "NaN"}, 31),
        ({"retry-after": "-1"}, 31),
        ({"retry-after": "999999"}, 300),
        ({}, 31),
    ],
)
def test_hub_rate_limit_wait_uses_reset_and_stays_bounded(headers, expected):
    error = SimpleNamespace(response=httpx.Response(429, headers=headers))
    assert download_retry_delay(error, 0) == expected


def test_non_throttling_failure_keeps_short_backoff():
    assert download_retry_delay(RuntimeError("network"), 1) == 1


def test_root_adapter_keeps_snapshot_download(monkeypatch, tmp_path):
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        return kwargs["local_dir"]

    monkeypatch.setattr("flash.artifacts.api.download_snapshot", snapshot)
    assert download_adapter(
        repo_id="org/adapter",
        repo_type="model",
        revision="a" * 40,
        local_dir=str(tmp_path),
        token=None,
        subfolder=None,
    ) == str(tmp_path)
    assert calls == [
        {
            "repo_id": "org/adapter",
            "repo_type": "model",
            "revision": "a" * 40,
            "local_dir": str(tmp_path),
            "token": None,
        }
    ]
