from flash.engine.worker.io import hf


def test_hf_cache_bytes_counts_blobs_and_reports_unmeasurable_as_none(tmp_path, monkeypatch):
    import huggingface_hub.constants as hconst

    monkeypatch.setattr(hconst, "HF_HUB_CACHE", str(tmp_path))
    assert hf._hf_cache_bytes("org/model") is None
    repo = tmp_path / "models--org--model"
    repo.mkdir(parents=True)
    assert hf._hf_cache_bytes("org/model") == 0
    blobs = repo / "blobs"
    blobs.mkdir()
    (blobs / "complete").write_bytes(b"x" * 100)
    (blobs / "partial.incomplete").write_bytes(b"y" * 50)
    assert hf._hf_cache_bytes("org/model") == 150
