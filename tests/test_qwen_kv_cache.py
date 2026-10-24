from flash.serve.runtime.kv_cache import qwen_cache_args


def test_qwen_native_cache_policy_is_bounded_and_detached():
    first = qwen_cache_args("Qwen/Qwen3.5-9B")
    config = first["kv_transfer_config"]
    assert config["kv_connector"] == "OffloadingConnector"
    assert config["kv_role"] == "kv_both"
    assert config["kv_connector_extra_config"] == {
        "cpu_bytes_to_use": 1024**3,
        "eviction_policy": "arc",
    }
    config["kv_connector_extra_config"]["cpu_bytes_to_use"] = 1
    assert (
        qwen_cache_args("Freesolo-Co/Qwen3.5-9B-FP8")["kv_transfer_config"][
            "kv_connector_extra_config"
        ]["cpu_bytes_to_use"]
        == 1024**3
    )
    assert qwen_cache_args("other/model") == {}
