"""Check pairing and arithmetic used in the GPU experiment report."""

import pytest

from scripts.summarize_prefill_consistency import summarize


def row(budget, repeat, rate):
    return {
        "case": {"name": "example"},
        "budget": budget,
        "block": 0,
        "repeat": repeat,
        "gpu": "GPU-example, H100",
        "requests": [{"prompt_hash": "same-input", "output_hash": "same-output"}],
        **dict.fromkeys(
            ("aggregate_tps", "p50_stream_tps", "p05_stream_tps", "p95_ttft", "p95_latency"),
            rate,
        ),
    }


def test_pairs_by_repeat_and_uses_geometric_ratio():
    rows = [row(8192, 1, 50), row(4096, 0, 100), row(8192, 0, 200), row(4096, 1, 100)]
    result = summarize(rows)[0]
    assert result["paired_waves"] == 2
    assert result["aggregate_tps_change_pct"] == pytest.approx(0)
    assert result["aggregate_tps_8192"] == 125
    assert result["aggregate_tps_min_change_pct"] == -50
    assert result["aggregate_tps_max_change_pct"] == 100
    assert result["output_hash_matches"] == 2


@pytest.mark.parametrize("mismatch", ["input", "gpu"])
def test_rejects_unmatched_conditions(mismatch):
    before, after = row(4096, 0, 100), row(8192, 0, 100)
    if mismatch == "input":
        after["requests"][0]["prompt_hash"] = "different-input"
    else:
        after["gpu"] = "GPU-other, H100"
    with pytest.raises(AssertionError):
        summarize([before, after])


def test_incomplete_pair_is_excluded_and_duplicate_is_rejected():
    before = row(4096, 0, 100)
    assert summarize([before]) == []
    with pytest.raises(ValueError, match="Duplicate wave"):
        summarize([before, before])
