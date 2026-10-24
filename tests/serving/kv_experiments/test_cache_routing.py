import pytest

from flash.serving.deepinfer.cache_routing import RankCandidate, choose_rank


def test_warm_rank_can_beat_idle_cold_rank():
    ranks = [RankCandidate(0, 8000, 1000, 1, 8), RankCandidate(1, 0, 0, 0, 8)]
    assert choose_rank(ranks, 9000, 100) == 0


def test_queue_cost_and_capacity_can_outweigh_cache_hit():
    cold = RankCandidate(1, 0, 0, 0, 8)
    assert choose_rank([RankCandidate(0, 8000, 10000, 7, 8), cold], 9000, 100) == 1
    assert choose_rank([RankCandidate(0, 8000, 0, 8, 8), cold], 9000, 100) == 1


def test_rejects_impossible_claims_and_full_pool():
    with pytest.raises(ValueError, match="cache claim"):
        choose_rank([RankCandidate(0, 100, 0, 0, 1)], 10, 0)
    with pytest.raises(OverflowError):
        choose_rank([RankCandidate(0, 0, 0, 1, 1)], 10, 0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_nonfinite_or_negative_work_cannot_change_routing(value):
    with pytest.raises(ValueError, match="Invalid"):
        choose_rank([RankCandidate(0, 0, value, 0, 1)], 10, 0)
    with pytest.raises(ValueError, match="Invalid"):
        choose_rank([RankCandidate(0, 0, 0, 0, 1)], 10, value)
