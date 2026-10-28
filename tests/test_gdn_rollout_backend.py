"""Scope the GDN prefill mitigation and exercise its actual Hydra representation."""

import pytest

from flash.engine.worker.train.entry.backend_common import rollout_gdn_prefill_overrides


@pytest.mark.parametrize("n_gpus", [2, 4, 8])
def test_hopper_gdn_tensor_parallel_rollouts_select_triton(n_gpus):
    assert rollout_gdn_prefill_overrides(True, (9, 0), n_gpus=n_gpus) == [
        (
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config="
            "{gdn_prefill_backend:triton}"
        )
    ]


@pytest.mark.parametrize(
    ("gdn_hybrid", "cc", "n_gpus"),
    [
        (False, (9, 0), 2),
        (True, (9, 0), 1),
        (True, (8, 0), 2),
        (True, (8, 9), 2),
        (True, (10, 0), 2),
        (True, (12, 0), 2),
        (True, None, 2),
    ],
)
def test_other_rollouts_keep_existing_engine_defaults(gdn_hybrid, cc, n_gpus):
    assert rollout_gdn_prefill_overrides(gdn_hybrid, cc, n_gpus=n_gpus) == []
