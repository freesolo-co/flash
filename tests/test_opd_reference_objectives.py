from __future__ import annotations

from types import SimpleNamespace

import pytest

from flash.engine.worker.adapter import (
    _tokenizer_signature,
    assert_reference_absent_from_optimizer,
    load_frozen_sft_reference,
    save_policy_adapter,
)
from flash.engine.worker.opd_objectives import (
    OPD_OBJECTIVES,
    ObjectiveView,
    forward_kl_topk_tail,
    sft_relative_top2_margin,
)


def test_forward_kl_topk_tail_matches_full_kl_when_k_covers_vocab():
    torch = pytest.importorskip("torch")
    policy = torch.tensor([[0.1, -0.2, 0.7], [1.2, -0.4, 0.0]], requires_grad=True)
    reference = torch.tensor([[0.5, 0.0, -0.3], [-0.1, 0.2, 0.9]])

    actual = forward_kl_topk_tail(policy, reference, top_k=3)
    ref_lp = reference.log_softmax(dim=-1)
    expected = (ref_lp.exp() * (ref_lp - policy.log_softmax(dim=-1))).sum(dim=-1).mean()

    assert torch.allclose(actual, expected, atol=1e-6)
    actual.backward()
    assert policy.grad is not None
    assert torch.isfinite(policy.grad).all()


def test_forward_kl_topk_tail_matches_manual_aggregated_tail():
    torch = pytest.importorskip("torch")
    policy = torch.tensor([[1.0, 0.2, -0.4, -1.0]], requires_grad=True)
    reference = torch.tensor([[2.0, 0.5, 0.0, -0.5]])

    actual = forward_kl_topk_tail(policy, reference, top_k=2)
    ref_lp = reference.log_softmax(dim=-1)
    policy_lp = policy.log_softmax(dim=-1)
    top_lp, top_idx = ref_lp.topk(2, dim=-1)
    top_policy_lp = policy_lp.gather(-1, top_idx)
    ref_tail = 1 - top_lp.exp().sum(dim=-1)
    policy_tail = 1 - top_policy_lp.exp().sum(dim=-1)
    expected = (
        (top_lp.exp() * (top_lp - top_policy_lp)).sum(dim=-1)
        + ref_tail * (ref_tail.log() - policy_tail.log())
    ).mean()

    assert torch.allclose(actual, expected, atol=1e-6)


def test_forward_kl_peaked_policy_matches_stable_oracle_and_tail_gradients():
    torch = pytest.importorskip("torch")
    policy = torch.tensor([[50.0, 49.0, 0.5, -0.5]], requires_grad=True)
    reference = torch.tensor([[4.0, 3.0, 2.0, 1.0]])

    actual = forward_kl_topk_tail(policy, reference, top_k=2)
    actual.backward()
    actual_grad = policy.grad.detach().clone()

    oracle_policy = policy.detach().clone().requires_grad_(True)
    reference_lp = reference.float().log_softmax(dim=-1)
    policy_lp = oracle_policy.float().log_softmax(dim=-1)
    top_ref_lp, top_idx = reference_lp.topk(2, dim=-1)
    top_mask = torch.zeros_like(policy_lp, dtype=torch.bool)
    top_mask.scatter_(1, top_idx, True)
    reference_tail_lp = torch.logsumexp(reference_lp.masked_fill(top_mask, -torch.inf), dim=-1)
    policy_tail_lp = torch.logsumexp(policy_lp.masked_fill(top_mask, -torch.inf), dim=-1)
    oracle = (top_ref_lp.exp() * (top_ref_lp - policy_lp.gather(-1, top_idx))).sum(dim=-1)
    oracle = oracle + reference_tail_lp.exp() * (reference_tail_lp - policy_tail_lp)
    oracle = oracle.mean()
    oracle.backward()

    assert torch.allclose(actual.detach(), oracle.detach(), atol=1e-6)
    assert torch.allclose(actual_grad, oracle_policy.grad, atol=1e-6)
    assert torch.count_nonzero(actual_grad[0, 2:]) == 2


def test_forward_kl_chunking_is_numerically_stable(monkeypatch):
    torch = pytest.importorskip("torch")
    import flash.engine.worker.opd_objectives as objectives

    generator = torch.Generator().manual_seed(42)
    policy = torch.randn(137, 97, generator=generator, requires_grad=True)
    reference = torch.randn(137, 97, generator=generator)
    monkeypatch.setattr(objectives, "_REFERENCE_ROW_CHUNK", 7)
    chunked = objectives.forward_kl_topk_tail(policy, reference, top_k=16)
    monkeypatch.setattr(objectives, "_REFERENCE_ROW_CHUNK", 1000)
    unchunked = objectives.forward_kl_topk_tail(policy, reference, top_k=16)

    assert torch.allclose(chunked, unchunked, atol=1e-6)


def test_sft_relative_margin_direction_ties_and_masking():
    torch = pytest.importorskip("torch")
    reference = torch.tensor([[3.0, 1.0, 0.0], [2.0, 2.0, 0.0], [4.0, 1.0, 0.0]])
    policy_good = torch.tensor([[4.0, 1.0, 0.0], [2.0, 2.0, 0.0], [0.0, 5.0, 0.0]])

    direct = sft_relative_top2_margin(policy_good[:2], reference[:2])
    assert float(direct) == pytest.approx(0.0)
    tied_reversed = torch.tensor([[0.0, 5.0, 0.0]])
    assert float(sft_relative_top2_margin(tied_reversed, reference[1:2])) == pytest.approx(0.0)

    plan = OPD_OBJECTIVES.plan(("c11",))
    evaluated = OPD_OBJECTIVES.evaluate(
        plan,
        ObjectiveView(
            {
                "completion_logits": policy_good,
                "reference_completion_logits": reference,
                "completion_mask": torch.tensor([True, True, False]),
            }
        ),
        base_term=torch.tensor(0.0),
    )
    assert float(evaluated.terms[0]) == pytest.approx(0.0)

    reversed_policy = torch.tensor([[0.0, 3.0, 0.0]])
    assert float(sft_relative_top2_margin(reversed_policy, reference[:1])) > 0


@pytest.mark.parametrize("objective_id", ["c06", "c11"])
def test_reference_objectives_normalize_bf16_terms_to_fp32_and_backward(objective_id):
    torch = pytest.importorskip("torch")
    policy = torch.tensor(
        [[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]], dtype=torch.bfloat16, requires_grad=True
    )
    reference = torch.tensor([[2.0, 1.0, 0.0], [3.0, 1.0, 0.0]], dtype=torch.bfloat16)
    plan = OPD_OBJECTIVES.plan((objective_id,))
    evaluated = OPD_OBJECTIVES.evaluate(
        plan,
        ObjectiveView(
            {
                "completion_logits": policy,
                "reference_completion_logits": reference,
                "completion_mask": torch.tensor([True, True]),
            }
        ),
        base_term=torch.tensor(0.0, dtype=torch.float32),
    )

    term = evaluated.terms[0]
    assert term.dtype == torch.float32
    term.backward()
    assert policy.grad is not None
    assert torch.isfinite(policy.grad).all()


@pytest.mark.parametrize("objective_id", ["c06", "c11"])
def test_reference_objectives_require_completion_mask(objective_id):
    torch = pytest.importorskip("torch")
    plan = OPD_OBJECTIVES.plan((objective_id,))

    with pytest.raises(RuntimeError, match="completion_mask"):
        OPD_OBJECTIVES.evaluate(
            plan,
            ObjectiveView(
                {
                    "completion_logits": torch.zeros(1, 3),
                    "reference_completion_logits": torch.zeros(1, 3),
                }
            ),
            base_term=torch.tensor(0.0),
        )


def test_reference_forward_restores_policy_adapter_on_success_and_failure():
    torch = pytest.importorskip("torch")
    from flash.engine.worker.opd import _reference_forward_logits

    class Model:
        def __init__(self):
            self.active_adapter = "default"
            self.fail = False
            self.training = True

        def set_adapter(self, name, *, inference_mode=False):
            self.active_adapter = name
            self.inference_mode = inference_mode

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

        def __call__(self, input_ids, **_kwargs):
            assert self.active_adapter == "sft_reference"
            assert self.inference_mode
            if self.fail:
                raise RuntimeError("boom")
            return SimpleNamespace(logits=torch.ones((*input_ids.shape, 4), requires_grad=True))

    model = Model()
    output = _reference_forward_logits(model, torch.tensor([[1, 2]]))
    assert model.active_adapter == "default"
    assert not model.inference_mode
    assert model.training
    assert not output.requires_grad

    model.fail = True
    with pytest.raises(RuntimeError, match="boom"):
        _reference_forward_logits(model, torch.tensor([[1, 2]]))
    assert model.active_adapter == "default"
    assert not model.inference_mode
    assert model.training


def test_load_frozen_reference_uses_named_adapter_and_restores_policy(monkeypatch):
    torch = pytest.importorskip("torch")
    from transformers import AutoTokenizer

    import flash.engine.worker.adapter as adapter

    class Tok:
        vocab_size = 2
        bos_token_id = 0
        eos_token_id = 1
        pad_token_id = 0
        unk_token_id = None

        def __len__(self):
            return 2

        def get_vocab(self):
            return {"a": 0, "b": 1}

    reference_param = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

    class Model:
        active_adapter = "default"

        def __init__(self):
            self.calls = []
            self._checkpoint_conversion_mapping = {"old": "new"}

        def get_base_model(self):
            return self

        def load_adapter(self, path, **kwargs):
            self.calls.append((path, kwargs))
            self.active_adapter = kwargs["adapter_name"]
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        def set_adapter(self, name, *, inference_mode=False):
            self.active_adapter = name
            self.inference_mode = inference_mode

        def named_parameters(self):
            return [("layer.sft_reference.weight", reference_param)]

    monkeypatch.setattr(
        adapter._w,
        "JOB_SPEC",
        SimpleNamespace(train=SimpleNamespace(opd_reference_adapter="repo:prefix")),
    )
    monkeypatch.setattr(adapter, "_download_adapter", lambda _ref: "/tmp/reference")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: Tok())

    model = Model()
    assert load_frozen_sft_reference(model, "fake/model", Tok()) is model
    assert model.active_adapter == "default"
    assert model.calls == [
        (
            "/tmp/reference",
            {
                "adapter_name": "sft_reference",
                "is_trainable": False,
                "key_mapping": {"old": "new"},
            },
        )
    ]


def test_reference_parameters_are_frozen_and_absent_from_optimizer():
    torch = pytest.importorskip("torch")
    policy = torch.nn.Parameter(torch.tensor(1.0))
    reference = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)

    class Model:
        def named_parameters(self):
            return [("layer.default.weight", policy), ("layer.sft_reference.weight", reference)]

    optimizer = torch.optim.SGD([policy], lr=0.1)
    assert_reference_absent_from_optimizer(Model(), optimizer)
    (policy - reference).square().backward()
    assert policy.grad is not None
    assert reference.grad is None

    bad_optimizer = torch.optim.SGD([policy, reference], lr=0.1)
    with pytest.raises(RuntimeError, match="absent from the optimizer"):
        assert_reference_absent_from_optimizer(Model(), bad_optimizer)


def test_policy_only_save_and_sync_select_default_adapter(tmp_path):
    from flash.engine.worker.opd_vllm import OpdVllmRolloutEngine

    class Model:
        def __init__(self):
            self.calls = []
            self.peft_config = {"default": object(), "sft_reference": object()}

        def save_pretrained(self, path, **kwargs):
            self.calls.append((path, kwargs))

    model = Model()
    save_policy_adapter(model, str(tmp_path / "save"))
    assert model.calls[-1][1] == {"selected_adapters": ["default"]}

    engine = object.__new__(OpdVllmRolloutEngine)
    engine._lora_int_id = None
    engine._sync_dirs = []
    engine._version = 0
    engine.adapter_root = str(tmp_path / "sync")
    engine._LoRARequest = lambda *args: args
    engine.sync_from_model(model)
    assert model.calls[-1][1] == {"selected_adapters": ["default"]}


def test_tokenizer_signature_detects_vocab_and_special_token_mismatch():
    class Tok:
        def __init__(self, size, vocab, eos, tokens=None):
            self.size = size
            self.vocab_size = vocab
            self.bos_token_id = 1
            self.eos_token_id = eos
            self.pad_token_id = 0
            self.unk_token_id = 2
            self.tokens = tokens or {"a": 0, "b": 1}

        def __len__(self):
            return self.size

        def get_vocab(self):
            return self.tokens

    baseline = _tokenizer_signature(Tok(10, 9, 3))
    assert baseline == _tokenizer_signature(Tok(10, 9, 3))
    assert baseline != _tokenizer_signature(Tok(11, 9, 3))
    assert baseline != _tokenizer_signature(Tok(10, 9, 4))
    assert baseline != _tokenizer_signature(Tok(10, 9, 3, {"a": 1, "b": 0}))


def test_reference_objectives_declare_extra_cost_and_vram():
    from flash.cost.analytical import seconds_per_step
    from flash.cost.types import RunConfig
    from flash.engine.vram import model_required_vram_gb

    base = RunConfig(
        model_id="Qwen/Qwen3.5-0.8B",
        method="opd",
        steps=1,
        seq_len=1024,
        completion_len=128,
        batch_size=1,
        group_size=1,
    )
    anchored = RunConfig(
        **{
            **base.__dict__,
            "opd_objective_ids": ("c06",),
            "opd_reference_lora_rank": 128,
        }
    )
    assert seconds_per_step(anchored, "A100-SXM-80GB") > seconds_per_step(base, "A100-SXM-80GB")
    assert anchored.train_knobs()["opd_reference_lora_rank"] == 128

    common = {
        "max_context_tokens": 1024,
        "max_completion_tokens": 128,
        "batch_size": 1,
        "group_size": 1,
    }
    plain_vram = model_required_vram_gb("Qwen/Qwen3.5-0.8B", "opd", train=common, headroom=1.0)
    anchored_vram = model_required_vram_gb(
        "Qwen/Qwen3.5-0.8B",
        "opd",
        train={**common, "opd_objective_ids": ("c11",)},
        headroom=1.0,
    )
    assert anchored_vram >= plain_vram


def test_reference_rank_increases_vram_and_changes_boundary_routing():
    from flash.engine.vram import model_required_vram_gb
    from flash.providers.base import cheapest_gpu

    common = {
        "lora_rank": 8,
        "max_context_tokens": 2048,
        "max_completion_tokens": 1024,
        "batch_size": 1,
        "group_size": 4,
        "opd_objective_ids": ("c06",),
    }
    low_rank_need = model_required_vram_gb(
        "openbmb/MiniCPM5-1B",
        "opd",
        train={**common, "opd_reference_lora_rank": 8},
    )
    high_rank_need = model_required_vram_gb(
        "openbmb/MiniCPM5-1B",
        "opd",
        train={**common, "opd_reference_lora_rank": 128},
    )

    assert high_rank_need > low_rank_need
    assert cheapest_gpu(high_rank_need) != cheapest_gpu(low_rank_need)


def test_moe_reference_rank_uses_total_resident_params_and_rejects_b200_boundary():
    from flash.engine.vram import estimate_vram_gb, model_required_vram_gb
    from flash.providers.base import UnsupportedGpuError, provisional_gpu

    common = {
        "seq_len": 2048,
        "max_tokens": 512,
        "lora_rank": 8,
        "batch_size": 2,
        "group_size": 16,
        "active_params_b": 3.0,
        "frozen_reference": True,
        "reference_lora_rank": 64,
    }
    with_reference = estimate_vram_gb(35.0, "opd", **common)
    without_reference = estimate_vram_gb(
        35.0,
        "opd",
        **{**common, "reference_lora_rank": None},
    )
    expected_total_residency = (64 / 16.0) * (0.3 + 0.04 * 35.0) / 4.0
    active_only_residency = (64 / 16.0) * (0.3 + 0.04 * 3.0) / 4.0

    assert with_reference - without_reference == pytest.approx(expected_total_residency)
    assert with_reference - without_reference > active_only_residency

    train = {
        "lora_rank": 8,
        "max_context_tokens": 2048,
        "max_completion_tokens": 512,
        "batch_size": 2,
        "group_size": 16,
        "opd_objective_ids": ("c06",),
        "opd_reference_lora_rank": 64,
    }
    assert model_required_vram_gb("Qwen/Qwen3.6-35B-A3B", "opd", train=train) == 181
    with pytest.raises(UnsupportedGpuError):
        provisional_gpu("Qwen/Qwen3.6-35B-A3B", "opd", train=train)


def test_dense_reference_residency_is_counted_once_and_active_hint_is_neutral():
    from flash.engine.vram import estimate_vram_gb

    common = {
        "seq_len": 1024,
        "max_tokens": 128,
        "lora_rank": 8,
        "batch_size": 1,
        "group_size": 1,
        "frozen_reference": True,
        "reference_lora_rank": 64,
    }
    dense = estimate_vram_gb(4.0, "opd", active_params_b=None, **common)
    dense_with_hint = estimate_vram_gb(4.0, "opd", active_params_b=4.0, **common)
    no_reference = estimate_vram_gb(
        4.0,
        "opd",
        active_params_b=None,
        **{**common, "reference_lora_rank": None},
    )
    expected_residency = (64 / 16.0) * (0.3 + 0.04 * 4.0) / 4.0

    assert dense == pytest.approx(dense_with_hint)
    assert dense - no_reference == pytest.approx(expected_residency)


def test_raw_and_warm_reference_lineage_parse_and_round_trip():
    from flash.cost.spec import runconfig_from_spec
    from flash.schema import spec_from_dict
    from flash.schema.fields import ConfigError
    from flash.spec import JobSpec

    base = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "opd",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
    }
    raw = spec_from_dict(
        {
            **base,
            "train": {
                "opd_objective_ids": ["c06"],
                "opd_reference_adapter": "sft-run",
            },
        }
    )
    assert raw.train.init_from_adapter == ""
    assert raw.train.opd_reference_adapter == "sft-run"

    warm = spec_from_dict(
        {
            **base,
            "train": {
                "opd_objective_ids": ["c11"],
                "init_from_adapter": "sft-run/step-20",
                "opd_reference_adapter": "sft-run/step-20",
            },
        }
    )
    restored = JobSpec.from_json(warm.to_json())
    assert restored.train.init_from_adapter == "sft-run/step-20"
    assert restored.train.opd_reference_adapter == "sft-run/step-20"

    internal = JobSpec.from_dict(
        {
            **warm.to_dict(),
            "train": {**warm.to_dict()["train"], "opd_reference_lora_rank": 128},
        }
    )
    assert internal.train.opd_reference_lora_rank == 128
    assert runconfig_from_spec(internal).opd_reference_lora_rank == 128

    with pytest.raises(ConfigError, match="unknown key"):
        spec_from_dict(
            {
                **base,
                "train": {
                    "opd_objective_ids": ["c06"],
                    "opd_reference_adapter": "sft-run",
                    "opd_reference_lora_rank": 1,
                },
            }
        )


def test_reference_resolution_requires_matching_sft_source(monkeypatch):
    from flash import runner
    from flash.spec import JobSpec, TrainSpec

    target = JobSpec(
        model="Qwen/Qwen3.5-0.8B",
        algorithm="opd",
        train=TrainSpec(opd_reference_adapter="sft-run"),
    )

    def status(algorithm="sft", model=target.model):
        source = JobSpec(
            model=model,
            algorithm=algorithm,
            run_id="sft-run",
            train=TrainSpec(hf_repo="owner/repo"),
        )
        return SimpleNamespace(
            state="done",
            spec=source.to_dict(),
            billing_context=None,
            platform_context=None,
        )

    monkeypatch.setattr(runner, "get_status", lambda _run_id: status())
    monkeypatch.setattr("flash.runner.checkpoints.final_adapter_exists", lambda _spec: True)
    resolved = runner._resolve_opd_reference_adapter(target)
    assert resolved.train.opd_reference_adapter == "owner/repo:sft/sft-run"
    assert resolved.train.opd_reference_lora_rank == 32
    assert resolved.train.init_from_adapter == ""

    monkeypatch.setattr(runner, "get_status", lambda _run_id: status(algorithm="opd"))
    with pytest.raises(ValueError, match="must reference an SFT run"):
        runner._resolve_opd_reference_adapter(target)

    monkeypatch.setattr(runner, "get_status", lambda _run_id: status(model="other/model"))
    with pytest.raises(ValueError, match="does not match target model"):
        runner._resolve_opd_reference_adapter(target)


def test_reference_field_is_opd_only_and_required_by_reference_objectives():
    from flash.schema import spec_from_dict
    from flash.schema.fields import ConfigError

    base = {
        "model": "Qwen/Qwen3.5-0.8B",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
    }
    with pytest.raises(ConfigError, match="only valid when algorithm"):
        spec_from_dict(
            {
                **base,
                "algorithm": "grpo",
                "train": {"opd_reference_adapter": "sft-run"},
            }
        )
    with pytest.raises(ConfigError, match="is required"):
        spec_from_dict(
            {
                **base,
                "algorithm": "opd",
                "train": {"opd_objective_ids": ["c06"]},
            }
        )


def test_c06_composes_with_c09_on_one_shared_view():
    """c06 (frozen reference) and c09 (repetition) plan and evaluate together, both terms live."""
    torch = pytest.importorskip("torch")
    from flash.engine.worker.opd_repetition import analyze_repetition

    plan = OPD_OBJECTIVES.plan(("c06", "c09"))
    assert plan.objective_ids == ("c06", "c09")
    requirements = plan.requirements
    assert requirements.frozen_sft_reference
    assert requirements.completion_mask
    assert requirements.student_logits
    assert requirements.repetition_weighting

    rows, vocab = 8, 16
    policy = torch.zeros(rows, vocab, requires_grad=True)
    reference = torch.randn(rows, vocab, generator=torch.Generator().manual_seed(7))
    # a repeating tail so c09's loop-closing unlikelihood contributes a real term.
    completion_ids = (9, 8, 1, 2, 1, 2, 1, 2)

    evaluated = OPD_OBJECTIVES.evaluate(
        plan,
        ObjectiveView(
            {
                "completion_logits": policy,
                "reference_completion_logits": reference,
                "completion_mask": torch.ones(rows, dtype=torch.bool),
                "completion_ids": completion_ids,
                "forced": (False,) * rows,
                "repetition_analysis": analyze_repetition(list(completion_ids)),
                "sequence_weight": 1.0,
            }
        ),
        base_term=torch.tensor(0.0, dtype=torch.float32),
    )

    assert len(evaluated.terms) == 2
    total = evaluated.terms[0] + evaluated.terms[1]
    assert total.dtype == torch.float32
    total.backward()
    assert policy.grad is not None
    assert torch.isfinite(policy.grad).all()
    assert any(key.startswith("opd/objectives/c06/") for key in evaluated.metrics)
    assert any(key.startswith("opd/objectives/c09/") for key in evaluated.metrics)


def test_c11_composes_with_c10_on_one_shared_view():
    """c11 (reference margin) and c10 (tail entropy) share a view and both back-propagate."""
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod
    from flash.engine.worker.opd_objectives import CvarEntropyConfig

    plan = OPD_OBJECTIVES.plan(("c11", "c10"))
    assert plan.objective_ids == ("c11", "c10")
    requirements = plan.requirements
    assert requirements.frozen_sft_reference
    assert requirements.completion_mask
    assert requirements.student_logits
    assert requirements.position_statistics

    rows, vocab = 4, 8
    policy = torch.zeros(rows, vocab, requires_grad=True)
    # reference has a clear top-1/top-2 ordering the uniform policy violates, so c11 is positive.
    reference = torch.tensor([[3.0, 1.0] + [0.0] * (vocab - 2)] * rows)
    stats = opd_mod._completion_position_statistics(policy, ())

    evaluated = OPD_OBJECTIVES.evaluate(
        plan,
        ObjectiveView(
            {
                "completion_logits": policy,
                "reference_completion_logits": reference,
                "completion_mask": torch.ones(rows, dtype=torch.bool),
                "position_statistics": stats,
                "entropy_for_rows": lambda indices: opd_mod._chunked_row_entropies(
                    policy, indices, grad=True
                ),
                # floor above the uniform-row entropy so the tail term activates.
                "cvar_entropy_config": CvarEntropyConfig(fraction=1.0, floor=10.0, coef=1.0),
                "base_term": torch.tensor(0.0),
            }
        ),
        base_term=torch.tensor(0.0),
    )

    assert len(evaluated.terms) == 2
    total = evaluated.terms[0] + evaluated.terms[1]
    total.backward()
    assert policy.grad is not None
    assert torch.isfinite(policy.grad).all()
    assert any(key.startswith("opd/objectives/c11/") for key in evaluated.metrics)
    assert evaluated.metrics["opd/objectives/c10/activation"] == 1.0


# --- c12: frozen anchor (c06) + greedy-sidecar unlikelihood (c07) + position-0 eos safety (c05) ---


def _c12_shared_view(torch, *, mask=None, eos_ids=(4, 5), termination_cause="eos"):
    """one controlled per-sample view exposing exactly the inputs c06, c05, and c12 read.

    prompt_len - 1 (the first completion row) is both the c06 anchor's leading row and the c05
    position-0 halting row, so a single leaf tensor feeds every component and the additive
    decomposition c12 == c06 + c05-safety can be checked on both value and gradient."""
    vocab, prompt_len, comp_len = 6, 2, 3
    seq_len = prompt_len - 1 + comp_len
    generator = torch.Generator().manual_seed(19)
    sample_logits = torch.randn(seq_len, vocab, generator=generator, requires_grad=True)
    completion_logits = sample_logits[prompt_len - 1 : prompt_len - 1 + comp_len]
    reference = torch.randn(comp_len, vocab, generator=generator)
    if mask is None:
        mask = torch.ones(comp_len, dtype=torch.bool)
    return sample_logits, completion_logits, reference, ObjectiveView(
        {
            "sample_logits": sample_logits,
            "completion_logits": completion_logits,
            "reference_completion_logits": reference,
            "completion_mask": mask,
            "prompt_len": prompt_len,
            "empty_rollout": False,
            "termination_cause": termination_cause,
            "stop_sequences": (),
            "eos_ids": tuple(eos_ids),
            "entropy": None,
        }
    )


def test_c12_term_and_grad_equal_sum_of_c06_anchor_and_c05_position0_safety():
    """c12's single term and its gradient equal, exactly, c06's frozen forward-kl anchor plus c05's
    standalone position-0 eos safety evaluated on the same inputs (the c07 sidecar half is applied
    once at batch level via the greedy_sidecar requirement, checked separately)."""
    torch = pytest.importorskip("torch")
    sample_logits, _completion, _reference, view = _c12_shared_view(torch)
    base = torch.tensor(0.0, dtype=torch.float32)

    c12_term = OPD_OBJECTIVES.evaluate(OPD_OBJECTIVES.plan(("c12",)), view, base_term=base).terms[0]
    c06_term = OPD_OBJECTIVES.evaluate(OPD_OBJECTIVES.plan(("c06",)), view, base_term=base).terms[0]
    c05_term = OPD_OBJECTIVES.evaluate(OPD_OBJECTIVES.plan(("c05",)), view, base_term=base).terms[0]

    assert torch.allclose(c12_term, c06_term + c05_term, atol=1e-6)
    grad_c12 = torch.autograd.grad(c12_term, sample_logits, retain_graph=True)[0]
    grad_sum = torch.autograd.grad(c06_term + c05_term, sample_logits, retain_graph=True)[0]
    assert torch.allclose(grad_c12, grad_sum, atol=1e-6)
    # both halves are live: the anchor touches every completion row, the safety touches the
    # position-0 row, so the composed gradient is strictly non-trivial.
    assert torch.isfinite(grad_c12).all()
    assert float(grad_c12.abs().sum()) > 0


def test_c12_metrics_are_namespaced_and_do_not_collide_with_c06():
    """c12 exposes the anchor and the safety submetrics under its own id; composing it with c06
    keeps both forward_kl metrics distinct instead of colliding."""
    torch = pytest.importorskip("torch")
    _sample, _completion, _reference, view = _c12_shared_view(torch)
    evaluated = OPD_OBJECTIVES.evaluate(
        OPD_OBJECTIVES.plan(("c12", "c06")), view, base_term=torch.tensor(0.0)
    )
    assert "opd/objectives/c12/forward_kl" in evaluated.metrics
    assert "opd/objectives/c12/position0_eos_probability" in evaluated.metrics
    assert "opd/objectives/c12/position0_eos_rank" in evaluated.metrics
    assert "opd/objectives/c12/position0_eos_margin" in evaluated.metrics
    assert "opd/objectives/c06/forward_kl" in evaluated.metrics
    # same underlying anchor, reported once per objective id, no duplicate-key error.
    assert evaluated.metrics["opd/objectives/c12/forward_kl"] == pytest.approx(
        evaluated.metrics["opd/objectives/c06/forward_kl"]
    )


def test_c12_anchor_masks_grammar_forced_rows():
    """the frozen forward-kl anchor honors the completion mask, so grammar-forced rows never
    contribute — identical to c06's masking."""
    torch = pytest.importorskip("torch")
    mask = torch.tensor([True, False, True])
    _sample, completion, reference, view = _c12_shared_view(torch, mask=mask)
    evaluated = OPD_OBJECTIVES.evaluate(
        OPD_OBJECTIVES.plan(("c12",)), view, base_term=torch.tensor(0.0)
    )
    expected_anchor = forward_kl_topk_tail(completion[mask], reference[mask])
    assert evaluated.metrics["opd/objectives/c12/forward_kl"] == pytest.approx(
        float(expected_anchor.detach())
    )


def test_c12_position0_safety_aggregates_the_complete_eos_set():
    """the position-0 halting probability sums over EVERY eos id, not just one — the aggregate
    halting-eos mass matches the softmax mass on the full eos set."""
    torch = pytest.importorskip("torch")
    sample_logits, _completion, _reference, view = _c12_shared_view(torch, eos_ids=(3, 4, 5))
    evaluated = OPD_OBJECTIVES.evaluate(
        OPD_OBJECTIVES.plan(("c12",)), view, base_term=torch.tensor(0.0)
    )
    expected = float(torch.softmax(sample_logits[1].detach(), dim=-1)[[3, 4, 5]].sum())
    assert evaluated.metrics["opd/objectives/c12/position0_eos_probability"] == pytest.approx(
        expected
    )


def test_c12_stop_sequences_disable_only_the_safety_half():
    """with stop sequences present the position-0 safety is ineligible (as in c05), leaving c12 as
    the pure frozen anchor — no safety term, no safety metrics."""
    torch = pytest.importorskip("torch")
    _sample, completion, reference, _view = _c12_shared_view(torch)
    view = ObjectiveView(
        {
            **{
                "sample_logits": _sample,
                "completion_logits": completion,
                "reference_completion_logits": reference,
                "completion_mask": torch.ones(3, dtype=torch.bool),
                "prompt_len": 2,
                "empty_rollout": False,
                "termination_cause": "stop_sequence",
                "stop_sequences": ("</answer>",),
                "eos_ids": (4, 5),
                "entropy": None,
            }
        }
    )
    evaluated = OPD_OBJECTIVES.evaluate(
        OPD_OBJECTIVES.plan(("c12",)), view, base_term=torch.tensor(0.0)
    )
    anchor_only = forward_kl_topk_tail(completion, reference)
    assert torch.allclose(evaluated.terms[0], anchor_only, atol=1e-6)
    assert "opd/objectives/c12/position0_eos_probability" not in evaluated.metrics


def test_c12_config_enables_only_position0_safety_no_floor_recovery_or_rkl_eos():
    """c12 fixes only the halting-eos coefficient: no entropy floor 1.75, no empty-rollout recovery,
    no reverse-kl eos boundary. worker-side aggregation (entropy floor / empty recovery derive from
    per-definition config) therefore stays off for a c12-only plan."""
    from flash.engine.worker.opd_objectives import ObjectiveConfig

    definition = OPD_OBJECTIVES.definitions["c12"]
    assert definition.config == ObjectiveConfig(position0_eos_coef=1.0)
    assert definition.config.entropy_floor is None
    assert definition.config.entropy_floor_coef == 0.0
    assert definition.config.recover_empty_rollouts is False
    assert definition.config.eos_in_rkl is False
    assert definition.config.teacher_boundary_probability is None

    plan = OPD_OBJECTIVES.plan(("c12",))
    # mirror opd.py's aggregation: entropy floor is enabled only if some config sets entropy_floor.
    assert not any(d.config.entropy_floor is not None for d in plan.definitions)
    assert not any(d.config.recover_empty_rollouts for d in plan.definitions)


def test_c12_requirements_are_component_union_without_c09_or_extra_work():
    """c12 requests frozen reference, completion mask, student logits, and greedy sidecar — the union
    of its c06/c07/c05 components — and NOT c09 sequence weighting or c10/c13 work."""
    plan = OPD_OBJECTIVES.plan(("c12",))
    req = plan.requirements
    assert req.frozen_sft_reference
    assert req.completion_mask
    assert req.student_logits
    assert req.greedy_sidecar
    # explicitly not c09 sequence weighting, not empty-rollout recovery machinery, not c10/c13.
    assert not req.repetition_weighting
    assert not req.empty_rollouts
    assert not req.position_statistics
    assert not req.candidate_topk
    assert not req.sampled_primary
    assert not req.teacher_scores


def test_c12_reference_and_sidecar_work_is_requested_once_when_composed():
    """composing c12 with c06 (also frozen reference) and c07 (also greedy sidecar) still aggregates
    to a single reference-forward flag and a single sidecar flag — the worker does that work once."""
    solo = OPD_OBJECTIVES.plan(("c12",)).requirements
    composed = OPD_OBJECTIVES.plan(("c12", "c06", "c07")).requirements
    assert solo.frozen_sft_reference is composed.frozen_sft_reference is True
    assert solo.greedy_sidecar is composed.greedy_sidecar is True
    # aggregation is idempotent: no per-objective duplication of the shared boolean requirements.
    assert composed.completion_mask is True


def test_c12_composes_with_c10_and_c13_without_them_firing():
    """c12 fires its anchor+safety while co-selected c10 (empty tail) and c13 (no candidate term)
    produce metrics but no term; c12 remains the only contributed term."""
    torch = pytest.importorskip("torch")
    from flash.engine.worker.opd_objectives import CvarEntropyConfig, PositionStatistics

    _sample_logits, _completion, _reference, base_view = _c12_shared_view(torch)
    candidate = SimpleNamespace(
        term=None,
        selected_prefixes=0,
        scored_prefixes=0,
        duplicate_surfaces=0,
        invalid_candidates=0,
        budget_exhausted=False,
        cache_hits=0,
        requests=0,
    )
    view = ObjectiveView(
        {
            **dict(base_view.values),
            "position_statistics": PositionStatistics(),
            "entropy_for_rows": lambda _indices: None,
            "cvar_entropy_config": CvarEntropyConfig(fraction=0.5, floor=0.0, coef=1.0),
            "candidate_topk": candidate,
        }
    )
    evaluated = OPD_OBJECTIVES.evaluate(
        OPD_OBJECTIVES.plan(("c12", "c10", "c13")), view, base_term=torch.tensor(0.0)
    )
    assert len(evaluated.terms) == 1  # only c12 contributed a term
    assert evaluated.metrics["opd/objectives/c10/activation"] == 0.0
    assert evaluated.metrics["opd/objectives/c13/selected_prefixes"] == 0.0
    assert "opd/objectives/c12/forward_kl" in evaluated.metrics


def test_c12_is_registered_in_canonical_order_between_c11_and_c13():
    from flash.opd_objectives import OPD_OBJECTIVE_IDS

    assert OPD_OBJECTIVES.objective_ids == OPD_OBJECTIVE_IDS
    ids = list(OPD_OBJECTIVE_IDS)
    assert ids.index("c11") + 1 == ids.index("c12")
    assert ids.index("c12") + 1 == ids.index("c13")


def test_c12_default_and_c0_paths_are_unchanged():
    """selecting nothing or c0 still plans to no work and contributes no term, exactly as before."""
    torch = pytest.importorskip("torch")
    assert OPD_OBJECTIVES.plan(()).definitions == ()
    c0_plan = OPD_OBJECTIVES.plan(("c0",))
    assert c0_plan.requirements == type(c0_plan.requirements)()
    evaluated = OPD_OBJECTIVES.evaluate(c0_plan, ObjectiveView({}), base_term=torch.tensor(0.0))
    assert evaluated.terms == ()
    assert evaluated.metrics == {}


def test_c12_requires_reference_adapter_and_parses_raw_reference_lineage():
    from flash.schema import spec_from_dict
    from flash.schema.fields import ConfigError

    base = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "opd",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
    }
    with pytest.raises(ConfigError, match="opd_reference_adapter is required"):
        spec_from_dict({**base, "train": {"opd_objective_ids": ["c12"]}})

    raw = spec_from_dict(
        {
            **base,
            "train": {"opd_objective_ids": ["c12"], "opd_reference_adapter": "sft-run"},
        }
    )
    assert raw.train.opd_objective_ids == ("c12",)
    assert raw.train.opd_reference_adapter == "sft-run"
    assert raw.train.init_from_adapter == ""


def test_c12_declares_reference_extra_cost_and_vram():
    from flash.cost.analytical import seconds_per_step
    from flash.cost.types import RunConfig
    from flash.engine.vram import model_required_vram_gb

    base = RunConfig(
        model_id="Qwen/Qwen3.5-0.8B",
        method="opd",
        steps=1,
        seq_len=1024,
        completion_len=128,
        batch_size=1,
        group_size=1,
    )
    composed = RunConfig(
        **{**base.__dict__, "opd_objective_ids": ("c12",), "opd_reference_lora_rank": 128}
    )
    assert seconds_per_step(composed, "A100-SXM-80GB") > seconds_per_step(base, "A100-SXM-80GB")

    common = {
        "max_context_tokens": 1024,
        "max_completion_tokens": 128,
        "batch_size": 1,
        "group_size": 1,
    }
    plain_vram = model_required_vram_gb("Qwen/Qwen3.5-0.8B", "opd", train=common, headroom=1.0)
    c12_vram = model_required_vram_gb(
        "Qwen/Qwen3.5-0.8B",
        "opd",
        train={**common, "opd_objective_ids": ("c12",), "opd_reference_lora_rank": 128},
        headroom=1.0,
    )
    assert c12_vram >= plain_vram
