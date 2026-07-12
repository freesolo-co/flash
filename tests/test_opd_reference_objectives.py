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
    anchored = RunConfig(**{**base.__dict__, "opd_objective_ids": ("c06",)})
    assert seconds_per_step(anchored, "A100-SXM-80GB") > seconds_per_step(base, "A100-SXM-80GB")

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


def test_raw_and_warm_reference_lineage_parse_and_round_trip():
    from flash.schema import spec_from_dict
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
