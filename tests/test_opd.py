"""On-policy distillation (opd): groupwise reverse-KL (gkd) cross-tokenizer alignment, the teacher
client, spec/cost plumbing, and the loss math.

All CPU-only. The loss-math tests need torch and are skipped where it is unavailable (they run in
CI, which has the training stack).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from flash.engine.worker.teacher import TeacherClient
from flash.engine.worker.tokenizer_align import (
    StudentToken,
    TeacherToken,
    groupwise_alignment,
    groupwise_coverage,
)


def _student(spans):
    return [StudentToken(token_id=i, start=a, end=b) for i, (a, b) in enumerate(spans)]


def _teacher(spans):
    """TeacherToken list from (start, end) spans; logprob = -(index+1)."""
    return [
        TeacherToken(text="", logprob=-(i + 1.0), start=a, end=b) for i, (a, b) in enumerate(spans)
    ]


# --------------------------------------------------------------------------------------------------
# gkd groupwise alignment — the coarsest common refinement of the two tokenizations
# --------------------------------------------------------------------------------------------------
def test_gkd_groups_are_one_per_shared_boundary_when_tokenizers_agree():
    # Both tokenizers segment identically -> every student token is its own group.
    student = _student([(0, 2), (2, 5)])
    teacher = _teacher([(0, 2), (2, 5)])
    groups = groupwise_alignment(student, teacher)
    assert [s_idx for s_idx, _ in groups] == [[0], [1]]
    assert groups[0][1] == -1.0  # teacher logprob of the first span
    assert groups[1][1] == -2.0
    assert groupwise_coverage(groups, len(student)) == 1.0


def test_gkd_span_grows_across_disagreement_and_covers_every_token():
    # Teacher emits one token where the student emits two; no shared interior boundary at char 3,
    # so both student tokens fall in a single group summing the (single) teacher logprob.
    student = _student([(0, 3), (3, 6)])
    teacher = _teacher([(0, 6)])
    groups = groupwise_alignment(student, teacher)
    assert len(groups) == 1
    assert groups[0][0] == [0, 1]  # both student tokens grouped together
    assert groups[0][1] == -1.0
    assert groupwise_coverage(groups, len(student)) == 1.0  # NO masking


def test_gkd_partial_agreement_splits_at_shared_boundaries_only():
    # Shared boundaries at chars 0 and 2; the student's interior boundary at 4 is not shared.
    student = _student([(0, 2), (2, 4), (4, 6)])
    teacher = _teacher([(0, 2), (2, 6)])
    groups = groupwise_alignment(student, teacher)
    assert [s_idx for s_idx, _ in groups] == [[0], [1, 2]]
    assert groupwise_coverage(groups, len(student)) == 1.0


def test_gkd_merges_leading_student_only_span_so_no_token_is_dropped():
    # Student starts at char 0 but the first teacher token starts at char 2, so [0,2) is a
    # student-only span. Those tokens must NOT be dropped — they merge into the first teacher-bearing
    # group (coverage stays 100%).
    student = _student([(0, 1), (1, 2), (2, 5)])
    teacher = _teacher([(2, 5)])
    groups = groupwise_alignment(student, teacher)
    assert len(groups) == 1
    assert groups[0][0] == [0, 1, 2]  # all three student tokens covered
    assert groupwise_coverage(groups, len(student)) == 1.0


def test_gkd_empty_inputs_yield_no_groups():
    assert groupwise_alignment([], _teacher([(0, 1)])) == []
    assert groupwise_alignment(_student([(0, 1)]), []) == []
    assert groupwise_coverage([], 3) == 0.0


# --------------------------------------------------------------------------------------------------
# student tokenization: the loss trains the SAMPLED ids (not a re-tokenization of decoded text)
# --------------------------------------------------------------------------------------------------
def test_student_tokens_use_sampled_ids_with_offsets_into_completion_text():
    from flash.engine.worker.opd import student_tokens_with_offsets

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            # id 1 -> 'h', id 2 -> 'i', id 3 -> a special token that decodes to nothing.
            m = {1: "h", 2: "i", 3: ""}
            return "".join(m[i] for i in ids)

    ids, toks = student_tokens_with_offsets(_Tok(), [1, 2, 3], "hi")
    assert ids == [1, 2, 3]  # SAMPLED ids preserved verbatim — no lossy re-tokenization
    assert (toks[0].start, toks[0].end) == (0, 1)  # 'h'
    assert (toks[1].start, toks[1].end) == (1, 2)  # 'i'
    assert (toks[2].start, toks[2].end) == (2, 2)  # special token -> zero-width span (excluded)


def test_train_one_full_loop_forwards_sampled_ids_and_ignores_zero_width_eos():
    """Exercise the PRODUCTION caller _train_one end-to-end (the direct-call unit test above can't
    catch a broken call site). The completion ends in a zero-width eos, so this also pins the
    coverage denominator: 2 alignable tokens fully covered -> 100%, not 2/3."""
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    completion_ids = [2, 3, 5]  # 2->'h', 3->'i', 5->eos (in-vocab id, decodes to '')

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            m = {2: "h", 3: "i", 5: ""}
            return "".join(m[int(x)] for x in ids)

    class _Teacher:
        def score(self, prompt, completion):  # one teacher token spanning all of "hi"
            return [TeacherToken(text="hi", logprob=-1.0, start=0, end=2)]

    class _GenLM(_TinyLM):
        def __init__(self, torch, prompt_len, completion_ids, V):
            super().__init__(torch, T=prompt_len + len(completion_ids), V=V)
            self.config = SimpleNamespace(use_cache=True)
            self._completion = torch.tensor([completion_ids])

        def eval(self):
            return self

        def train(self):
            return self

        def generate(self, prompt_tensor, **cfg):
            return torch.cat([prompt_tensor, self._completion], dim=1)

    model = _GenLM(torch, prompt_len=1, completion_ids=completion_ids, V=8)
    loss = opd_mod._train_one(
        model=model,
        tok=_Tok(),
        teacher=_Teacher(),
        device="cpu",
        prompt_ids=[1],
        prompt_tensor=torch.tensor([[1]]),
        prompt_messages=[{"role": "user", "content": "say hi"}],
        gen_cfg={},
        knobs={"kl_coef": 1.0},
        torch=torch,
    )
    assert loss is not None
    assert loss.requires_grad
    loss.backward()  # the sampled ids reached gkd_loss and produce a real gradient
    assert model.w.grad is not None
    assert model.w.grad.abs().sum() > 0
    # eos is zero-width and joins no group; coverage is over the 2 alignable tokens -> 100%.
    assert opd_mod._train_one.last_coverage == 1.0
    assert opd_mod._train_one.last_gen_tokens == 3


# --------------------------------------------------------------------------------------------------
# teacher client (mocked HTTP)
# --------------------------------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _mock_urlopen(monkeypatch, payload, capture=None):
    import flash.engine.worker.teacher as tm

    def fake_urlopen(req, timeout=None):
        if capture is not None:
            capture["url"] = req.full_url
            capture["body"] = json.loads(req.data.decode())
        return _FakeResp(payload)

    monkeypatch.setattr(tm.urllib.request, "urlopen", fake_urlopen)


def test_teacher_score_returns_completion_region_with_rebased_offsets_and_logprobs(monkeypatch):
    # prompt "P: " (len 3) + completion "hi" ; teacher tokens: "P", ":", " ", "hi".
    payload = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["P", ":", " ", "hi"],
                    "token_logprobs": [0.0, -1.0, -2.0, -0.5],
                    "text_offset": [0, 1, 2, 3],
                }
            }
        ]
    }
    capture = {}
    _mock_urlopen(monkeypatch, payload, capture)
    client = TeacherClient("k", "https://api.example/v1", "glm")
    toks = client.score("P: ", "hi")
    # only the completion token survives; offset rebased to the completion (start 0).
    assert len(toks) == 1
    assert toks[0].text == "hi"
    assert toks[0].start == 0
    assert toks[0].logprob == -0.5  # the realized-token logprob gkd consumes
    # scoring must not pay for generation, and asks for the minimal logprobs that return token_logprobs.
    assert capture["body"]["max_tokens"] == 0
    assert capture["body"]["echo"] is True
    assert capture["body"]["logprobs"] == 1


def test_teacher_score_raises_on_malformed_response(monkeypatch):
    from flash.engine.worker.teacher import TeacherError

    _mock_urlopen(monkeypatch, {"choices": [{"logprobs": {}}]})
    client = TeacherClient("k", "https://api.example/v1", "glm")
    with pytest.raises(TeacherError):
        client.score("", "hi")


def test_teacher_client_requires_key():
    from flash.engine.worker.teacher import TeacherError

    with pytest.raises(TeacherError):
        TeacherClient("", "https://api.example/v1", "glm")


# --------------------------------------------------------------------------------------------------
# spec + cost plumbing
# --------------------------------------------------------------------------------------------------
def test_opd_spec_json_round_trip():
    from flash.schema import spec_from_dict
    from flash.spec import JobSpec

    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "opd",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {
                "steps": 25,
                "teacher_model": "accounts/fireworks/models/glm-5p1",
                "hf_repo": "owner/runs",
            },
        },
        run_id="x",
    )
    restored = JobSpec.from_json(spec.to_json())
    assert restored == spec
    assert restored.phase == "opd"
    assert restored.train.teacher_model == "accounts/fireworks/models/glm-5p1"
    # FIREWORKS_API_KEY is auto-declared a required secret for opd.
    assert "FIREWORKS_API_KEY" in restored.environment.secrets


def test_opd_cost_is_step_priced_and_bills_teacher_tokens():
    from flash.cost.spec import estimate_for_spec, spec_steps
    from flash.schema import spec_from_dict

    # No [train].max_examples set — opd must NOT fall into the SFT example-count path (which
    # would raise); it is step-driven like GRPO.
    spec = spec_from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "opd",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {"steps": 30, "hf_repo": "owner/runs"},
        },
        run_id="x",
    )
    assert spec_steps(spec) == 30
    est = estimate_for_spec(spec)
    assert est.method == "opd"
    assert est.teacher_api_usd > 0.0  # external teacher token spend is itemized
    assert est.total_usd >= est.teacher_api_usd
    assert "opd step" in " ".join(est.notes)


# --------------------------------------------------------------------------------------------------
# loss math (needs torch)
# --------------------------------------------------------------------------------------------------
class _TinyLM:
    """Minimal stand-in for a causal LM: per-position learnable logits, ignores the input ids."""

    def __init__(self, torch, T, V):
        self.w = torch.zeros(T, V, requires_grad=True)

    def __call__(self, input_ids):
        T = input_ids.shape[1]
        return SimpleNamespace(logits=self.w[:T].unsqueeze(0))

    def parameters(self):
        return [self.w]


def test_gkd_loss_backpropagates_over_grouped_spans():
    torch = pytest.importorskip("torch")
    from flash.engine.worker.opd import gkd_loss

    V = 8
    prompt_ids = [1]
    student_ids = [2, 3]  # 2 completion tokens
    model = _TinyLM(torch, T=len(prompt_ids) + len(student_ids), V=V)
    # One group covering both completion tokens (as when the teacher tokenizes them as one span).
    groups = [([0, 1], -1.5)]
    loss = gkd_loss(model, prompt_ids, student_ids, groups, device="cpu", kl_coef=1.0)
    assert loss is not None
    assert loss.requires_grad
    loss.backward()
    # logits at index P+j-1 predict completion token j: index 0 -> tok 0, index 1 -> tok 1.
    assert model.w.grad[0].abs().sum() > 0
    assert model.w.grad[1].abs().sum() > 0


def test_gkd_loss_none_without_groups_or_tokens():
    torch = pytest.importorskip("torch")
    from flash.engine.worker.opd import gkd_loss

    model = _TinyLM(torch, T=3, V=4)
    assert gkd_loss(model, [1], [2, 3], [], device="cpu") is None
    assert gkd_loss(model, [1], [], [([0], -1.0)], device="cpu") is None


def test_gkd_loss_coefficient_tracks_student_minus_teacher_logprob():
    # The per-span coefficient is (student_logsum.detach() - teacher_logsum)/|span|; a more
    # confident teacher (lower/more-negative teacher_logsum) makes the coefficient larger.
    torch = pytest.importorskip("torch")
    from flash.engine.worker.opd import gkd_loss

    V = 8
    prompt_ids = [1]
    student_ids = [2]
    model = _TinyLM(torch, T=2, V=V)  # uniform logits -> student logprob = -log V per token
    hi = gkd_loss(model, prompt_ids, student_ids, [([0], -5.0)], device="cpu", kl_coef=1.0)
    lo = gkd_loss(model, prompt_ids, student_ids, [([0], -0.5)], device="cpu", kl_coef=1.0)
    # loss = coeff * student_logprob, student_logprob < 0, and coeff = (s_det - teacher)/1.
    # teacher=-5.0 -> larger coeff -> more-negative loss than teacher=-0.5.
    assert float(hi.detach()) < float(lo.detach())
