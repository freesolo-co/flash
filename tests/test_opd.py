"""On-policy distillation: alignment strategies, teacher client, spec/cost plumbing, loss math.

All CPU-only. The loss-math tests need torch and are skipped where it is unavailable (they run in
CI, which has the training stack).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from flash.engine.worker.teacher import TeacherClient, strip_reasoning
from flash.engine.worker.tokenizer_align import (
    StudentToken,
    TeacherToken,
    align_targets,
    coverage,
    uld_targets,
)


# --------------------------------------------------------------------------------------------------
# tokenizer alignment (the 3 cross-tokenizer strategies)
# --------------------------------------------------------------------------------------------------
def _student(spans):
    return [StudentToken(token_id=i, start=a, end=b) for i, (a, b) in enumerate(spans)]


def test_align_projects_teacher_topk_onto_student_vocab():
    # Student tokens at char 0 and 5; teacher tokens start at the same boundaries -> both align.
    student = _student([(0, 5), (5, 10)])
    teacher = [
        TeacherToken("hello", -0.1, (("hello", -0.1), (" hi", -1.6)), 0, 5),
        TeacherToken("world", -0.2, (("world", -0.2),), 5, 10),
    ]
    vocab = {"hello": 100, " hi": 101, "world": 200}
    tgts = align_targets(student, teacher, lambda s: vocab.get(s))
    assert coverage(tgts) == 1.0
    # position 0: two candidates, softmax-weighted and renormalized over the mapped ids.
    assert set(tgts[0]) == {100, 101}
    assert tgts[0][100] > tgts[0][101]  # 'hello' had the higher logprob
    assert abs(sum(tgts[0].values()) - 1.0) < 1e-9
    assert tgts[1] == {200: 1.0}


def test_align_masks_positions_without_a_coincident_teacher_boundary():
    # Student boundary at char 3 has no teacher token starting there -> masked (None).
    student = _student([(0, 3), (3, 7)])
    teacher = [TeacherToken("hello", -0.1, (("hello", -0.1),), 0, 5)]
    tgts = align_targets(student, teacher, lambda s: {"hello": 1}.get(s))
    assert tgts[0] is not None  # boundary 0 aligns
    assert tgts[1] is None  # boundary 3 does not
    assert coverage(tgts) == 0.5


def test_align_masks_when_no_candidate_maps_into_student_vocab():
    student = _student([(0, 5)])
    teacher = [TeacherToken("xx", -0.1, (("xx", -0.1),), 0, 5)]
    tgts = align_targets(student, teacher, lambda s: None)  # nothing maps
    assert tgts == [None]


def test_uld_returns_sorted_normalized_teacher_probs_no_identity():
    student = _student([(0, 5)])
    teacher = [TeacherToken("w", -0.2, (("a", -2.0), ("b", -0.1), ("c", -3.0)), 0, 5)]
    tgts = uld_targets(student, teacher)
    assert len(tgts) == 1
    vec = tgts[0]
    assert vec == sorted(vec, reverse=True)  # descending
    assert abs(sum(vec) - 1.0) < 1e-9  # normalized over the top-k


def test_kd_temperature_flattens_the_target_distribution():
    student = _student([(0, 5)])
    teacher = [TeacherToken("w", -0.1, (("a", -0.1), ("b", -2.0)), 0, 5)]
    sharp = align_targets(student, teacher, lambda s: {"a": 1, "b": 2}.get(s), kd_temperature=1.0)
    flat = align_targets(student, teacher, lambda s: {"a": 1, "b": 2}.get(s), kd_temperature=5.0)
    # Higher temperature -> the low-prob candidate gets relatively more mass.
    assert flat[0][2] > sharp[0][2]


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


def test_teacher_score_returns_completion_region_with_rebased_offsets(monkeypatch):
    # prompt "P: " (len 3) + completion "hi" ; teacher tokens: "P", ":", " ", "hi".
    payload = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["P", ":", " ", "hi"],
                    "token_logprobs": [0.0, -1.0, -2.0, -0.5],
                    "text_offset": [0, 1, 2, 3],
                    "top_logprobs": [None, None, None, {"hi": -0.5, "hello": -1.2}],
                }
            }
        ]
    }
    capture = {}
    _mock_urlopen(monkeypatch, payload, capture)
    client = TeacherClient("k", "https://api.example/v1", "glm", top_logprobs=5)
    toks = client.score("P: ", "hi")
    # only the completion token survives; offset rebased to the completion (start 0).
    assert len(toks) == 1
    assert toks[0].text == "hi"
    assert toks[0].start == 0
    assert dict(toks[0].top) == {"hi": -0.5, "hello": -1.2}
    # scoring must not pay for generation.
    assert capture["body"]["max_tokens"] == 0
    assert capture["body"]["echo"] is True


def test_teacher_score_injects_realized_token_when_missing_from_topk(monkeypatch):
    payload = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["", "hi"],
                    "token_logprobs": [0.0, -0.5],
                    "text_offset": [0, 0],  # prompt is empty here
                    "top_logprobs": [None, {"other": -0.1}],  # realized 'hi' absent
                }
            }
        ]
    }
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient("k", "https://api.example/v1", "glm")
    toks = client.score("", "hi")
    assert any(s == "hi" for s, _ in toks[-1].top), "realized token must always be represented"


def test_teacher_score_clamps_logprobs_to_fireworks_cap(monkeypatch):
    # Fireworks' /completions echo endpoint rejects logprobs > 5; the client must clamp.
    payload = {
        "choices": [
            {"logprobs": {"tokens": ["hi"], "token_logprobs": [-0.5], "text_offset": [0]}}
        ]
    }
    capture = {}
    _mock_urlopen(monkeypatch, payload, capture)
    client = TeacherClient("k", "https://api.example/v1", "glm", top_logprobs=20)
    client.score("", "hi")
    assert capture["body"]["logprobs"] == 5


def test_teacher_generate_strips_reasoning(monkeypatch):
    payload = {"choices": [{"message": {"content": "think think </think>  Final."}}]}
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient("k", "https://api.example/v1", "glm")
    assert client.generate([{"role": "user", "content": "x"}], max_tokens=8) == "Final."


def test_teacher_client_requires_key():
    from flash.engine.worker.teacher import TeacherError

    with pytest.raises(TeacherError):
        TeacherClient("", "https://api.example/v1", "glm")


def test_strip_reasoning_variants():
    assert strip_reasoning("no markers here") == "no markers here"
    assert strip_reasoning("a</think>b</think>c") == "c"  # last marker wins
    assert strip_reasoning("") == ""


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
                "tokenizer_alignment": "uld",
                "teacher_top_logprobs": 12,
                "hf_repo": "owner/runs",
            },
        },
        run_id="x",
    )
    restored = JobSpec.from_json(spec.to_json())
    assert restored == spec
    assert restored.phase == "opd"
    assert restored.train.tokenizer_alignment == "uld"
    assert restored.train.teacher_top_logprobs == 12


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


def test_align_loss_is_differentiable_only_at_aligned_positions():
    torch = pytest.importorskip("torch")
    from flash.engine.worker.opd import align_loss

    V = 8
    prompt_ids = [1, 2]
    student_ids = [3, 4, 5]  # 3 completion tokens
    model = _TinyLM(torch, T=len(prompt_ids) + len(student_ids), V=V)
    # target only at completion position 0 (predicted by logits at index P-1 = 1).
    targets = [{3: 1.0}, None, None]
    loss = align_loss(model, prompt_ids, student_ids, targets, device="cpu")
    assert loss is not None
    assert loss.requires_grad
    loss.backward()
    grad = model.w.grad
    assert grad[1].abs().sum() > 0  # aligned position got a gradient
    assert grad[3].abs().sum() == 0  # a non-target position did not


def test_align_loss_none_when_all_masked():
    torch = pytest.importorskip("torch")
    from flash.engine.worker.opd import align_loss

    model = _TinyLM(torch, T=3, V=4)
    assert align_loss(model, [1], [2, 3], [None, None], device="cpu") is None


def test_uld_loss_runs_and_backpropagates():
    torch = pytest.importorskip("torch")
    from flash.engine.worker.opd import uld_loss

    V = 8
    prompt_ids = [1]
    student_ids = [2, 3]
    model = _TinyLM(torch, T=3, V=V)
    tgts = [[0.7, 0.3], None]
    loss = uld_loss(model, prompt_ids, student_ids, tgts, device="cpu", top_k=2)
    assert loss is not None
    assert loss.requires_grad
    loss.backward()
    assert model.w.grad.abs().sum() > 0


def test_seqkd_loss_matches_cross_entropy_on_completion():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from flash.engine.worker.opd import seqkd_loss

    V = 6
    prompt_ids = [1, 2]
    target_ids = [3, 4]
    model = _TinyLM(torch, T=len(prompt_ids) + len(target_ids), V=V)
    loss = seqkd_loss(model, prompt_ids, target_ids, device="cpu")
    assert loss is not None
    assert loss.requires_grad
    # Reference: CE over the two completion positions (logits at indices 1 and 2 predict 3 and 4).
    logits = model.w[:-1]
    labels = torch.tensor([-100, 3, 4])
    ref = F.cross_entropy(logits, labels, ignore_index=-100)
    assert torch.allclose(loss, ref, atol=1e-5)
