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
    assert groupwise_coverage(groups, student) == 1.0


def test_gkd_span_grows_across_disagreement_and_covers_every_token():
    # Teacher emits one token where the student emits two; no shared interior boundary at char 3,
    # so both student tokens fall in a single group summing the (single) teacher logprob.
    student = _student([(0, 3), (3, 6)])
    teacher = _teacher([(0, 6)])
    groups = groupwise_alignment(student, teacher)
    assert len(groups) == 1
    assert groups[0][0] == [0, 1]  # both student tokens grouped together
    assert groups[0][1] == -1.0
    assert groupwise_coverage(groups, student) == 1.0  # NO masking


def test_gkd_partial_agreement_splits_at_shared_boundaries_only():
    # Shared boundaries at chars 0 and 2; the student's interior boundary at 4 is not shared.
    student = _student([(0, 2), (2, 4), (4, 6)])
    teacher = _teacher([(0, 2), (2, 6)])
    groups = groupwise_alignment(student, teacher)
    assert [s_idx for s_idx, _ in groups] == [[0], [1, 2]]
    assert groupwise_coverage(groups, student) == 1.0


def test_gkd_merges_leading_student_only_span_so_no_token_is_dropped():
    # Student starts at char 0 but the first teacher token starts at char 2, so [0,2) is a
    # student-only span. Those tokens must NOT be dropped — they merge into the first teacher-bearing
    # group (coverage stays 100%).
    student = _student([(0, 1), (1, 2), (2, 5)])
    teacher = _teacher([(2, 5)])
    groups = groupwise_alignment(student, teacher)
    assert len(groups) == 1
    assert groups[0][0] == [0, 1, 2]  # all three student tokens covered
    assert groupwise_coverage(groups, student) == 1.0


def test_gkd_empty_inputs_yield_no_groups():
    assert groupwise_alignment([], _teacher([(0, 1)])) == []
    assert groupwise_alignment(_student([(0, 1)]), []) == []
    assert groupwise_coverage([], []) == 0.0


def test_gkd_coverage_never_exceeds_100pct_with_in_span_zero_width_token():
    # A zero-width student token (partial-byte fragment / mid-completion special) at char 2 rides
    # ALONG inside the group (so the span's student logprob sum stays complete) but must NOT be
    # counted as covered: 2 alignable tokens, both grouped -> exactly 100%, not 150% (the real-run
    # bug the probe surfaced).
    student = _student([(0, 2), (2, 2), (2, 5)])  # middle token is zero-width
    teacher = _teacher([(0, 5)])
    groups = groupwise_alignment(student, teacher)
    assert groups[0][0] == [0, 1, 2]  # all three ride in the group (logprob sum stays whole)
    assert groupwise_coverage(groups, student) == 1.0


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


def test_trim_trailing_stop_drops_delimiter_from_ids_and_text():
    from flash.engine.worker.opd import _trim_trailing_stop

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            m = {1: "A", 2: "n", 3: "s", 4: "</", 5: "answer>"}
            return "".join(m[int(i)] for i in ids)

    # completion "Ans</answer>"; stop "</answer>" spans ids 4,5 -> keep "Ans" / ids [1,2,3].
    ids, text = _trim_trailing_stop(_Tok(), [1, 2, 3, 4, 5], "Ans</answer>", ["</answer>"])
    assert text == "Ans"  # teacher scores the answer only, not the delimiter
    assert ids == [1, 2, 3]  # ids trimmed in lockstep with the text (no gkd_loss/count desync)
    # no trailing stop -> unchanged (ids normalized to a list)
    assert _trim_trailing_stop(_Tok(), [1, 2, 3], "Ans", ["</answer>"]) == ([1, 2, 3], "Ans")


def test_opd_vram_sizing_uses_completion_budget_not_sft_default():
    # OPD generates on-policy (loss forward runs model(prompt+completion)), so allocator sizing must
    # use the prompt+completion budget, not the SFT 1024 default — else a raised max_tokens OOMs an
    # under-sized GPU.
    from flash.engine.vram import opd_rollout_seq_len

    assert opd_rollout_seq_len(0, None, False) == 1536  # 1024 prompt + 512 completion default
    assert opd_rollout_seq_len(0, 8192, False) == 9216  # raised max_tokens sizes up (was 1024)
    assert opd_rollout_seq_len(4096, 8192, False) == 4096  # explicit max_length pins the sequence


def test_opd_rejects_unpriced_teacher_model_but_accepts_priced():
    from flash.schema import ConfigError, spec_from_dict

    def _spec(teacher):
        return spec_from_dict(
            {
                "model": "Qwen/Qwen3.5-4B",
                "algorithm": "opd",
                "environment": {"id": "github:owner/repo@main:env/environment.py"},
                "train": {"steps": 5, "hf_repo": "owner/runs", "teacher_model": teacher},
            },
            run_id="x",
        )

    _spec("accounts/fireworks/models/glm-5p2")  # priced -> ok
    with pytest.raises(ConfigError):
        _spec("accounts/fireworks/models/mystery-9000")  # unpriced override -> reject at parse


def test_opd_rejects_prompt_budget_at_parse_time_before_provisioning():
    """max_length <= max_tokens leaves no prompt budget; opd must reject it at spec-parse time
    (before a paid worker is provisioned), not only inside run_opd after GPU setup."""
    from flash.schema import ConfigError, spec_from_dict

    def _spec(train_extra):
        return spec_from_dict(
            {
                "model": "Qwen/Qwen3.5-4B",
                "algorithm": "opd",
                "environment": {"id": "github:owner/repo@main:env/environment.py"},
                "train": {"steps": 5, "hf_repo": "owner/runs", **train_extra},
            },
            run_id="x",
        )

    # max_length leaves room after an explicit max_tokens -> ok.
    _spec({"max_length": 2048, "max_tokens": 512})
    # max_length <= max_tokens -> no prompt budget -> reject at parse.
    with pytest.raises(ConfigError, match="prompt budget"):
        _spec({"max_length": 400, "max_tokens": 512})
    # max_tokens omitted -> resolves to the opd recipe default (512); max_length below it -> reject.
    with pytest.raises(ConfigError, match="prompt budget"):
        _spec({"max_length": 256})


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
        knobs={"kl_coef": 1.0, "stop_sequences": ()},
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


def test_all_skip_step_emits_stall_refresh_opd_step_heartbeat(monkeypatch):
    """Regression (codex[bot], opd.py:380-381): when EVERY sample in a step skips (empty completion
    / no teacher signal, or an over-budget re-render), the per-sample SUCCESS ping is never reached.
    Without a skip-path ping the step would emit only liveness heartbeats — which the pollers ignore
    — so a prolonged all-skip stretch on a later step could be reaped as stalled. Assert the skip path
    emits a NON-liveness opd_step heartbeat, and that it reports step==opt_steps (==0 while the first
    step is still accumulating) so it keeps the WIDE setup grace rather than flipping to the tight
    training window (opd_step is step-gated in the poller)."""
    torch = pytest.importorskip("torch")
    from flash.engine.worker import opd as opd_mod

    beats: list[tuple[str, dict]] = []

    class _Tok:
        pad_token = "<pad>"
        eos_token = "<eos>"
        pad_token_id = 0

        def apply_chat_template(self, messages, **kw):
            return "PROMPT"

        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=[1, 2])  # 2 tokens, well within budget

    class _Model(_TinyLM):
        def __init__(self):
            super().__init__(torch, T=4, V=8)
            self.config = SimpleNamespace(use_cache=False)

    env = SimpleNamespace(
        dataset=lambda: [{"q": "a"}, {"q": "b"}],
        prompt_messages=lambda ex: [{"role": "user", "content": ex["q"]}],
    )
    fake_w = SimpleNamespace(
        require_active_env=lambda: env,
        JOB_SPEC=SimpleNamespace(
            train=SimpleNamespace(init_from_adapter=""),
            model="fake/model",
            gpu=SimpleNamespace(type=None),
        ),
        THINKING=False,
        SEED=0,
        heartbeat=lambda stage, **kw: beats.append((stage, kw)),
        prefetch_model=lambda mid: 0.0,
        hf_resume_checkpoint=lambda: "",
        publish_deployable_checkpoint=lambda *a, **k: None,
        hf_upload_folder=lambda *a, **k: None,
        write_train_meta=lambda **k: None,
    )
    monkeypatch.setattr(opd_mod, "_w", fake_w)
    # Deterministic knobs: 1 step / 1 prompt / group 1 -> a single sample, forced to skip.
    monkeypatch.setattr(
        opd_mod,
        "_resolve_opd_knobs",
        lambda: {
            "teacher_model": "accounts/fireworks/models/glm-5p2",
            "teacher_base_url": "http://teacher.invalid",
            "steps": 1,
            "learning_rate": 1e-4,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_completion": 8,
            "prompts_per_step": 1,
            "group_size": 1,
            "kl_coef": 1.0,
            "save_every": 0,
            "max_length": 0,
            "stop_sequences": (),
        },
    )
    monkeypatch.setattr(opd_mod, "_student_model", lambda *a, **k: _Model())
    monkeypatch.setattr(opd_mod, "wait_for_gpu", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "setup_perf_backends", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "optimal_attn_impl", lambda *a, **k: None)
    monkeypatch.setattr(opd_mod, "grad_checkpointing_on", lambda *a, **k: False)
    monkeypatch.setattr(opd_mod, "gpu_diagnostics", lambda *a, **k: {})
    monkeypatch.setattr(opd_mod, "_train_one", lambda **k: None)  # EVERY sample skips

    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tok()
    )
    import flash.engine.worker.teacher as tmod

    monkeypatch.setattr(tmod, "TeacherClient", lambda *a, **k: object())
    monkeypatch.setenv("FIREWORKS_API_KEY", "unit-test-teacher-key")

    # An all-skip run lands no optimizer step, so run_opd raises its no-trained-step guard AFTER the
    # loop; we assert on the heartbeats captured before that raise.
    with pytest.raises(RuntimeError, match="no trained step"):
        opd_mod.run_opd()

    # Per-sample opd_step pings carry samples_done (liveness pings never do), so this isolates the
    # skip-path progress ping from any liveness heartbeat.
    per_sample = [kw for (stage, kw) in beats if stage == "opd_step" and "samples_done" in kw]
    assert per_sample, "an all-skip step emitted no per-sample opd_step ping -> stall clock unrefreshed"
    assert all(kw.get("step") == 0 for kw in per_sample), (
        "skip-path ping must report opt_steps (0 during the first, still-accumulating step) so the "
        "poller keeps the wide setup grace instead of the tight training window"
    )


def test_student_model_accepts_vl_warmstart_without_raising(monkeypatch):
    """opd VL warm-start (Qwen3.5/3.6) must build a fresh LoRA on the SFT-merged base — parity with
    GRPO — instead of raising. _init_adapter_model returns (merged_dir, fresh_lora) and records the
    VL merge; _student_model must LOAD the merged dir and wrap it in a PeftModel (run_opd then
    recombines SFT⊕opd at publish)."""
    pytest.importorskip("torch")  # AutoModelForCausalLM (patched below) needs torch present
    pytest.importorskip("transformers")
    import sys
    import types

    from flash.engine.worker import opd as opd_mod

    sentinel = object()
    fake_lora = object()
    seen = {}

    class _Base:
        def to(self, device):
            seen["device"] = device
            return self

    def _fake_init_adapter_model(model_id):
        opd_mod._w._VL_WARMSTART_SFT_DIR = "/tmp/fake_sft_dir"  # the VL merge path records this
        return "/tmp/merged_vl_dir", fake_lora  # merged base dir + FRESH LoRA config

    monkeypatch.setattr(opd_mod._w, "_VL_WARMSTART_SFT_DIR", None, raising=False)
    monkeypatch.setattr(opd_mod._w, "_init_adapter_model", _fake_init_adapter_model, raising=False)

    import transformers

    def _fake_from_pretrained(path, **kw):
        seen["path"] = path
        return _Base()

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(_fake_from_pretrained)
    )
    # _student_model does `from peft import get_peft_model`; peft is a worker-only dep, so inject a
    # stub module (works with or without a real peft install — hermetic local + CI).
    fake_peft = types.ModuleType("peft")
    fake_peft.get_peft_model = lambda base, cfg: (sentinel, base, cfg)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    out = opd_mod._student_model("Qwen/Qwen3.5-4B", {"dtype": "bf16"}, "cpu")
    assert out[0] is sentinel  # did NOT raise; wrapped the merged base in a PeftModel
    assert out[2] is fake_lora  # used the fresh LoRA from _init_adapter_model
    assert seen["path"] == "/tmp/merged_vl_dir"  # loaded the MERGED dir, not the raw base id


def test_publish_opd_deployable_recombines_for_vl_and_cleans_up(tmp_path, monkeypatch):
    """VL warm-start: the trained adapter is SFT-less, so publish must deploy the RECOMBINED SFT⊕opd
    adapter (as the served default AND the step checkpoint) and clean up the temp recombine dir."""
    from flash.engine.worker import opd as opd_mod

    calls = {"upload": [], "publish": []}
    recomb = tmp_path / "recomb"
    recomb.mkdir()
    monkeypatch.setattr(
        opd_mod._w, "recombined_warmstart_adapter_dir", lambda d: str(recomb), raising=False
    )
    monkeypatch.setattr(
        opd_mod._w,
        "hf_upload_folder",
        lambda d, sub, required=False: calls["upload"].append((d, sub)),
        raising=False,
    )
    monkeypatch.setattr(
        opd_mod._w,
        "publish_deployable_checkpoint",
        lambda d, step: calls["publish"].append((d, step)),
        raising=False,
    )

    opd_mod._publish_opd_deployable(str(tmp_path / "adapter"), 42, as_default=True)
    assert calls["upload"] == [(str(recomb), "adapter")]  # served default = RECOMBINED, not raw
    assert calls["publish"] == [(str(recomb), 42)]  # step checkpoint = RECOMBINED
    assert not recomb.exists()  # temp recombine dir cleaned up


def test_publish_opd_deployable_noop_recombine_deploys_adapter_dir(tmp_path, monkeypatch):
    """Non-VL / fresh runs: recombine is a no-op (returns None); the trained adapter_dir already
    carries the SFT, so it is deployed as-is. as_default=False publishes only the step checkpoint."""
    from flash.engine.worker import opd as opd_mod

    calls = {"upload": [], "publish": []}
    monkeypatch.setattr(
        opd_mod._w, "recombined_warmstart_adapter_dir", lambda d: None, raising=False
    )
    monkeypatch.setattr(
        opd_mod._w,
        "hf_upload_folder",
        lambda d, sub, required=False: calls["upload"].append((d, sub)),
        raising=False,
    )
    monkeypatch.setattr(
        opd_mod._w,
        "publish_deployable_checkpoint",
        lambda d, step: calls["publish"].append((d, step)),
        raising=False,
    )

    adir = str(tmp_path / "adapter")
    opd_mod._publish_opd_deployable(adir, 7, as_default=False)
    assert calls["upload"] == []  # as_default=False -> no served-default upload
    assert calls["publish"] == [(adir, 7)]  # no-op recombine -> deploys adapter_dir as-is


def test_publish_opd_deployable_best_effort_survives_recombine_failure(tmp_path, monkeypatch):
    """Per-step publish is best-effort: a recombine failure (e.g. an evicted SFT dir) is swallowed so
    training continues; the strict finalize path re-raises (it must never ship an SFT-less default)."""
    from flash.engine.worker import opd as opd_mod

    def _boom(d):
        raise RuntimeError("SFT dir evicted")

    monkeypatch.setattr(opd_mod._w, "recombined_warmstart_adapter_dir", _boom, raising=False)
    monkeypatch.setattr(
        opd_mod._w, "publish_deployable_checkpoint", lambda d, s: None, raising=False
    )
    monkeypatch.setattr(
        opd_mod._w, "hf_upload_folder", lambda d, sub, required=False: None, raising=False
    )

    # best_effort=True (per-step): swallowed, training continues (no raise).
    opd_mod._publish_opd_deployable(str(tmp_path / "a"), 20, as_default=False, best_effort=True)
    # best_effort=False (finalize): fatal — must not ship an SFT-less served default.
    with pytest.raises(RuntimeError, match="SFT dir evicted"):
        opd_mod._publish_opd_deployable(str(tmp_path / "a"), 100, as_default=True)


def test_opd_vram_reserves_dense_logits_unlike_fused_sft():
    """opd's gkd loss materializes dense logits (no fused CE), so its VRAM estimate must reserve the
    logits a >=3B SFT job fuses away — else a long-completion opd run is sized for a card that OOMs."""
    from flash.engine.vram import estimate_vram_gb

    kw = {"seq_len": 9216, "max_tokens": 8192, "vocab": 248_320, "lora_rank": 16}
    sft = estimate_vram_gb(4.0, "sft", "bf16", **kw)  # >=3B fuses CE -> 0 logits budgeted
    opd = estimate_vram_gb(4.0, "opd", "bf16", **kw)  # dense logits reserved
    assert opd > sft + 10  # ~12.7 GB of dense logits for opd vs 0 for fused SFT


def test_opd_teacher_rate_matches_fireworks_glm5p2_input_price():
    """glm-5p2 (and the omitted-teacher default) price at Fireworks' $1.40/M input, not the old $0.90
    — opd echo-scoring bills input tokens from the submit-time quote."""
    from flash.cost.facts import teacher_price_per_1m

    assert teacher_price_per_1m("accounts/fireworks/models/glm-5p2")[0] == 1.40
    assert teacher_price_per_1m("")[0] == 1.40  # omitted teacher -> representative default rate


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


def test_teacher_score_keeps_boundary_crossing_token_clamped_to_completion(monkeypatch):
    # Prompt ends in whitespace ("P: ", plen=3); the teacher emits a leading-space merge token
    # " hi" that starts at char 2 (inside the prompt) and ends at 5 (inside the completion). Rather
    # than DROP it — which for a one-token completion would leave zero teacher tokens and skip the
    # sample — score() KEEPS it with its completion span clamped to [0, end-plen) so the first
    # completion token still carries a teacher logprob. Only tokens ENTIRELY in the prompt (end<=plen)
    # are dropped.
    payload = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["P", ":", " hi", "!"],
                    "token_logprobs": [0.0, -1.0, -0.5, -0.2],
                    "text_offset": [0, 1, 2, 5],
                }
            }
        ]
    }
    _mock_urlopen(monkeypatch, payload)
    client = TeacherClient("k", "https://api.example/v1", "glm")
    toks = client.score("P: ", "hi!")
    # "P" and ":" lie entirely in the prompt (end <= plen=3) -> dropped. The boundary-crossing " hi"
    # is kept, clamped to completion span [0, 2); "!" keeps [2, 3). "hi!" is fully covered.
    assert [t.text for t in toks] == [" hi", "!"]
    assert (toks[0].start, toks[0].end) == (0, 2)  # max(0, 2-3)=0 ; 5-3=2
    assert (toks[1].start, toks[1].end) == (2, 3)  # 5-3=2 ; 6-3=3
    assert toks[0].logprob == -0.5  # the merged token's realized logprob is preserved


def test_teacher_score_raises_on_malformed_response(monkeypatch):
    from flash.engine.worker.teacher import TeacherError

    _mock_urlopen(monkeypatch, {"choices": [{"logprobs": {}}]})
    client = TeacherClient("k", "https://api.example/v1", "glm")
    with pytest.raises(TeacherError) as ei:
        client.score("", "hi")
    assert ei.value.permanent is True  # malformed response -> abort the run, not skip-and-burn


def test_teacher_4xx_is_permanent_but_5xx_is_transient(monkeypatch):
    import urllib.error

    import flash.engine.worker.teacher as tm
    from flash.engine.worker.teacher import TeacherError

    def raise_http(code):
        def fake_urlopen(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, code, f"HTTP {code}", {}, None)

        monkeypatch.setattr(tm.urllib.request, "urlopen", fake_urlopen)

    client = TeacherClient("k", "https://api.example/v1", "glm", max_retries=1)
    # 401 (bad key) is permanent -> raised immediately so the worker aborts, not burns every step.
    raise_http(401)
    with pytest.raises(TeacherError) as ei:
        client.score("P", "hi")
    assert ei.value.permanent is True
    # 503 is transient -> retries exhaust to a non-permanent error (a skipped sample, run continues).
    raise_http(503)
    with pytest.raises(TeacherError) as ei:
        client.score("P", "hi")
    assert ei.value.permanent is False


def test_gkd_loss_skips_empty_student_group_without_crashing():
    # A group with an empty student-index list (a teacher-only span) must be skipped, not divide by
    # zero in the per-span coefficient (len(s_idx) == 0).
    torch = pytest.importorskip("torch")
    from flash.engine.worker.opd import gkd_loss

    model = _TinyLM(torch, T=2, V=8)
    loss = gkd_loss(model, [1], [2], [([], -1.0), ([0], -2.0)], device="cpu", kl_coef=1.0)
    assert loss is not None  # the empty group is ignored; the real group still trains
    loss.backward()
    assert model.w.grad[0].abs().sum() > 0


def test_groupwise_alignment_emits_no_empty_student_group():
    # Teacher covers [0,2) but the student's first token starts at char 2 (teacher-only leading
    # span). No group may have an empty student-index list.
    student = _student([(2, 3), (3, 5)])
    teacher = _teacher([(0, 2), (2, 5)])
    groups = groupwise_alignment(student, teacher)
    assert all(s_idx for s_idx, _ in groups)  # every group has >= 1 student token
    assert [s_idx for s_idx, _ in groups] == [[0, 1]]


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
