"""CPU/offline tests for the math grader and the HF-backed `math` environment."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests._examples import load_example_module

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_extract_boxed_balanced():
    mg = load_example_module("math", "grading")

    assert mg.extract_boxed(r"so \boxed{42}.") == "42"
    # nested braces must stay balanced
    assert mg.extract_boxed(r"answer \boxed{\frac{2}{3}} done") == r"\frac{2}{3}"
    # last boxed wins
    assert mg.extract_boxed(r"\boxed{1} then \boxed{7}") == "7"
    assert mg.extract_boxed("no box here") is None


def test_extract_answer_fallbacks():
    mg = load_example_module("math", "grading")

    assert mg.extract_answer("The final answer is 17.") == "17"
    assert mg.extract_answer("blah\nresult is 5") in {"5", "result is 5"}
    assert mg.extract_answer("foo 3 bar 9\nnothing") == "9"


def test_numeric_equivalence():
    mg = load_example_module("math", "grading")

    assert mg.is_equivalent("0.5", r"\frac{1}{2}")
    assert mg.is_equivalent(r"\frac{2}{3}", "2/3")
    assert mg.is_equivalent("1,000", "1000")
    assert mg.is_equivalent("+7", "7")
    assert not mg.is_equivalent("0.5", r"\frac{1}{3}")


def test_latex_equivalence():
    mg = load_example_module("math", "grading")

    assert mg.is_equivalent(r"\left( 3, \frac{\pi}{2} \right)", r"(3, \frac{\pi}{2})")
    assert mg.is_equivalent(r"2\sqrt{3}", r"2 \sqrt{3}")
    assert mg.is_equivalent(r"\dfrac{2}{3}", r"\frac{2}{3}")
    assert not mg.is_equivalent(r"\frac{2}{3}", r"\frac{3}{2}")


def test_reward_shaping():
    mg = load_example_module("math", "grading")

    assert mg.reward(r"work... \boxed{9}", "9") == 1.0
    # wrong but boxed -> format bonus only
    assert mg.reward(r"work... \boxed{8}", "9") == 0.1
    # wrong and unformatted -> zero
    assert mg.reward("the answer is 8", "9") == 0.0


def test_math_env_offline_dataset_and_grade():
    from autoslm.envs.registry import load_environment

    env = load_environment(
        "math",
        {
            "train_path": os.path.join(FIXTURES, "math_train.jsonl"),
            "eval_path": os.path.join(FIXTURES, "math_eval.jsonl"),
            "eval_examples": 10,
        },
    )
    assert env.id == "math"

    train = env.dataset("train")
    assert len(train) == 2
    assert train[0]["question"]
    assert train[0]["gold"]

    ev = env.dataset("eval")
    assert len(ev) == 3

    # polar-coordinates example: graded equivalent despite \left/\right spacing
    polar = ev[0]
    completion = r"... so the answer is \boxed{\left( 3, \frac{\pi}{2} \right)}."
    assert env.grade(completion, polar) is True
    assert env.reward(completion, polar) == 1.0

    # 2/3 example: equivalent forms accepted, a different fraction rejected
    frac = ev[1]
    assert env.grade(r"\boxed{\frac{2}{3}}", frac) is True
    assert env.grade(r"\boxed{2/3}", frac) is True
    assert env.grade(r"\boxed{\frac{3}{4}}", frac) is False


def test_math_env_sft_target_has_boxed():
    from autoslm.envs.registry import load_environment

    env = load_environment("math", {"eval_path": os.path.join(FIXTURES, "math_eval.jsonl")})
    # solution without boxed -> grader-compatible boxed tail appended
    tgt = env.sft_target({"gold": "9", "solution": "196 has nine divisors"})
    assert r"\boxed{9}" in tgt
    # prompt carries the boxed-format instruction
    msgs = env.prompt_messages({"question": "Q?"})
    assert msgs[0]["role"] == "system"
    assert "boxed" in msgs[0]["content"].lower()


def test_math_env_config_parses():
    from autoslm.config_schema import spec_from_file

    cfg = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "examples", "math", "math_grpo_autoslm.toml")
    )
    spec = spec_from_file(cfg, run_id="math-test")
    assert spec.environment.id == "math"
    assert spec.phase == "rl"
    assert spec.train.seeds == (0, 1)
    assert spec.environment.params["train_dataset"] == "EleutherAI/hendrycks_math"
    assert "algebra" in spec.environment.params["train_configs"]


def test_math_env_live_hf_smoke():
    """Live HF load of MATH-500 (skipped offline / when datasets unavailable)."""
    if os.environ.get("AUTOSLM_SKIP_NET"):
        print("SKIP test_math_env_live_hf_smoke (AUTOSLM_SKIP_NET set)")
        return
    try:
        import datasets  # noqa: F401
    except Exception:
        print("SKIP test_math_env_live_hf_smoke (datasets not installed)")
        return
    try:
        from autoslm.envs.registry import load_environment

        env = load_environment("math", {"eval_examples": 5})
        ev = env.dataset("eval")
    except Exception as exc:  # network/hub hiccup -> don't fail the suite
        print(f"SKIP test_math_env_live_hf_smoke (load failed: {exc})")
        return
    assert len(ev) == 5
    assert all(e["question"] and e["gold"] for e in ev)
    print("live MATH-500 eval rows:", len(ev), "example gold:", ev[0]["gold"])


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
            passed += 1
        except Exception:
            print("FAIL", fn.__name__)
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
