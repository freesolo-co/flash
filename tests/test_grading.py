"""Unit tests for the shared grader (run on CPU, no network)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests._examples import load_example_module

g = load_example_module("gsm8k", "grading")
d = load_example_module("gsm8k", "data")


def test_extract_gold():
    assert g.extract_gold_answer("Some reasoning.\n#### 42") == "42"
    assert g.extract_gold_answer("steps\n#### 1,234") == "1234"
    assert g.extract_gold_answer("x\n#### 3.50") == "3.5"
    assert g.extract_gold_answer("no marker but ends with 7") == "7"


def test_extract_pred_boxed_priority():
    assert g.extract_pred_answer("blah \\boxed{18} done") == "18"
    # boxed wins over trailing number
    assert g.extract_pred_answer("\\boxed{18}\nso the answer is 99") == "18"
    assert g.extract_pred_answer("#### 21") == "21"
    assert g.extract_pred_answer("The answer is 21") == "21"
    assert g.extract_pred_answer("no numbers here") is None


def test_is_correct():
    assert g.is_correct("\\boxed{42}", "reason\n#### 42")
    assert g.is_correct("answer: 1,234", "#### 1234")
    assert g.is_correct("\\boxed{3.0}", "#### 3")
    assert not g.is_correct("\\boxed{41}", "#### 42")
    assert not g.is_correct("no answer", "#### 42")
    # gold passed as already-extracted answer
    assert g.is_correct("\\boxed{42}", "42")


def test_format_and_reward():
    assert g.has_format("\\boxed{5}")
    assert g.has_format("#### 5")
    assert not g.has_format("just 5 in text")
    # correct + format
    assert abs(g.reward("\\boxed{42}", "#### 42") - 1.1) < 1e-9
    # correct via last-number, no format bonus
    assert abs(g.reward("the answer is 42", "#### 42") - 1.0) < 1e-9
    # wrong but formatted
    assert abs(g.reward("\\boxed{41}", "#### 42") - 0.1) < 1e-9
    # wrong, unformatted
    assert g.reward("41", "#### 42") == 0.0


def test_accuracy():
    comps = ["\\boxed{1}", "\\boxed{2}", "\\boxed{9}"]
    golds = ["#### 1", "#### 2", "#### 3"]
    assert abs(g.accuracy(comps, golds) - (2 / 3)) < 1e-9


def test_build_target_text():
    sol = "Tom has 3 apples. He buys 2 more. 3+2=5.\n#### 5"
    t = d.build_target_text(sol)
    assert "\\boxed{5}" in t
    assert "####" not in t
    # the reasoning body is preserved
    assert "Tom has 3 apples" in t


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
