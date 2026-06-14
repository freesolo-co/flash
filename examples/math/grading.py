"""Verifiable grader for competition-math answers (MATH / DeepScaleR style).

GSM8K answers are integers, so the GSM8K grader in ``grading.py`` only needs to
match a final number. "Real, extended" math post-train tasks (MATH-500,
DeepScaleR, AIME/AMC) have LaTeX answers like ``\\frac{2}{3}``,
``\\left( 3, \\frac{\\pi}{2} \\right)`` or ``2\\sqrt{3}``, so the grader has to:

  1. pull the model's final answer out of a ``\\boxed{...}`` (or a
     "final answer is ..." tail), and
  2. decide equivalence with the gold answer after normalizing LaTeX and, when
     both sides are numeric, comparing values.

This module is intentionally dependency-free (pure stdlib) so it can run on the
GPU worker, in the CLI, and in offline unit tests without ``sympy``/``datasets``.
It implements the widely-used Hendrycks-MATH normalization rules plus a numeric
fallback; it is deliberately conservative (no false positives) rather than a
full CAS. The same grader is used for the SFT target check, the RL reward, and
the eval, so all arms are scored identically.
"""

from __future__ import annotations

import re
from fractions import Fraction


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
def extract_boxed(text: str) -> str | None:
    r"""Return the content of the LAST ``\boxed{...}`` in ``text`` (brace-balanced)."""
    if not text:
        return None
    idx = text.rfind(r"\boxed")
    if idx < 0:
        idx = text.rfind(r"\fbox")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        # \boxed 123  (no braces) -> take the next token
        rest = text[idx + len(r"\boxed") :].strip()
        return rest.split()[0] if rest else None
    depth = 0
    out = []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out).strip()


_FINAL_ANSWER_RE = re.compile(
    r"(?:final answer is|answer is|answer:)\s*\$?(.+?)\$?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def extract_answer(text: str) -> str | None:
    r"""Best-effort extraction of a model's final answer.

    Order: ``\boxed{...}`` > "the final answer is X" tail > last line / last number.
    """
    if not text:
        return None
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    # "The final answer is X." style (take the last such match)
    matches = list(_FINAL_ANSWER_RE.finditer(text.strip().splitlines()[-1])) if text.strip() else []
    if not matches:
        matches = list(_FINAL_ANSWER_RE.finditer(text))
    if matches:
        cand = matches[-1].group(1).strip().rstrip(".")
        if cand:
            return cand
    # Fallback: last number-ish token on the last non-empty line.
    for line in reversed(text.strip().splitlines()):
        nums = re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", line)
        if nums:
            return nums[-1]
    return None


# ---------------------------------------------------------------------------
# Normalization (Hendrycks-MATH style)
# ---------------------------------------------------------------------------
# Unwrap \text{...}/\mbox{...}/\mathrm{...} to their inner content: some MATH
# answers ARE entirely text (e.g. \text{Evelyn}, \text{(C)}), so deleting them
# would wrongly blank the answer; keeping the inner content lets both sides match.
_TEXT_RE = re.compile(r"\\(?:text|mbox|mathrm|textbf|textrm|textsf|texttt)\s*\{([^{}]*)\}")


def _strip_wrappers(s: str) -> str:
    s = s.strip()
    # strip a single matched \left( ... \right) / outer delimiters used for points/tuples
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\!", "").replace(r"\,", "").replace("~", "")
    s = s.replace(r"\$", "").replace("$", "")
    s = s.replace(r"^{\circ}", "").replace(r"^\circ", "")
    s = s.replace(r"\%", "").replace("%", "")
    s = s.replace("\\ ", " ")
    return s.strip()


def _fix_fracs(s: str) -> str:
    r"""Normalize \frac shorthands: \frac34 -> \frac{3}{4}, \dfrac -> \frac, etc."""
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac").replace(r"\cfrac", r"\frac")
    s = s.replace(r"\frac ", r"\frac")
    out = s
    # \fracAB  (single-char numerator/denominator without braces)
    out = re.sub(r"\\frac([0-9a-zA-Z])([0-9a-zA-Z])", r"\\frac{\1}{\2}", out)
    # \frac{A}B  -> \frac{A}{B}
    return re.sub(r"\\frac\{([^{}]+)\}([0-9a-zA-Z])", r"\\frac{\1}{\2}", out)


def normalize_answer(ans: str) -> str:
    r"""Canonicalize a LaTeX/plain math answer string for string-equality compare."""
    if ans is None:
        return ""
    s = str(ans).strip()
    # remove trailing punctuation and surrounding math delimiters
    s = s.strip().strip(".").strip()
    for _ in range(3):  # unwrap nested \text{...} a few layers
        new = _TEXT_RE.sub(r"\1", s)
        if new == s:
            break
        s = new
    s = _strip_wrappers(s)
    s = _fix_fracs(s)
    s = s.replace(r"\cdot", "").replace(r"\times", "")
    s = s.replace(r"\!", "").replace("\\ ", "")
    s = s.replace(r"\,", "").replace("{,}", "")
    s = s.replace(r"^{\circ}", "").replace("^\\circ", "")
    s = s.replace(r"\pi", "pi")
    s = s.replace(r"\sqrt", "sqrt")
    s = s.replace(" ", "")
    s = s.replace(",", "")  # 1{,}000 / 1,000 -> 1000
    # 0.50 -> 0.5 ; strip a leading +
    s = s.lstrip("+")
    # remove a single layer of outer braces / parens that wrap the whole thing
    while len(s) >= 2 and s[0] in "({[" and s[-1] in ")}]":
        s = s[1:-1]
    return s.strip().lower()


def _to_number(s: str) -> Fraction | None:
    """Parse a plain number / fraction / a/b into a Fraction; else None."""
    if s is None:
        return None
    t = s.strip()
    if t == "":
        return None
    # \frac{a}{b}
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", t)
    if m:
        try:
            return Fraction(int(m.group(1)), int(m.group(2)))
        except (ValueError, ZeroDivisionError):
            return None
    # a/b
    if re.fullmatch(r"-?\d+/-?\d+", t):
        try:
            num, den = t.split("/")
            return Fraction(int(num), int(den))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return Fraction(t)  # ints and decimals
    except (ValueError, ZeroDivisionError):
        return None


def numbers_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    fa, fb = _to_number(a), _to_number(b)
    if fa is not None and fb is not None:
        if fa == fb:
            return True
        return abs(float(fa) - float(fb)) <= tol
    return False


# ---------------------------------------------------------------------------
# Equivalence + grading entry points
# ---------------------------------------------------------------------------
def is_equivalent(pred: str | None, gold: str | None) -> bool:
    """True if the predicted answer matches the gold answer."""
    if pred is None or gold is None:
        return False
    # numeric compare on the RAW strings (handles \frac{2}{3} vs 0.6667 etc.)
    if numbers_equal(pred, gold):
        return True
    np, ng = normalize_answer(pred), normalize_answer(gold)
    if np == "" or ng == "":
        return False
    if np == ng:
        return True
    return numbers_equal(np, ng)


def grade(completion: str, gold: str) -> bool:
    """Extract the model's answer from ``completion`` and compare to ``gold``."""
    return is_equivalent(extract_answer(completion or ""), gold)


def has_boxed_format(completion: str) -> bool:
    return bool(completion) and (r"\boxed" in completion or r"\fbox" in completion)


def reward(
    completion: str, gold: str, correct_reward: float = 1.0, format_reward: float = 0.1
) -> float:
    """Rule-based reward: full reward if correct, small bonus for boxed format."""
    if grade(completion, gold):
        return float(correct_reward)
    return float(format_reward) if has_boxed_format(completion) else 0.0


def accuracy(completions: list[str], golds: list[str]) -> float:
    if not completions:
        return 0.0
    n = sum(1 for c, g in zip(completions, golds, strict=False) if grade(c, g))
    return n / len(completions)
