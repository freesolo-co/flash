"""Self-CTRL — biased-sampler task (verifiers environment for Flash).

Reproduces the formal probabilistic-reasoning experiment from "Self-CTRL:
Self-Consistency Training with Reinforcement Learning" (arXiv:2606.18327) on the
Flash LoRA service. The model must imitate a *biased sampler* over a small outcome
set and, crucially, be **self-consistent**: in one response it (1) states the
probability distribution it will use (a self-explanation) and (2) emits N concrete
draws (its behavior). Self-CTRL optimizes the agreement between the two.

Flash scores ONE completion at a time and rejects group/batch reward funcs
(flash/envs/adapter.py), so the paper's distribution-level metric is reformulated as
an *intra-completion* reward: predicted distribution vs. the empirical distribution of
the same completion's draws. The headline R^2 (self-reported vs. measured bias) is
surfaced as a weight-0 eval metric.

Output contract the model must follow:

    <predict>{"1": 0.1, "2": 0.1, ..., "6": 0.4}</predict>
    <samples>3, 6, 6, 1, 6, 2, 6, ...</samples>   # exactly N integers in 1..6

The same env serves both phases: SFT reads the gold ``answer`` (a perfectly consistent
demonstration) via the adapter's ``sft_target``; GRPO uses the rubric below.

Module-level logic is stdlib-only and pure (unit-testable without verifiers); the
``verifiers``/``datasets`` imports are deferred into ``load_environment``.
"""

from __future__ import annotations

import json
import math
import random
import re

# Outcome space: faces of a biased die. Small enough that a consumer-GPU LoRA can move
# the distribution within a short GRPO run, matching the paper's "family of biased samplers".
FACES: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
N_SAMPLES: int = 20  # draws the model must emit per completion (the behavioral sample)

_PREDICT_RE = re.compile(r"<predict>(.*?)</predict>", re.DOTALL | re.IGNORECASE)
_SAMPLES_RE = re.compile(r"<samples>(.*?)</samples>", re.DOTALL | re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")


# --------------------------------------------------------------------------- parsing
def completion_text(completion) -> str:
    """The assistant text from a verifiers ``completion`` (message list or raw string)."""
    if isinstance(completion, list):
        return str(completion[-1].get("content", "")) if completion else ""
    return str(completion or "")


def parse_prediction(text: str) -> list[float] | None:
    """Parse the ``<predict>`` JSON into a normalized probability vector over FACES.

    Returns a list aligned to FACES (summing to 1), or None if absent/unparseable.
    """
    m = _PREDICT_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1).strip())
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    vec = []
    for face in FACES:
        v = obj.get(str(face), obj.get(face, 0.0))
        try:
            vec.append(max(0.0, float(v)))
        except (ValueError, TypeError):
            vec.append(0.0)
    total = sum(vec)
    if total <= 0:
        return None
    return [v / total for v in vec]


def parse_samples(text: str) -> list[int]:
    """Parse the ``<samples>`` block into the list of in-range face draws."""
    m = _SAMPLES_RE.search(text)
    if not m:
        return []
    return [int(tok) for tok in _INT_RE.findall(m.group(1)) if int(tok) in FACES]


def empirical_dist(samples: list[int]) -> list[float] | None:
    """Empirical probability vector over FACES from the draws (None if no valid draws)."""
    if not samples:
        return None
    counts = [0] * len(FACES)
    index = {f: i for i, f in enumerate(FACES)}
    for s in samples:
        counts[index[s]] += 1
    n = sum(counts)
    return [c / n for c in counts]


# --------------------------------------------------------------------------- metrics
def tv_distance(p: list[float], q: list[float]) -> float:
    """Total-variation distance in [0, 1] between two equal-length distributions."""
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q, strict=False))


def r2_score(p: list[float], q: list[float]) -> float:
    """Squared Pearson correlation (an R^2 proxy) between two vectors; 0 if degenerate."""
    n = len(p)
    mp, mq = sum(p) / n, sum(q) / n
    cov = sum((a - mp) * (b - mq) for a, b in zip(p, q, strict=False))
    vp = sum((a - mp) ** 2 for a in p)
    vq = sum((b - mq) ** 2 for b in q)
    if vp <= 0 or vq <= 0:
        return 0.0
    r = cov / math.sqrt(vp * vq)
    return max(0.0, min(1.0, r * r))


# --------------------------------------------------------------------- reward funcs
# All take only adapter-supplied args (completion/info) so they pass Flash's
# per-completion fail-fast guard. The weighted func drives GRPO; weight-0 funcs are
# eval metrics (run for reporting, not counted toward the reward).
def consistency_reward(completion=None, **kwargs) -> float:
    """Self-CTRL reward: agreement between the stated distribution and the empirical
    distribution of the emitted draws. ``1 - TV`` in [0, 1]; 0 if either block is
    missing/unparseable (which also penalizes off-format completions)."""
    text = completion_text(completion)
    predicted = parse_prediction(text)
    samples = parse_samples(text)
    if predicted is None or len(samples) < 2:
        return 0.0
    empirical = empirical_dist(samples)
    if empirical is None:
        return 0.0
    return 1.0 - tv_distance(predicted, empirical)


def r2_self_vs_empirical(completion=None, **kwargs) -> float:
    """Eval metric (weight 0): the paper's R^2 between self-reported and measured bias,
    per completion. Aggregated across the eval set this is the headline 0.24 -> 0.64."""
    text = completion_text(completion)
    predicted = parse_prediction(text)
    empirical = empirical_dist(parse_samples(text))
    if predicted is None or empirical is None:
        return 0.0
    return r2_score(predicted, empirical)


def imitation_match(completion=None, info=None, **kwargs) -> float:
    """Eval metric (weight 0): how well the emitted draws match the *target* sampler the
    row asked the model to imitate (``1 - TV`` against ``info['target']``)."""
    info = info or {}
    target = info.get("target")
    empirical = empirical_dist(parse_samples(completion_text(completion)))
    if not target or empirical is None:
        return 0.0
    return 1.0 - tv_distance(list(target), empirical)


# --------------------------------------------------------------------------- dataset
def _random_target(rng: random.Random) -> list[float]:
    """A biased distribution over FACES (Dirichlet-like via normalized exponentials)."""
    raw = [rng.expovariate(1.0) for _ in FACES]
    total = sum(raw)
    return [r / total for r in raw]


def _gold_answer(target: list[float], rng: random.Random) -> str:
    """A perfectly self-consistent demonstration (SFT target): a rounded predicted
    distribution plus N draws sampled from that same distribution."""
    rounded = [round(p, 3) for p in target]
    s = sum(rounded) or 1.0
    rounded = [round(p / s, 3) for p in rounded]
    pred = {str(face): rounded[i] for i, face in enumerate(FACES)}
    draws = rng.choices(FACES, weights=target, k=N_SAMPLES)
    return (
        f"<predict>{json.dumps(pred)}</predict>\n"
        f"<samples>{', '.join(str(d) for d in draws)}</samples>"
    )


_INSTRUCTIONS = (
    "You are a biased random sampler over the faces 1, 2, 3, 4, 5, 6.\n"
    "Respond in EXACTLY this format and nothing else:\n"
    '<predict>{{"1": p1, "2": p2, "3": p3, "4": p4, "5": p5, "6": p6}}</predict>\n'
    f"<samples>d1, d2, ..., d{N_SAMPLES}</samples>\n"
    "The <predict> block is the probability you assign to each face (must sum to ~1).\n"
    f"The <samples> block is EXACTLY {N_SAMPLES} independent draws from your sampler.\n"
    "Be self-consistent: the empirical frequencies of your draws must match the "
    "probabilities you state. Target bias for this round: {hint}."
)


def _hint(target: list[float]) -> str:
    order = sorted(range(len(FACES)), key=lambda i: target[i], reverse=True)
    top = FACES[order[0]]
    bottom = FACES[order[-1]]
    return f"favor {top}, avoid {bottom}"


def build_rows(n: int, seed: int) -> list[dict]:
    """Procedurally generate the synthetic biased-sampler dataset rows."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        target = _random_target(rng)
        prompt = [{"role": "user", "content": _INSTRUCTIONS.format(hint=_hint(target))}]
        rows.append(
            {
                "prompt": prompt,
                "answer": _gold_answer(target, rng),
                "info": {"target": target, "n_samples": N_SAMPLES},
            }
        )
    return rows


def load_environment(num_examples: int = 512, seed: int = 0, **kwargs):
    """Build the Self-CTRL biased-sampler ``SingleTurnEnv``.

    ``num_examples`` / ``seed`` control the synthetic dataset. Extra kwargs are forwarded
    to ``SingleTurnEnv`` (so Flash's ``[environment.params]`` flow through).
    """
    import verifiers as vf
    from datasets import Dataset

    dataset = Dataset.from_list(build_rows(num_examples, seed))
    rubric = vf.Rubric(
        funcs=[consistency_reward, r2_self_vs_empirical, imitation_match],
        weights=[1.0, 0.0, 0.0],
    )
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric, **kwargs)
