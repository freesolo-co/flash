"""``PaperCase`` manifest: the unit of work for the benchmark.

A ``PaperCase`` captures everything the harness needs to replicate one paper's training
result: the task goal handed to the agent, the dataset refs (train + held-out eval), the
Flash model to use, the algorithm, the paper's reported metric, the difficulty mode, and —
in easy mode — the supplied reward module the agent is told not to touch.

Manifests are TOML files (see ``autoenv/cases/*.toml``) parsed into frozen dataclasses in
the tolerant style of ``flash/spec.py``: a present-but-wrong-typed field raises, a missing
field takes its default. Relative paths (dataset files, supplied reward) resolve against the
manifest file's directory so a case folder is self-contained and relocatable.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from flash.catalog import ALGORITHMS

DIFFICULTIES = ("easy", "hard")


class ManifestError(ValueError):
    """A PaperCase manifest is malformed or internally inconsistent."""


def _req_str(raw: dict[str, Any], key: str, *, where: str) -> str:
    try:
        value = raw[key]
    except KeyError as exc:
        raise ManifestError(f"{where}: missing required field {key!r}") from exc
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{where}: {key!r} must be a non-empty string")
    return value.strip()


def _opt_str(raw: dict[str, Any], key: str, default: str = "", *, where: str) -> str:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ManifestError(f"{where}: {key!r} must be a string")
    return value.strip()


def _opt_float(raw: dict[str, Any], key: str, default: float | None, *, where: str) -> float | None:
    value = raw.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{where}: {key!r} must be a number")
    return float(value)


def _opt_int(raw: dict[str, Any], key: str, default: int | None, *, where: str) -> int | None:
    value = raw.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{where}: {key!r} must be an integer")
    return int(value)


def _opt_bool(raw: dict[str, Any], key: str, default: bool, *, where: str) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ManifestError(f"{where}: {key!r} must be a boolean")
    return value


@dataclass(frozen=True)
class DatasetRef:
    """Where the train rows and the held-out eval rows come from.

    ``train``/``eval`` are dataset refs resolved by ``autoenv.ingest.sources``: a local
    ``.jsonl``/``.json`` path, an ``http(s)`` URL to JSON/JSONL, or a Hugging Face dataset
    (``hf:<id>[:<config>]`` or a bare ``org/name``). ``input_field``/``output_field`` map the
    source columns onto Freesolo's canonical ``input``/``output`` row keys.
    """

    train: str
    eval: str
    train_split: str = "train"
    eval_split: str = "test"
    input_field: str = "input"
    output_field: str = "output"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DatasetRef:
        where = "[dataset]"
        if not isinstance(raw, dict):
            raise ManifestError(f"{where} must be a table")
        return cls(
            train=_req_str(raw, "train", where=where),
            eval=_req_str(raw, "eval", where=where),
            train_split=_opt_str(raw, "train_split", "train", where=where),
            eval_split=_opt_str(raw, "eval_split", "test", where=where),
            input_field=_opt_str(raw, "input_field", "input", where=where),
            output_field=_opt_str(raw, "output_field", "output", where=where),
        )


@dataclass(frozen=True)
class PaperMetric:
    """The metric the paper reports on its eval split, and the number it reports.

    ``name`` must be a key in ``autoenv.eval.metrics.METRICS``. ``reported`` is the paper's
    headline number; ``base_reported`` (optional) is the paper's untrained-baseline number,
    which the eval stage currently uses as the baseline for improvement-normalized scoring
    (Flash serving is adapter-scoped, so there is no bare-base endpoint to measure yet —
    independently measuring the base is a later milestone).
    """

    name: str
    reported: float
    higher_is_better: bool = True
    base_reported: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PaperMetric:
        where = "[metric]"
        if not isinstance(raw, dict):
            raise ManifestError(f"{where} must be a table")
        reported = _opt_float(raw, "reported", None, where=where)
        if reported is None:
            raise ManifestError(f"{where}: missing required field 'reported'")
        return cls(
            name=_req_str(raw, "name", where=where),
            reported=reported,
            higher_is_better=_opt_bool(raw, "higher_is_better", True, where=where),
            base_reported=_opt_float(raw, "base_reported", None, where=where),
        )


@dataclass(frozen=True)
class PaperCase:
    """One paper to replicate. Loaded from a TOML manifest; see ``autoenv/cases/``."""

    id: str
    goal: str
    paper_url: str
    base_model_paper: str
    flash_model: str
    algorithm: str
    dataset: DatasetRef
    metric: PaperMetric
    difficulty: str = "easy"
    # Easy mode: the reward module copied into the agent's environment and frozen. Resolved
    # to an absolute path against the manifest dir at load time. Required when difficulty=easy.
    supplied_reward_path: str | None = None
    # Guardrails. ``max_usd`` gates every real submit (preflight cost must be <=); the cap knobs
    # bound a single run's footprint so an early-milestone smoke case can't run away.
    max_usd: float = 5.0
    max_train_steps: int | None = None
    max_train_examples: int | None = None
    notes: str = ""
    # Absolute path the manifest was loaded from (None for programmatically built cases).
    source_path: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ManifestError(
                f"case {self.id!r}: algorithm {self.algorithm!r} not in {ALGORITHMS}"
            )
        if self.difficulty not in DIFFICULTIES:
            raise ManifestError(
                f"case {self.id!r}: difficulty {self.difficulty!r} not in {DIFFICULTIES}"
            )
        if self.difficulty == "easy" and not self.supplied_reward_path:
            raise ManifestError(
                f"case {self.id!r}: easy-mode cases must set supplied_reward_path "
                "(the reward the agent is given and not allowed to change)"
            )

    @property
    def base_dir(self) -> Path:
        """Directory the manifest's relative paths resolve against."""
        return Path(self.source_path).parent if self.source_path else Path.cwd()

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, source_path: str | None = None) -> PaperCase:
        if not isinstance(raw, dict):
            raise ManifestError("a PaperCase manifest must be a table")
        where = "case"
        case_id = _req_str(raw, "id", where=where)
        base = Path(source_path).parent if source_path else Path.cwd()

        reward = raw.get("supplied_reward_path")
        if reward is not None:
            if not isinstance(reward, str):
                raise ManifestError(f"case {case_id!r}: supplied_reward_path must be a string")
            reward = str((base / reward).resolve()) if reward.strip() else None

        return cls(
            id=case_id,
            goal=_req_str(raw, "goal", where=where),
            paper_url=_opt_str(raw, "paper_url", "", where=where),
            base_model_paper=_req_str(raw, "base_model_paper", where=where),
            flash_model=_req_str(raw, "flash_model", where=where),
            algorithm=_opt_str(raw, "algorithm", "grpo", where=where).lower(),
            dataset=DatasetRef.from_dict(raw.get("dataset") or {}),
            metric=PaperMetric.from_dict(raw.get("metric") or {}),
            difficulty=_opt_str(raw, "difficulty", "easy", where=where).lower(),
            supplied_reward_path=reward,
            max_usd=_opt_float(raw, "max_usd", 5.0, where=where),
            max_train_steps=_opt_int(raw, "max_train_steps", None, where=where),
            max_train_examples=_opt_int(raw, "max_train_examples", None, where=where),
            notes=_opt_str(raw, "notes", "", where=where),
            source_path=str(Path(source_path).resolve()) if source_path else None,
        )

    @classmethod
    def load(cls, path: str | Path) -> PaperCase:
        """Parse a PaperCase from a TOML manifest file."""
        p = Path(path)
        try:
            raw = tomllib.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ManifestError(f"manifest not found: {p}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise ManifestError(f"{p}: invalid TOML — {exc}") from exc
        return cls.from_dict(raw, source_path=str(p))

    def resolved_train(self) -> str:
        """The train dataset ref with any local relative path made absolute."""
        return _resolve_ref(self.dataset.train, self.base_dir)

    def resolved_eval(self) -> str:
        """The eval dataset ref with any local relative path made absolute."""
        return _resolve_ref(self.dataset.eval, self.base_dir)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("source_path", None)
        return data


# Kept in sync with what ``autoenv.ingest.sources`` actually resolves: remote http(s)/hf:
# refs, and local .jsonl/.json files. (Bare ``org/name`` HF ids have no suffix and are handled
# by the no-suffix branch below.)
_REMOTE_PREFIXES = ("http://", "https://", "hf:")
_LOCAL_SUFFIXES = (".jsonl", ".json")


def _resolve_ref(ref: str, base: Path) -> str:
    """Make a local relative dataset ref absolute; leave remote refs untouched.

    Remote = an explicit ``http(s)``/``hf:`` prefix, OR a bare ``org/name`` Hugging Face
    dataset id (no recognised file suffix). A ref with a local data suffix is treated as a
    file path and resolved against the manifest dir.
    """
    if ref.lower().startswith(_REMOTE_PREFIXES):
        return ref
    if not Path(ref).suffix:
        # No file suffix -> a bare HF dataset id (``org/name``), resolved remotely later.
        return ref
    if Path(ref).suffix.lower() not in _LOCAL_SUFFIXES:
        return ref
    if Path(ref).is_absolute():
        return ref
    return str((base / ref).resolve())
