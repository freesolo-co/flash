from __future__ import annotations

import concurrent.futures
import contextlib
import importlib.metadata
import json
import re
import shlex
import sys
import threading
import tomllib
import types
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
from flash.engine.profiling.dataset_profile import (
    PackagedDatasetUnavailable,
    profile_packaged_sft_dataset,
)
from flash.engine.profiling.image_tokens import ImageGeometry
from flash.engine.profiling.workload_profile import sft_profile_input_digest

_ROOT = Path(__file__).resolve().parent.parent
_REQUIREMENT_NAME_PREFIX = re.compile(r"^([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")
_SHELL_COMMAND_CHARACTERS = frozenset(";&|")


def _docker_run_scripts(dockerfile: str) -> list[str]:
    scripts: list[str] = []
    instruction: list[str] = []
    for line in dockerfile.splitlines():
        stripped = line.lstrip()
        if not instruction and (not stripped or stripped.startswith("#")):
            continue
        continued = line.rstrip().endswith("\\")
        instruction.append(line.rstrip()[:-1] if continued else line)
        if continued:
            continue
        logical = " ".join(instruction).strip()
        instruction = []
        parts = logical.split(maxsplit=1)
        if len(parts) == 2 and parts[0].upper() == "RUN":
            scripts.append(parts[1])
    if instruction:
        raise AssertionError("managed worker Dockerfile ends with an incomplete instruction")
    return scripts


def _pip_install_argument_starts(tokens: list[str]) -> list[int]:
    starts: set[int] = set()
    for index, token in enumerate(tokens):
        executable = token.rsplit("/", 1)[-1]
        if re.fullmatch(r"pip(?:3(?:\.\d+)?)?", executable) and tokens[index + 1 : index + 2] == [
            "install"
        ]:
            starts.add(index + 2)
        elif executable == "uv" and tokens[index + 1 : index + 3] == ["pip", "install"]:
            starts.add(index + 3)
        elif re.fullmatch(r"python(?:3(?:\.\d+)?)?", executable) and tokens[
            index + 1 : index + 4
        ] == ["-m", "pip", "install"]:
            starts.add(index + 4)
    return sorted(starts)


def _docker_pip_requirement_tokens(dockerfile: str) -> list[str]:
    requirements: list[str] = []
    for script in _docker_run_scripts(dockerfile):
        lexer = shlex.shlex(script, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        try:
            tokens = list(lexer)
        except ValueError as exc:
            raise AssertionError("managed worker has an invalid RUN shell command") from exc
        for start in _pip_install_argument_starts(tokens):
            for token in tokens[start:]:
                if token and set(token).issubset(_SHELL_COMMAND_CHARACTERS):
                    break
                if not token.startswith("-"):
                    requirements.append(token)
    return requirements


def _managed_worker_freesolo_version(dockerfile: str | None = None) -> Version:
    if dockerfile is None:
        dockerfile = (_ROOT / "Dockerfile.worker").read_text(encoding="utf-8")
    expected_name = canonicalize_name("freesolo")
    matches: list[tuple[str, Version]] = []
    for value in _docker_pip_requirement_tokens(dockerfile):
        name_match = _REQUIREMENT_NAME_PREFIX.match(value)
        if name_match is None or canonicalize_name(name_match.group(1)) != expected_name:
            continue
        try:
            requirement = Requirement(value)
        except InvalidRequirement as exc:
            raise AssertionError(
                f"managed worker has an invalid freesolo requirement: {value!r}"
            ) from exc
        assert canonicalize_name(requirement.name) == expected_name
        assert requirement.marker is None, (
            "managed worker freesolo requirement must not use an environment marker, "
            f"got {requirement!s}"
        )
        exact = [
            Version(item.version)
            for item in requirement.specifier
            if item.operator in {"==", "==="}
        ]
        assert len(exact) == 1, f"managed worker must exactly pin freesolo, got {requirement!s}"
        assert len(requirement.specifier) == 1, (
            f"managed worker must use only one freesolo specifier, got {requirement!s}"
        )
        matches.append((value, exact[0]))
    assert len(matches) == 1, (
        "expected exactly one managed-worker freesolo requirement, "
        f"found {[value for value, _version in matches]}"
    )
    return matches[0][1]


def _locked_freesolo_version(lockfile: str | None = None) -> Version:
    lockfile = lockfile or (_ROOT / "uv.lock").read_text(encoding="utf-8")
    packages = tomllib.loads(lockfile).get("package", [])
    expected_name = canonicalize_name("freesolo")
    matches = [
        package
        for package in packages
        if isinstance(name := package.get("name"), str) and canonicalize_name(name) == expected_name
    ]
    assert len(matches) == 1, f"expected exactly one locked freesolo package, found {len(matches)}"
    version = matches[0].get("version")
    assert isinstance(version, str), "locked freesolo package version must be a string"
    assert version, "locked freesolo package is missing a version"
    return Version(version)


def _assert_decoder_conformance_versions(
    *,
    installed_version: str | None = None,
    dockerfile: str | None = None,
    lockfile: str | None = None,
) -> None:
    expected = _managed_worker_freesolo_version(dockerfile)
    installed = Version(installed_version or importlib.metadata.version("freesolo"))
    assert installed == expected, (
        f"decoder conformance requires managed-worker freesolo {expected}, installed {installed}"
    )
    locked = _locked_freesolo_version(lockfile)
    assert locked == expected, f"uv.lock resolves freesolo {locked}, managed worker pins {expected}"


@pytest.fixture(scope="module")
def _decoder_sdk_lockstep() -> None:
    _assert_decoder_conformance_versions()


def test_decoder_conformance_sdk_matches_the_managed_worker_pin() -> None:
    _assert_decoder_conformance_versions()


@pytest.mark.parametrize(
    ("marker", "message"),
    [
        ("python_version < '3.0'", "must not use an environment marker"),
        ("python_version == '3.11'", "must not use an environment marker"),
        ("python_version == '3.12'", "must not use an environment marker"),
        ("python_version <", "has an invalid freesolo requirement"),
    ],
)
def test_decoder_conformance_worker_guard_rejects_requirement_markers(marker, message) -> None:
    expected = _managed_worker_freesolo_version()
    dockerfile = (_ROOT / "Dockerfile.worker").read_text(encoding="utf-8")
    marked = dockerfile.replace(f'"freesolo=={expected}"', f'"freesolo=={expected}; {marker}"', 1)

    with pytest.raises(AssertionError, match=message):
        _managed_worker_freesolo_version(marked)


@pytest.mark.parametrize(
    ("extra_install", "message"),
    [
        (
            'RUN pip install --no-cache-dir "freesolo==9.9.9"',
            "expected exactly one managed-worker freesolo requirement",
        ),
        (
            'RUN uv pip install --python /opt/worker/bin/python "FreeSolo==9.9.9"',
            "expected exactly one managed-worker freesolo requirement",
        ),
        (
            'RUN /opt/worker/bin/python -m pip install "FREESOLO==9.9.9"',
            "expected exactly one managed-worker freesolo requirement",
        ),
        (
            "RUN pip3 install \"freesolo==9.9.9; python_version == '3.11'\"",
            "must not use an environment marker",
        ),
        (
            "RUN python3.12 -m pip install \"freesolo==9.9.9; python_version == '3.12'\"",
            "must not use an environment marker",
        ),
    ],
)
def test_decoder_conformance_worker_guard_rejects_later_overrides(extra_install, message) -> None:
    dockerfile = (_ROOT / "Dockerfile.worker").read_text(encoding="utf-8")

    with pytest.raises(AssertionError, match=message):
        _managed_worker_freesolo_version(f"{dockerfile}\n{extra_install}\n")


@pytest.mark.parametrize(
    "install",
    [
        'RUN pip install "freesolo==0.4.2"',
        'RUN uv pip install --python /opt/worker/bin/python "FreeSolo==0.4.2"',
        'RUN /opt/worker/bin/python -m pip install "FREESOLO==0.4.2"',
    ],
)
def test_decoder_conformance_worker_guard_counts_each_install_once(install) -> None:
    assert _managed_worker_freesolo_version(install) == Version("0.4.2")


@pytest.mark.parametrize("distinct_name", ["free_solo", "free.solo", "FREE-SOLO"])
def test_decoder_conformance_worker_guard_ignores_distinct_separator_names(
    distinct_name,
) -> None:
    dockerfile = (_ROOT / "Dockerfile.worker").read_text(encoding="utf-8")

    assert canonicalize_name(distinct_name) == canonicalize_name("free-solo")
    assert canonicalize_name(distinct_name) != canonicalize_name("freesolo")
    assert _managed_worker_freesolo_version(
        f'{dockerfile}\nRUN pip install "{distinct_name}==9.9.9"\n'
    ) == Version("0.4.2")


def test_decoder_conformance_worker_guard_rejects_second_install_in_the_same_run() -> None:
    dockerfile = (_ROOT / "Dockerfile.worker").read_text(encoding="utf-8")
    duplicate = dockerfile.replace(
        '    "runpod==1.12.0"',
        '    "runpod==1.12.0" && pip install "FreeSolo==9.9.9"',
        1,
    )

    with pytest.raises(
        AssertionError, match="expected exactly one managed-worker freesolo requirement"
    ):
        _managed_worker_freesolo_version(duplicate)


def test_decoder_conformance_worker_guard_rejects_same_line_duplicate() -> None:
    expected = _managed_worker_freesolo_version()
    dockerfile = (_ROOT / "Dockerfile.worker").read_text(encoding="utf-8")
    duplicate = dockerfile.replace(
        f'    "freesolo=={expected}" \\\n',
        f'    "freesolo=={expected}" "FreeSolo==9.9.9" \\\n',
        1,
    )

    with pytest.raises(
        AssertionError, match="expected exactly one managed-worker freesolo requirement"
    ):
        _managed_worker_freesolo_version(duplicate)


def test_decoder_conformance_worker_guard_rejects_same_line_malformed_requirement() -> None:
    expected = _managed_worker_freesolo_version()
    dockerfile = (_ROOT / "Dockerfile.worker").read_text(encoding="utf-8")
    malformed = dockerfile.replace(
        f'    "freesolo=={expected}" \\\n',
        f'    "freesolo=={expected}" "freesolo==" \\\n',
        1,
    )

    with pytest.raises(AssertionError, match="has an invalid freesolo requirement"):
        _managed_worker_freesolo_version(malformed)


def test_decoder_conformance_version_guard_detects_independent_drift() -> None:
    expected = _managed_worker_freesolo_version()
    drifted = Version(f"{expected.major}.{expected.minor}.{expected.micro + 1}")
    dockerfile = (_ROOT / "Dockerfile.worker").read_text(encoding="utf-8")
    lockfile = (_ROOT / "uv.lock").read_text(encoding="utf-8")

    drifted_worker = dockerfile.replace(f'"freesolo=={expected}"', f'"freesolo=={drifted}"', 1)
    with pytest.raises(AssertionError, match=rf"installed {expected}"):
        _assert_decoder_conformance_versions(
            installed_version=str(expected), dockerfile=drifted_worker
        )

    drifted_lock = lockfile.replace(
        f'name = "freesolo"\nversion = "{expected}"',
        f'name = "freesolo"\nversion = "{drifted}"',
        1,
    )
    with pytest.raises(AssertionError, match=rf"uv\.lock resolves freesolo {drifted}"):
        _assert_decoder_conformance_versions(installed_version=str(expected), lockfile=drifted_lock)

    package_start = lockfile.index('[[package]]\nname = "freesolo"\n')
    package_end = lockfile.index("\n[[package]]", package_start)
    lock_without_freesolo = lockfile[:package_start] + lockfile[package_end + 1 :]
    with pytest.raises(
        AssertionError, match="expected exactly one locked freesolo package, found 0"
    ):
        _assert_decoder_conformance_versions(
            installed_version=str(expected), lockfile=lock_without_freesolo
        )

    with pytest.raises(AssertionError, match=rf"installed {drifted}"):
        _assert_decoder_conformance_versions(installed_version=str(drifted))


@pytest.mark.parametrize("duplicate_name", ["FreeSolo", "FREESOLO"])
def test_decoder_conformance_lock_guard_rejects_canonical_name_duplicates(
    duplicate_name,
) -> None:
    expected = _managed_worker_freesolo_version()
    lockfile = (_ROOT / "uv.lock").read_text(encoding="utf-8")
    duplicate = f'\n[[package]]\nname = "{duplicate_name}"\nversion = "{expected}"\n'

    with pytest.raises(
        AssertionError, match="expected exactly one locked freesolo package, found 2"
    ):
        _locked_freesolo_version(lockfile + duplicate)


@pytest.mark.parametrize("distinct_name", ["free_solo", "free.solo", "FREE-SOLO"])
def test_decoder_conformance_lock_guard_ignores_distinct_separator_variants(
    distinct_name,
) -> None:
    expected = _managed_worker_freesolo_version()
    lockfile = (_ROOT / "uv.lock").read_text(encoding="utf-8")
    distinct = f'\n[[package]]\nname = "{distinct_name}"\nversion = "0"\n'

    assert _locked_freesolo_version(lockfile + distinct) == expected


class FakeTokenizer:
    eos_token = "|"
    eos_token_id = 2
    pad_token = None
    pad_token_id = 0
    all_special_ids: ClassVar[list[int]] = [0, 1, 2]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
        **_kwargs,
    ):
        assert not tokenize
        text = "".join(str(message.get("content") or "") for message in messages)
        return text + (">" if add_generation_prompt else "")

    def __call__(self, texts, *, truncation=False, max_length=None):
        if isinstance(texts, str):
            texts = [texts]
        ids = [[3 + ord(char) % 89 for char in text] for text in texts]
        if truncation:
            assert max_length is not None
            ids = [row[:max_length] for row in ids]
        return {"input_ids": ids}


class FakeMultimodalTokenizer(FakeTokenizer):
    """A tokenizer that renders image blocks the way a real VL chat template does.

    One ``<|image_pad|>`` per image block, which is exactly what the plain tokenizer emits; the
    profiler is responsible for expanding it to the run the processor would produce.
    """

    IMAGE_PAD_ID = 700

    def convert_tokens_to_ids(self, token):
        return self.IMAGE_PAD_ID if token == "<|image_pad|>" else 5

    def convert_ids_to_tokens(self, token_id):
        return "<|image_pad|>" if token_id == self.IMAGE_PAD_ID else "<unknown>"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
        return_dict=False,
        **_kwargs,
    ):
        if not tokenize:
            return super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        ids: list[int] = []
        for message in messages:
            content = message.get("content")
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for block in blocks:
                if block.get("type") == "image":
                    ids.append(self.IMAGE_PAD_ID)
                else:
                    ids.extend(3 + ord(char) % 89 for char in str(block.get("text") or ""))
        if add_generation_prompt:
            ids.append(9)
        return {"input_ids": ids} if return_dict else ids


def _spec(
    *,
    params: dict | None = None,
    environment_id: str = "team/example",
    model: str = "test/model",
) -> JobSpec:
    base = JobSpec(
        model=model,
        model_revision="a" * 40,
        algorithm="sft",
        environment=EnvironmentSpec(
            id=environment_id,
            resolved_sha="b" * 40,
            params=params or {},
        ),
        train=TrainSpec(epochs=2, batch_size=2, max_context_tokens=128),
        seed=7,
    )
    digest = sft_profile_input_digest(
        base,
        tokenizer_revision=base.model_revision,
        producer_version="1.2.3",
    )
    return replace(
        base,
        workload_profile_input_digest=digest,
        workload_profile_producer_version="1.2.3",
    )


def _package(tmp_path: Path, files: dict[str, str]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    entrypoint = tmp_path / "environment.py"
    entrypoint.write_text("raise RuntimeError('environment code executed')\n", encoding="utf-8")
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return entrypoint


def _profile(entrypoint: Path, *, params: dict | None = None):
    spec = _spec(params=params, environment_id=str(entrypoint))
    profile = profile_packaged_sft_dataset(
        spec,
        producer_version="1.2.3",
        tokenizer_loader=lambda _model, _revision: FakeTokenizer(),
        packing_support=lambda _model, _revision: ("pure-attention", True),
    )
    return spec, profile


class _ConcurrentAutoTokenizerImportTrap(types.ModuleType):
    """a fake lazy module that rejects overlapping autotokenizer resolution."""

    def __init__(self) -> None:
        super().__init__("transformers")
        self._state_lock = threading.Lock()
        self._overlap = threading.Event()
        self._active = 0
        self.max_active = 0

    def __getattr__(self, name: str):
        if name != "AutoTokenizer":
            raise AttributeError(name)
        with self._state_lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            if self._active > 1:
                self._overlap.set()
        self._overlap.wait(timeout=0.05)
        with self._state_lock:
            overlapped = self._active > 1
            self._active -= 1
        if overlapped:
            raise ModuleNotFoundError(
                "Could not import module 'AutoTokenizer'. Are this object's requirements defined "
                "correctly?"
            )

        class _AutoTokenizer:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                return FakeTokenizer()

        return _AutoTokenizer


def test_concurrent_profile_serializes_transformers_lazy_import(tmp_path, monkeypatch) -> None:
    import flash.engine.profiling.tokenizer as tokenizer_module

    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": (
                '{"input":"one","output":"alpha"}\n{"input":"two","output":"beta"}\n'
            )
        },
    )
    spec = _spec(environment_id=str(entrypoint))

    def run_profiles():
        gate = threading.Barrier(2)

        def prepare():
            gate.wait(timeout=1)
            return profile_packaged_sft_dataset(
                spec,
                producer_version="1.2.3",
                packing_support=lambda _model, _revision: ("pure-attention", True),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            return [pool.submit(prepare) for _ in range(2)]

    serialized = _ConcurrentAutoTokenizerImportTrap()
    monkeypatch.setitem(sys.modules, "transformers", serialized)
    profiles = [future.result() for future in run_profiles()]
    assert serialized.max_active == 1
    assert [profile.retained_examples for profile in profiles] == [2, 2]

    unguarded = _ConcurrentAutoTokenizerImportTrap()
    monkeypatch.setitem(sys.modules, "transformers", unguarded)
    monkeypatch.setattr(tokenizer_module, "_TRANSFORMERS_IMPORT_LOCK", contextlib.nullcontext())
    errors = [future.exception() for future in run_profiles()]
    assert unguarded.max_active == 2
    assert any(
        "Could not import module 'AutoTokenizer'" in str(error)
        for error in errors
        if error is not None
    )


def test_profile_reads_packaged_train_jsonl_without_executing_environment_code(tmp_path) -> None:
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": (
                '{"input":"one","output":"alpha"}\n{"input":"two","output":"beta"}\n'
            )
        },
    )

    spec, profile = _profile(entrypoint)

    assert profile.source_examples == 2
    assert profile.selected_examples == 2
    assert profile.retained_examples == 2
    assert profile.authoritative_steps == 2
    assert profile.input_digest == spec.workload_profile_input_digest


def test_profile_rejects_the_plural_dataset_layout_that_the_worker_never_reads(tmp_path) -> None:
    entrypoint = _package(
        tmp_path,
        {"datasets/train.jsonl": '{"input":"one","output":"alpha"}\n'},
    )

    with pytest.raises(PackagedDatasetUnavailable, match=r"dataset/train\.jsonl"):
        _profile(entrypoint)


def test_profile_prefers_canonical_dataset_over_plural_layout(tmp_path) -> None:
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": '{"input":"canonical","output":"yes"}\n',
            "datasets/train.jsonl": (
                '{"input":"plural-one","output":"no"}\n{"input":"plural-two","output":"no"}\n'
            ),
        },
    )

    _spec_value, profile = _profile(entrypoint)

    assert profile.source_examples == 1


def test_profile_honors_split_and_dataset_path(tmp_path) -> None:
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": '{"input":"train","output":"no"}\n',
            "dataset/sft.jsonl": '{"input":"split","output":"yes"}\n',
            "custom.json": json.dumps(
                [
                    {"input": "custom-one", "output": "yes"},
                    {"input": "custom-two", "output": "yes"},
                ]
            ),
        },
    )

    _split_spec, split_profile = _profile(entrypoint, params={"split": "sft"})
    _path_spec, path_profile = _profile(
        entrypoint, params={"dataset_path": "custom.json", "split": "sft"}
    )

    assert split_profile.source_examples == 1
    assert path_profile.source_examples == 2


@pytest.mark.usefixtures("_decoder_sdk_lockstep")
@pytest.mark.parametrize(
    ("suffix", "content", "expected"),
    [
        (
            ".json",
            '[{"input":"object","output":"yes"},"string row"]',
            [
                {"input": "object", "output": "yes"},
                {"input": "string row"},
            ],
        ),
        (
            ".json",
            '{"data":[{"input":"wrapped","output":"yes"},"string row"]}',
            [
                {"input": "wrapped", "output": "yes"},
                {"input": "string row"},
            ],
        ),
        (
            ".jsonl",
            '{"input":"object","output":"yes"}\n"string row"\n\n',
            [
                {"input": "object", "output": "yes"},
                {"input": "string row"},
            ],
        ),
        (".json", "[]", []),
        (".json", '{"data":[]}', []),
        (".jsonl", "\n", []),
    ],
)
def test_control_plane_decoder_matches_locked_freesolo_rows(
    tmp_path, suffix, content, expected
) -> None:
    from freesolo.datasets.records import load_records

    from flash.engine.profiling.dataset_profile import _read_dataset_rows

    path = tmp_path / f"train{suffix}"
    path.write_text(content, encoding="utf-8")

    worker_rows = load_records(path)
    source_examples, control_rows = _read_dataset_rows(path, max_examples=0)

    assert worker_rows == expected
    assert control_rows == expected
    assert source_examples == len(expected)


@pytest.mark.usefixtures("_decoder_sdk_lockstep")
@pytest.mark.parametrize(
    ("suffix", "content"),
    [
        (".json", '{"records":[{"input":"worker rejects this wrapper"}]}'),
        (".json", '"scalar"'),
        (".json", "1"),
        (".json", "{}"),
        (".json", '{"data":"not a list"}'),
        (".json", "[1]"),
        (".json", "{"),
        (".jsonl", "1\n"),
        (".jsonl", "[1]\n"),
        (".jsonl", "{\n"),
    ],
)
def test_control_plane_decoder_rejects_every_file_the_locked_worker_rejects(
    tmp_path, suffix, content
) -> None:
    from freesolo.datasets.records import load_records

    from flash.engine.profiling.dataset_profile import _read_dataset_rows

    path = tmp_path / f"train{suffix}"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="Expect"):
        load_records(path)
    with pytest.raises((PackagedDatasetUnavailable, ValueError)):
        _read_dataset_rows(path, max_examples=0)


@pytest.mark.usefixtures("_decoder_sdk_lockstep")
@pytest.mark.parametrize(
    ("suffix", "content"),
    [
        (".json", '[{"output":"missing"}]'),
        (".json", '[{"input":null}]'),
        (".json", '{"data":[{"input":null}]}'),
        (".jsonl", '{"output":"missing"}\n'),
        (".jsonl", '{"input":null}\n'),
    ],
)
def test_control_plane_and_locked_worker_reject_the_same_missing_inputs(
    tmp_path, suffix, content
) -> None:
    from freesolo.datasets.records import load_records, load_task_examples

    from flash.engine.profiling.dataset_profile import _read_dataset_rows

    path = tmp_path / f"train{suffix}"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="input"):
        load_task_examples(load_records(path))
    with pytest.raises(ValueError, match="input"):
        _read_dataset_rows(path, max_examples=0)


def test_profile_rejects_an_empty_packaged_dataset_at_the_shared_worker_gate(tmp_path) -> None:
    entrypoint = _package(tmp_path, {"dataset/train.json": "[]"})

    with pytest.raises(ValueError, match="every SFT example has an empty completion"):
        _profile(entrypoint)


def test_profile_uses_explicit_records_instead_of_a_packaged_dataset_file(
    tmp_path, monkeypatch
) -> None:
    from flash.engine.profiling import dataset_profile

    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": (
                '{"input":"packaged-one","output":"no"}\n{"input":"packaged-two","output":"no"}\n'
            )
        },
    )
    seen = {}
    real_prepare = dataset_profile.prepare_sft_workload

    def capture_rows(spec, env, **kwargs):
        seen["rows"] = list(env.rows)
        return real_prepare(spec, env, **kwargs)

    monkeypatch.setattr(dataset_profile, "prepare_sft_workload", capture_rows)

    _spec_value, profile = _profile(
        entrypoint,
        params={"records": [{"input": "explicit", "output": "yes"}]},
    )

    assert profile.source_examples == 1
    assert seen["rows"] == [{"input": "explicit", "output": "yes"}]


@pytest.mark.parametrize(
    "value",
    [
        {"question": "what is 2+2?"},
        {"a": 1, "messages": [{"role": "user", "content": "hi"}]},
        {"messages": [{"role": "user", "content": "hi"}]},
        [{"role": "user", "content": "hi"}],
        [1, 2, 3],
        "  padded  ",
        42,
    ],
)
def test_profile_builds_the_prompt_text_the_worker_would_build(value, tmp_path) -> None:
    # the worker reaches the environment through a TaskExample, so the sdk has already rendered
    # .input to text by the time any prompt is built. quoting a python repr, or reading a
    # message-shaped input as a transcript, prices a prompt training never sees. assert against the
    # sdk call the worker makes rather than a restatement of its rules.
    from freesolo.datasets.records import task_example_from_record

    from flash.engine.profiling.dataset_profile import _RawRecordEnvironment

    row = {"input": value, "output": "gold"}
    env = _RawRecordEnvironment(rows=[row], package_root=tmp_path)

    prompt = env.prompt_messages(row)

    assert prompt == [{"role": "user", "content": task_example_from_record(row).input}]


def test_profile_coerces_a_scalar_output_the_way_the_worker_adapter_does(tmp_path) -> None:
    # the gold completion does NOT pass through record normalization: the adapter coerces a scalar
    # output with str(). matching serialize_value here instead would re-introduce the divergence in
    # the other direction.
    from flash.engine.profiling.dataset_profile import _RawRecordEnvironment

    row = {"input": "q", "output": {"answer": 4}}
    env = _RawRecordEnvironment(rows=[row], package_root=tmp_path)

    completion, inferred = env.sft_completion_with_provenance(row)

    assert inferred is True
    assert completion == [{"role": "assistant", "content": str({"answer": 4})}]


def test_profile_uses_explicit_records_when_the_package_has_no_dataset_file(tmp_path) -> None:
    entrypoint = _package(tmp_path, {})

    _spec_value, profile = _profile(
        entrypoint,
        params={"records": [{"input": "explicit", "output": "yes"}]},
    )

    assert profile.source_examples == 1
    assert profile.selected_examples == 1
    assert profile.retained_examples == 1


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([], "records contain no rows"),
        (["not-an-object"], "records must contain JSON object rows"),
    ],
)
def test_profile_refuses_invalid_explicit_records_instead_of_falling_back_to_a_packaged_file(
    tmp_path, records, message
) -> None:
    entrypoint = _package(
        tmp_path,
        {"dataset/train.jsonl": '{"input":"packaged","output":"must not be used"}\n'},
    )

    with pytest.raises(PackagedDatasetUnavailable, match=message):
        _profile(entrypoint, params={"records": records})


def test_profile_refuses_missing_or_unreadable_dataset(tmp_path) -> None:
    entrypoint = _package(tmp_path, {})
    with pytest.raises(PackagedDatasetUnavailable, match=r"dataset/train\.jsonl"):
        _profile(entrypoint)

    unreadable = _package(tmp_path / "bad", {"dataset/train.jsonl": "not json\n"})
    with pytest.raises(PackagedDatasetUnavailable, match="not readable JSON"):
        _profile(unreadable)


def test_profile_includes_the_default_training_contract(tmp_path) -> None:
    with_contract = _package(
        tmp_path / "with-contract",
        {
            "dataset/train.jsonl": '{"input":"prompt","output":"answer"}\n',
            "TRAINING_CONTRACT.md": "follow this fixed training contract",
        },
    )
    without_contract = _package(
        tmp_path / "without-contract",
        {"dataset/train.jsonl": '{"input":"prompt","output":"answer"}\n'},
    )

    _spec_value, contracted = _profile(with_contract)
    _spec_value, raw = _profile(without_contract)

    assert contracted.real_tokens_per_epoch > raw.real_tokens_per_epoch
    assert contracted.supervised_tokens_per_epoch == raw.supervised_tokens_per_epoch


def test_profile_honors_training_contract_precedence(tmp_path) -> None:
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": '{"input":"prompt","output":"answer"}\n',
            "TRAINING_CONTRACT.md": "default contract is intentionally longest",
            "custom-contract.txt": "custom contract",
        },
    )

    _spec_value, default = _profile(entrypoint)
    _spec_value, custom_path = _profile(entrypoint, params={"contract_path": "custom-contract.txt"})
    _spec_value, authored = _profile(
        entrypoint,
        params={
            "contract_text": "x",
            "contract_path": "custom-contract.txt",
        },
    )

    # an explicitly empty contract_path is a string, so the loader passes it through and
    # _load_contract_text loads nothing (loader.py:838-843). quoting the packaged default here
    # would bill a contract training never consumes.
    _spec_value, empty_path = _profile(entrypoint, params={"contract_path": ""})

    assert default.real_tokens_per_epoch > custom_path.real_tokens_per_epoch
    assert custom_path.real_tokens_per_epoch > authored.real_tokens_per_epoch
    assert empty_path.real_tokens_per_epoch < default.real_tokens_per_epoch


def test_training_contract_participates_in_retention_and_truncation(tmp_path) -> None:
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": '{"input":"prompt","output":"answer"}\n',
            "TRAINING_CONTRACT.md": "c" * 256,
        },
    )

    with pytest.raises(ValueError, match="empty completion after"):
        _profile(entrypoint)


def test_profile_prices_the_training_contract_as_the_system_prompt(tmp_path) -> None:
    # a raw record cannot carry its own system turn: the sdk renders input to text, so the contract
    # is always the system message and its size is part of every quoted row.
    entrypoint = _package(
        tmp_path,
        {"dataset/train.jsonl": '{"input":"prompt","output":"answer"}\n'},
    )

    _spec_value, short = _profile(entrypoint, params={"contract_text": "short"})
    _spec_value, long_contract = _profile(
        entrypoint, params={"contract_text": "a slightly longer training contract"}
    )

    assert long_contract.real_tokens_per_epoch > short.real_tokens_per_epoch


def test_profile_refuses_an_oversized_packaged_contract(tmp_path) -> None:
    # the contract is rendered into every row's system turn, so an unbounded one is multiplied
    # across the dataset inside the shared plane process.
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": '{"input":"a","output":"b"}\n',
            "TRAINING_CONTRACT.md": "x" * (256 * 1024 + 1),
        },
    )

    with pytest.raises(PackagedDatasetUnavailable, match="control-plane profiling limit"):
        _profile(entrypoint)


def test_profile_refuses_oversized_inline_records(tmp_path) -> None:
    # inline records bypass the packaged-file size guard: they arrive in the request body.
    entrypoint = _package(tmp_path, {})
    row = {"input": "x" * 4096, "output": "y"}
    records = [dict(row) for _ in range(32 * 1024 * 1024 // 4096 + 8)]

    with pytest.raises(PackagedDatasetUnavailable, match="control-plane profiling limit"):
        _profile(entrypoint, params={"records": records})


def test_profile_treats_a_missing_packaged_contract_path_as_empty(tmp_path) -> None:
    # _resolve_path_arg returns a relative path unresolved when it does not exist, so a missing
    # in-package contract still reads as a bare relative string. the loader turns that into an empty
    # contract and trains; the quote must agree rather than refuse a config the worker accepts.
    entrypoint = _package(tmp_path, {"dataset/train.jsonl": '{"input":"a","output":"b"}\n'})

    _spec, profile = _profile(entrypoint, params={"contract_path": "TRAINING_CONTRACT.md"})

    assert profile.selected_examples == 1


@pytest.mark.parametrize("configured", ["/etc/passwd", "../outside.md"])
def test_profile_refuses_a_contract_path_outside_the_package(tmp_path, configured) -> None:
    # the missing-file allowance above must not weaken containment: absolute and dotdot paths still
    # land outside base_dir and stay refused.
    entrypoint = _package(tmp_path, {"dataset/train.jsonl": '{"input":"a","output":"b"}\n'})

    with pytest.raises(PackagedDatasetUnavailable, match="outside the environment package"):
        _profile(entrypoint, params={"contract_path": configured})


def test_profile_refuses_an_oversized_authored_contract(tmp_path) -> None:
    entrypoint = _package(tmp_path, {"dataset/train.jsonl": '{"input":"a","output":"b"}\n'})

    with pytest.raises(PackagedDatasetUnavailable, match="control-plane profiling limit"):
        _profile(entrypoint, params={"contract_text": "x" * (256 * 1024 + 1)})


def test_profile_rejects_dataset_path_outside_the_package(tmp_path) -> None:
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"input":"outside","output":"no"}\n', encoding="utf-8")
    entrypoint = _package(tmp_path / "package", {})

    with pytest.raises(PackagedDatasetUnavailable, match="readable packaged dataset"):
        _profile(entrypoint, params={"dataset_path": str(outside)})


def test_profile_rejects_contract_path_outside_the_package_before_reading_it(tmp_path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("sensitive host content", encoding="utf-8")
    entrypoint = _package(
        tmp_path / "package",
        {"dataset/train.jsonl": '{"input":"prompt","output":"answer"}\n'},
    )

    with pytest.raises(PackagedDatasetUnavailable, match=r"contract_path.*outside"):
        _profile(entrypoint, params={"contract_path": str(outside)})


def test_profile_tokenizes_raw_record_fields_and_message_shapes_only(tmp_path) -> None:
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": (
                '{"input":"short","output":"answer"}\n'
                '{"input":[{"role":"user","content":"message prompt"}],'
                '"output":{"messages":[{"role":"assistant","content":"message answer"}]}}\n'
            )
        },
    )

    _spec_value, profile = _profile(entrypoint)

    assert profile.source_examples == 2
    assert profile.real_tokens_per_epoch > profile.supervised_tokens_per_epoch > 0


def test_profile_rejects_mixed_message_lists(tmp_path) -> None:
    # only the gold completion is read as a message list; the adapter validates that shape and the
    # profile has to refuse the same rows rather than quote a run the worker will reject.
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": (
                '{"input":"prompt","output":[{"role":"assistant","content":"a"},"invalid"]}\n'
            )
        },
    )

    with pytest.raises(ValueError, match="non-object message entries"):
        _profile(entrypoint)


def _image_package(tmp_path: Path, width: int = 640, height: int = 480) -> Path:
    """An environment package whose single row carries a real packaged png."""
    image_module = pytest.importorskip("PIL.Image")
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": (
                '{"input":"describe","output":"red","image":"dataset/red.png"}\n'
            )
        },
    )
    image_module.new("RGB", (width, height), (200, 10, 10)).save(
        entrypoint.parent / "dataset" / "red.png", format="PNG"
    )
    return entrypoint


_QWEN_TEST_GEOMETRY = ImageGeometry(
    patch_size=16, merge_size=2, min_pixels=65536, max_pixels=16777216
)


@pytest.fixture
def published_geometry(monkeypatch):
    """Pin the published vision geometry so quotes are arithmetic, not a hub round trip."""
    monkeypatch.setattr(
        "flash.engine.profiling.image_tokens.load_image_geometry",
        lambda *_args, **_kwargs: _QWEN_TEST_GEOMETRY,
    )
    return _QWEN_TEST_GEOMETRY


def _image_profile(entrypoint: Path, *, max_context_tokens: int = 4096):
    """Quote one packaged image dataset through the torch-free control-plane path."""
    spec = replace(
        _spec(environment_id=str(entrypoint), model="Qwen/Qwen3.5-9B"),
        train=TrainSpec(epochs=2, batch_size=2, max_context_tokens=max_context_tokens),
    )
    return profile_packaged_sft_dataset(
        spec,
        producer_version="1.2.3",
        tokenizer_loader=lambda _model, _revision: FakeMultimodalTokenizer(),
        packing_support=lambda _model, _revision: ("gdn-hybrid", True),
    )


def _image_profile_from_spec(spec):
    """Quote an explicit spec through the same torch-free path, for failure cases."""
    return profile_packaged_sft_dataset(
        spec,
        producer_version="1.2.3",
        tokenizer_loader=lambda _model, _revision: FakeMultimodalTokenizer(),
        packing_support=lambda _model, _revision: ("gdn-hybrid", True),
    )


def _spy_on_pixel_decodes(monkeypatch) -> list:
    """Record the size of every image whose pixels are actually decoded."""
    from flash.content import multimodal

    decoded: list = []
    real_decode = multimodal._decode_image_bytes

    def count_successful_decode(data):
        image = real_decode(data)
        decoded.append(image.size)
        return image

    monkeypatch.setattr(multimodal, "_decode_image_bytes", count_successful_decode)
    return decoded


def test_control_plane_profiles_image_rows_without_loading_a_processor(
    tmp_path, monkeypatch, published_geometry
) -> None:
    """Image sft is quotable on the torch-free plane, and never touches the VL processor.

    The processor is what needs torch; the quote needs only token counts. The loader is booby
    trapped so the test fails if profiling ever reaches for it again.
    """
    entrypoint = _image_package(tmp_path)
    monkeypatch.setattr(
        "flash.engine.profiling.sft_workload._default_processor_loader",
        lambda *_args: (_ for _ in ()).throw(AssertionError("processor loader reached")),
    )

    # the 640x480 image alone expands to 300 tokens, so the context has to hold it plus the
    # completion or every row truncates to an empty target.
    profile = _image_profile(entrypoint, max_context_tokens=1024)

    assert profile.retained_examples == 1
    # a 640x480 image expands to 300 pad tokens, so the row cannot be near-empty text.
    assert profile.real_tokens_per_epoch > 300
    # image rows are never packed, and the profile has to say so rather than quoting a packed run.
    assert profile.packing_mode == "exact-unpacked"
    assert profile.architecture_mode == "multimodal"


def test_the_quote_grows_with_the_image_the_user_packaged(tmp_path, published_geometry) -> None:
    """The billed token total has to track image size, not just be non-zero.

    A profile that counted one token per image would still "work" -- it would return, and the run
    would train -- while quoting a 1024x768 screenshot at the price of a thumbnail. The whole point
    of the arithmetic is that the number moves.
    """

    def tokens_for(width: int, height: int, directory: str) -> int:
        entrypoint = _image_package(tmp_path / directory, width=width, height=height)
        return _image_profile(entrypoint).real_tokens_per_epoch

    small = tokens_for(56, 56, "small")
    large = tokens_for(1024, 768, "large")
    # 64 pad tokens versus 768: the difference is the image, since every other input is identical.
    assert large - small == 768 - 64


def test_repeated_descriptors_decode_once_but_count_every_occurrence(
    tmp_path, monkeypatch, published_geometry
) -> None:
    image_module = pytest.importorskip("PIL.Image")
    rows = [
        {"input": "same", "output": "answer", "images": ["dataset/a.png"]},
        {
            "input": "same",
            "output": "answer",
            "images": ["dataset/a.png", "dataset/a.png"],
        },
        {
            "input": "same",
            "output": "answer",
            "images": ["dataset/a.png", "dataset/b.png"],
        },
    ]
    entrypoint = _package(
        tmp_path / "images",
        {"dataset/train.jsonl": "".join(json.dumps(row) + "\n" for row in rows)},
    )
    image_module.new("RGB", (56, 56), (200, 10, 10)).save(
        entrypoint.parent / "dataset" / "a.png", format="PNG"
    )
    image_module.new("RGB", (300, 200), (10, 200, 10)).save(
        entrypoint.parent / "dataset" / "b.png", format="PNG"
    )
    successful_decodes = _spy_on_pixel_decodes(monkeypatch)
    image_paths = {
        (entrypoint.parent / "dataset" / "a.png").resolve(),
        (entrypoint.parent / "dataset" / "b.png").resolve(),
    }
    image_reads = []
    real_read_bytes = Path.read_bytes

    def count_image_read(path):
        resolved = path.resolve()
        if resolved in image_paths:
            image_reads.append(resolved)
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", count_image_read)

    profile = _image_profile(entrypoint)

    assert successful_decodes == [(56, 56), (300, 200)]
    assert sorted(path.name for path in image_reads) == ["a.png", "b.png"]
    assert profile.retained_examples == 3
    # 64 + 2*64 + 64 + 70 image pads, plus the fixed text ids for three rows.
    assert profile.real_tokens_per_epoch == 356


def test_profile_rejects_cumulative_unique_decoded_work_before_crossing_decode(
    tmp_path, monkeypatch, published_geometry
) -> None:
    from flash.engine.profiling import image_tokens

    image_module = pytest.importorskip("PIL.Image")
    rows = [
        {"input": "first", "output": "answer", "image": "dataset/a.png"},
        {"input": "second", "output": "answer", "image": "dataset/b.png"},
    ]
    entrypoint = _package(
        tmp_path / "images",
        {"dataset/train.jsonl": "".join(json.dumps(row) + "\n" for row in rows)},
    )
    image_module.new("RGB", (56, 56), (200, 10, 10)).save(
        entrypoint.parent / "dataset" / "a.png", format="PNG"
    )
    image_module.new("RGB", (300, 200), (10, 200, 10)).save(
        entrypoint.parent / "dataset" / "b.png", format="PNG"
    )
    monkeypatch.setattr(image_tokens, "MAX_PROFILE_DECODED_WORK_BYTES", 56 * 56 * 9)
    successful_decodes = _spy_on_pixel_decodes(monkeypatch)

    with pytest.raises(ValueError, match="profile decoded image work"):
        _image_profile(entrypoint)

    assert successful_decodes == [(56, 56)]


def test_an_unreadable_image_is_a_packaging_error_not_a_crash(tmp_path, published_geometry) -> None:
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": (
                '{"input":"describe","output":"red","image":"dataset/red.png"}\n'
            ),
            "dataset/red.png": "this is not a png",
        },
    )
    spec = _spec(environment_id=str(entrypoint), model="Qwen/Qwen3.5-9B")

    with pytest.raises(ValueError, match="not a valid image"):
        _image_profile_from_spec(spec)


def test_image_sft_is_rejected_for_a_model_that_cannot_see_images(tmp_path) -> None:
    # the capability gate stays: the fix makes image sft work on models that advertise it, not on
    # every model.
    entrypoint = _image_package(tmp_path)
    spec = _spec(environment_id=str(entrypoint), model="test/model")

    with pytest.raises(ValueError, match="does not support image-bearing"):
        _image_profile_from_spec(spec)


def test_profile_streams_jsonl_and_tokenizes_only_the_deterministic_max_examples_prefix(
    tmp_path, monkeypatch
) -> None:
    from flash.engine.profiling import dataset_profile

    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": "".join(
                json.dumps({"input": f"prompt-{index}", "output": f"answer-{index}"}) + "\n"
                for index in range(100)
            )
        },
    )
    seen = {}
    real_prepare = dataset_profile.prepare_sft_workload

    def capture_rows(spec, env, **kwargs):
        seen["rows"] = list(env.rows)
        return real_prepare(spec, env, **kwargs)

    monkeypatch.setattr(dataset_profile, "prepare_sft_workload", capture_rows)
    spec = _spec(params={}, environment_id=str(entrypoint))
    spec = replace(spec, train=replace(spec.train, max_examples=3))

    profile = profile_packaged_sft_dataset(
        spec,
        producer_version="1.2.3",
        tokenizer_loader=lambda _model, _revision: FakeTokenizer(),
        packing_support=lambda _model, _revision: ("pure-attention", True),
    )

    assert profile.source_examples == 100
    assert profile.selected_examples == 3
    assert {row["input"] for row in seen["rows"]} == {"prompt-0", "prompt-1", "prompt-2"}


def test_jsonl_profile_never_slurps_the_dataset_into_one_control_plane_string(
    tmp_path, monkeypatch
) -> None:
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": (
                '{"input":"one","output":"alpha"}\n{"input":"two","output":"beta"}\n'
            )
        },
    )
    dataset_path = tmp_path / "dataset/train.jsonl"
    read_text = Path.read_text

    def refuse_dataset_slurp(path, *args, **kwargs):
        if path == dataset_path:
            raise AssertionError("jsonl dataset was slurped with Path.read_text")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", refuse_dataset_slurp)

    _spec_value, profile = _profile(entrypoint)

    assert profile.source_examples == 2


def test_profile_refuses_a_dataset_file_above_the_control_plane_memory_bound(
    tmp_path, monkeypatch
) -> None:
    from flash.engine.profiling import dataset_profile

    entrypoint = _package(
        tmp_path,
        {"dataset/train.jsonl": '{"input":"prompt","output":"answer"}\n'},
    )
    monkeypatch.setattr(dataset_profile, "_MAX_PROFILE_DATASET_BYTES", 1)

    with pytest.raises(PackagedDatasetUnavailable, match="profiling limit"):
        _profile(entrypoint)
