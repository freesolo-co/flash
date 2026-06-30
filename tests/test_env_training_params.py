"""Tests for training-specific environment loader params."""

from __future__ import annotations

from flash.envs.training_params import _FLASH_TRAIN_MAX_EXAMPLES, apply_training_max_examples


def _apply(env_file, marker=None, **params):
    params[_FLASH_TRAIN_MAX_EXAMPLES] = marker
    return apply_training_max_examples(str(env_file), params)


def test_training_max_examples_maps_to_loader_max_examples(tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(max_examples=1024):\n    return None\n")

    assert _apply(env_file) == {"max_examples": None}


def test_training_max_examples_honors_python_source_encoding(tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_bytes(
        b"# coding: latin-1\n# caf\xe9\ndef load_environment(max_examples=1024):\n    return None\n"
    )

    assert _apply(env_file) == {"max_examples": None}


def test_training_max_examples_follows_reexported_loader(tmp_path):
    env_file = tmp_path / "environment.py"
    helper_file = tmp_path / "helper.py"
    env_file.write_text("from helper import load_environment\n")
    helper_file.write_text("def load_environment(max_examples=1024):\n    return None\n")

    assert _apply(env_file) == {"max_examples": None}


def test_training_max_examples_uses_later_local_loader_after_reexport(tmp_path):
    env_file = tmp_path / "environment.py"
    helper_file = tmp_path / "helper.py"
    env_file.write_text(
        "from helper import load_environment\n\n"
        "def load_environment(limit=1024):\n"
        "    return None\n"
    )
    helper_file.write_text("def load_environment():\n    return None\n")

    assert _apply(env_file) == {"limit": None}


def test_training_max_examples_keeps_flash_cap_out_of_loader(tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(max_examples=1024):\n    return None\n")

    assert _apply(env_file, marker=320) == {"max_examples": None}


def test_training_max_examples_prefers_declared_limit(tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text(
        "def load_environment(use_hf=True, limit=None, **kwargs):\n    return None\n"
    )

    assert _apply(env_file, marker=10_178) == {"limit": None}


def test_training_max_examples_clears_all_declared_loader_caps(tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(max_examples=None, limit=1024):\n    return None\n")

    assert _apply(env_file) == {"max_examples": None, "limit": None}


def test_training_max_examples_clears_kwargs_only_loader_caps(tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**kwargs):\n    return None\n")

    assert _apply(env_file) == {"max_examples": None, "limit": None}


def test_training_max_examples_does_not_inject_positional_only_loader_cap(tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(max_examples=1024, /):\n    return None\n")

    assert _apply(env_file) == {}


def test_training_max_examples_omits_positional_only_cap_from_kwargs_fallback(tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(max_examples=1024, /, **kwargs):\n    return None\n")

    assert _apply(env_file) == {"limit": None}


def test_training_max_examples_preserves_explicit_loader_cap_params(tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(max_examples=1024):\n    return None\n")

    assert _apply(env_file, max_examples=25) == {"max_examples": 25}
