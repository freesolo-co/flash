"""Truthy-aware env-flag parsing — the offline toggle honours 0/false (online)."""

from __future__ import annotations

import pytest

from flash._env import env_flag, flash_skip_net


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on", "ON", " 1 ", "True"])
def test_flash_skip_net_truthy_is_offline(monkeypatch, value):
    monkeypatch.setenv("FLASH_SKIP_NET", value)
    assert flash_skip_net() is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", "", "  ", "anything"])
def test_flash_skip_net_falsy_is_online(monkeypatch, value):
    # Crucially "0" reads as online — a bare os.environ.get() would read the non-empty
    # string "0" as truthy and wrongly stay offline (the bug `make test-live` hit).
    monkeypatch.setenv("FLASH_SKIP_NET", value)
    assert flash_skip_net() is False


def test_flash_skip_net_unset_is_online(monkeypatch):
    monkeypatch.delenv("FLASH_SKIP_NET", raising=False)
    assert flash_skip_net() is False


def test_env_flag_default_applies_only_when_unset(monkeypatch):
    monkeypatch.delenv("FLASH_X", raising=False)
    assert env_flag("FLASH_X") is False
    assert env_flag("FLASH_X", default=True) is True
    # An explicit value always wins over the default (including the false-y "0").
    monkeypatch.setenv("FLASH_X", "0")
    assert env_flag("FLASH_X", default=True) is False
    monkeypatch.setenv("FLASH_X", "1")
    assert env_flag("FLASH_X", default=False) is True
