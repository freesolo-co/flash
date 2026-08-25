"""Exact credential-list coverage for each provider-specific preflight helper."""

from __future__ import annotations

import pytest

import flash.providers.lambda_.client.preflight as lambda_preflight
import flash.providers.runpod.client.preflight as runpod_preflight
import flash.providers.vast.client.preflight as vast_preflight


@pytest.mark.parametrize("require_hf", [False, True])
def test_lambda_preflight_ignores_hf_and_reports_only_its_api_key(monkeypatch, require_hf) -> None:
    """Lambda preflight must leave shared Hugging Face checks to the central preflight layer."""
    monkeypatch.setattr(lambda_preflight, "load_api_key", lambda: None)
    assert lambda_preflight.missing_credentials(require_hf=require_hf) == [
        "  - LAMBDA_API_KEY: the operator's Lambda Cloud API key (for the lambda provider)"
    ]
    monkeypatch.setattr(lambda_preflight, "load_api_key", lambda: "lambda-key")
    assert lambda_preflight.missing_credentials(require_hf=require_hf) == []


@pytest.mark.parametrize("require_hf", [False, True])
def test_vast_preflight_ignores_hf_and_reports_only_its_api_key(monkeypatch, require_hf) -> None:
    """Vast preflight must leave shared Hugging Face checks to the central preflight layer."""
    monkeypatch.setattr(vast_preflight, "load_api_key", lambda: None)
    assert vast_preflight.missing_credentials(require_hf=require_hf) == [
        "  - VAST_API_KEY: the operator's Vast.ai API key (for the vast provider)"
    ]
    monkeypatch.setattr(vast_preflight, "load_api_key", lambda: "vast-key")
    assert vast_preflight.missing_credentials(require_hf=require_hf) == []


@pytest.mark.parametrize(
    ("api_key", "hf_token", "require_hf", "expected_names"),
    [
        (None, None, True, ["RUNPOD_API_KEY", "HF_TOKEN"]),
        ("runpod-key", None, True, ["HF_TOKEN"]),
        (None, "hf-key", True, ["RUNPOD_API_KEY"]),
        ("runpod-key", "hf-key", True, []),
        (None, None, False, ["RUNPOD_API_KEY"]),
        ("runpod-key", None, False, []),
    ],
)
def test_runpod_preflight_independently_checks_api_and_optional_hf_token(
    monkeypatch, api_key, hf_token, require_hf, expected_names
) -> None:
    """RunPod preflight must independently report its API key and the conditionally required HF token."""
    monkeypatch.setattr(runpod_preflight, "load_api_key", lambda: api_key)
    if hf_token is None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HF_TOKEN", hf_token)

    problems = runpod_preflight.missing_credentials(require_hf=require_hf)

    assert [
        name for name in ("RUNPOD_API_KEY", "HF_TOKEN") if any(name in p for p in problems)
    ] == expected_names
