from __future__ import annotations

import re
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
DEV_SERVING_URL = "https://serve-dev.freesolo.co"
DEV_MODAL_URL = "https://freesolo-dev--freesolo-lora-serving.modal.run"
PRODUCTION_SERVING_URL = "https://serve.freesolo.co"


def _read(relative_path: str) -> str:
    return (REPO_DIR / relative_path).read_text(encoding="utf-8")


def _compose_service(compose: str, service_name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(service_name)}:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:\n|^volumes:\n|\Z)",
        compose,
    )
    assert match is not None, f"missing compose service {service_name}"
    return match.group("body")


def test_development_deployment_is_isolated_from_production() -> None:
    workflow = _read(".github/workflows/deploy-modal-dev.yml")

    assert "run: modal deploy --env dev modal_app.py" in workflow
    assert "SERVING_DEPLOYMENT_MODE: development" in workflow
    assert "SERVING_CUSTOM_DOMAIN: serve-dev.freesolo.co" in workflow
    assert "PLATFORM_BACKEND_URL: https://api-dev.freesolo.co" in workflow
    assert f"EXPECTED_MODAL_URL: {DEV_MODAL_URL}" in workflow
    assert f"EXPECTED_SERVING_URL: {DEV_SERVING_URL}" in workflow
    assert 'if [ "$resolved_url" != "$EXPECTED_MODAL_URL" ]; then' in workflow
    assert 'endpoint="${EXPECTED_SERVING_URL}/healthz"' in workflow
    assert PRODUCTION_SERVING_URL not in workflow


def test_development_deployment_uses_a_full_history_in_job_gate() -> None:
    workflow = _read(".github/workflows/deploy-modal-dev.yml")
    trigger = workflow.split("\njobs:", maxsplit=1)[0]

    assert "paths:" not in trigger
    assert "fetch-depth: 0" in workflow
    assert 'git diff --name-only "$BEFORE_SHA" "$CURRENT_SHA"' in workflow
    assert "id: gate" in workflow
    assert "if: steps.gate.outputs.deploy == 'true'" in workflow
    assert "if: steps.gate.outputs.deploy != 'true'" in workflow
    assert "Nothing to deploy" in workflow


def test_development_services_use_the_isolated_serving_url() -> None:
    compose = _read("docker-compose.dev.yml")
    backend_service = _compose_service(compose, "backend")
    flash_service = _compose_service(compose, "flash")

    assert f"FREESOLO_SERVING_URL: {DEV_SERVING_URL}" in backend_service
    assert 'INFISICAL_KEEP: "FREESOLO_SERVING_URL"' in backend_service
    assert f"- FREESOLO_SERVING_URL={DEV_SERVING_URL}" in flash_service
    assert "- INFISICAL_KEEP=FREESOLO_BASE_URL FREESOLO_SERVING_URL" in flash_service
    assert DEV_MODAL_URL not in compose
    assert PRODUCTION_SERVING_URL not in compose


def test_production_deployment_contract_is_unchanged() -> None:
    production_workflow = _read(".github/workflows/deploy-modal.yml")
    production_compose = _read("docker-compose.yml")

    assert "SERVING_CUSTOM_DOMAIN: serve.freesolo.co" in production_workflow
    assert "run: modal deploy modal_app.py" in production_workflow
    assert "modal deploy --env dev modal_app.py" not in production_workflow
    assert DEV_SERVING_URL not in production_workflow
    assert DEV_MODAL_URL not in production_workflow
    assert DEV_SERVING_URL not in production_compose
    assert DEV_MODAL_URL not in production_compose
