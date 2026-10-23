import pytest

from flash.benchmarks.evidence import digest
from flash.benchmarks.schema import (
    Cache,
    Case,
    Complete,
    Configuration,
    Provenance,
    Revisions,
    Target,
)


@pytest.fixture
def case():
    return Case(
        name="sft",
        workload="short_sft",
        command=["python", "driver.py"],
        configuration={},
        revisions=Revisions(
            model="example/model",
            model_revision="a" * 40,
            image="sha256:" + "b" * 64,
            code="c" * 40,
            environment="sha256:" + "d" * 64,
            dataset="sha256:" + "e" * 64,
            evaluator="sha256:" + "f" * 64,
        ),
        target=Target(metric="accuracy", value=0.8, min_samples=10),
    )


@pytest.fixture
def request_data():
    return {
        "seed": 17,
        "cache": "cold",
        "configuration": {"parameters": {}, "seed": 17, "cache": "cold"},
    }


@pytest.fixture
def evidence(case, request_data):
    return [
        Provenance(
            revisions=case.revisions,
            configuration_digest=digest(request_data["configuration"]),
            resolved_configuration=Configuration.model_validate(request_data["configuration"]),
            evidence="worker manifest",
        ),
        Cache(observed="cold", evidence="new worker and empty model cache"),
        Complete(at_s=20.0, outcome="succeeded"),
    ]
