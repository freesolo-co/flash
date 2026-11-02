"""Customer-owned serving packaging and authentication boundaries."""

import pathlib
import tomllib


def test_serving_does_not_offer_the_removed_modal_extra() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    extras = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    assert "serve-modal" not in extras


def test_customer_app_authenticates_with_bearer_instead_of_the_plane_key() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    app = (root / "flash" / "serve" / "app" / "http.py").read_text(encoding="utf-8")
    assert "internal-key" not in app.casefold(), "the packaged app must not read the plane key"
    assert '"bearer"' in app.casefold(), "the packaged app must authenticate with bearer"
