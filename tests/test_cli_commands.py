"""In-process CLI coverage: every read/manage command against a fake ApiClient.

`flash login`/`flash train` subprocess flows live in test_cli_managed.py; these tests
drive main() directly so the table rendering, exit codes, and client wiring of
the remaining commands are covered without a server.
"""

from __future__ import annotations

import io
import json
import tomllib
import types

import pytest

import flash.cli.parsing.main as cli
from flash.cli.commands.ops import account as account_commands
from flash.cli.commands.ops import deploy as deploy_commands
from flash.cli.commands.ops import log_follow
from flash.cli.commands.ops import runs as run_commands
from flash.cli.commands.ops import traces as cli_traces
from flash.cli.commands.ops import train as train_commands
from flash.cli.ui import cost as cost_ui
from flash.client.config import DEFAULT_API_URL
from flash.providers._lifecycle.instances.poll import _format_heartbeat


def test_format_heartbeat_appends_named_reward_metrics() -> None:
    heartbeat = {"stage": "rl_step", "step": 4, "reward": 0.65}
    base_line = _format_heartbeat(heartbeat)

    assert base_line == "worker: stage=rl_step step=4 reward=0.650"
    assert _format_heartbeat({**heartbeat, "reward_metrics": {"success": 0.8, "format": 0.5}}) == (
        base_line + " success=0.800 format=0.500"
    )


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.log_text = "hello from the worker\n"

    def me(self) -> dict:
        return {"key_prefix": "freesolo", "email": "t@example.com"}

    def create_run(
        self,
        spec: dict,
        runtime_secrets=None,
        dry_run: bool = False,
        client_train_schema=None,
    ) -> dict:
        self.calls.append(("create_run", spec, runtime_secrets, dry_run, client_train_schema))
        response = {"run_id": "flash-dry", "state": "dry_run", "spec": spec}
        if dry_run:
            response["train_schema_compatibility"] = {"status": "agreement"}
        return response

    def models(self, include_experimental: bool = False) -> list[dict]:
        rows = [
            {
                "id": "Qwen/Qwen3.5-9B",
                "display_name": "Qwen3 0.6B",
                "params": "0.6B dense",
                "algos": ["sft", "grpo"],
                "min_vram_gb": 12,
                "quant": "bf16",
                "recommended_gpu": "RTX 4090",
                "experimental": False,
                "notes": "",
            }
        ]
        if include_experimental:
            rows.append({**rows[0], "id": "test/experimental-model", "experimental": True})
        return rows

    def list_runs(self) -> list[dict]:
        return [
            {
                "run_id": "flash-1",
                "state": "done",
                "cost_usd": 0.25,
                "updated_at": 1700000000.0,
                "spec": {
                    "project": "11111111-1111-4111-8111-111111111111",
                    "model": "Qwen/Qwen3.5-9B",
                    "algorithm": "sft",
                },
            }
        ]

    def get_run(self, run_id: str) -> dict:
        return {
            "run_id": run_id,
            "state": "done",
            "cost_usd": 0.25,
            "error": None,
            "spec": {"model": "Qwen/Qwen3.5-9B"},
        }

    def get_logs(self, run_id: str, offset: int = 0) -> dict:
        logs = self.log_text[max(0, int(offset)) :]
        return {
            "run_id": run_id,
            "logs": logs,
            "offset": len(self.log_text),
            "state": "done",
        }

    def get_worker_output(self, run_id: str) -> dict[str, str]:
        return {"console_sft.txt": "worker stdout line\n"}

    def cancel_run(self, run_id: str) -> dict:
        self.calls.append(("cancel", run_id))
        return {"run_id": run_id, "state": "cancelled"}

    def checkpoints(self, run_id: str) -> list[dict]:
        self.calls.append(("checkpoints", run_id))
        return [{"step": 20}, {"step": 40}]

    def deploy(self, run_id: str, **kwargs) -> dict:
        self.calls.append(("deploy", run_id, kwargs))
        return {
            "run_id": run_id,
            "openai_model": f"flash-{run_id}",
            "endpoint_name": "https://serve.example",
            "openai_base_url": "https://serve.example/v1",
            "state": "deploying",
        }

    def undeploy(self, run_id: str) -> dict:
        self.calls.append(("undeploy", run_id))
        return {"run_id": run_id, "deleted_endpoints": ["live-x"]}

    def deployments(self, timeout: float | None = None) -> list[dict]:
        self.calls.append(("deployments", timeout))
        return [
            {
                "run_id": "flash-1",
                "deployment": {
                    "state": "ready",
                    "endpoint_name": "https://serve.example",
                    "openai_base_url": "https://serve.example/v1",
                },
            }
        ]

    def deployment_for(self, run_id: str, timeout: float | None = None) -> dict | None:
        self.calls.append(("deployment_for", run_id))
        for entry in self.deployments():
            deployment = entry.get("deployment") or {}
            if entry.get("run_id") == run_id.split("/", 1)[0]:
                return {**deployment, "run_id": entry["run_id"]}
        return None

    def deployed_checkpoint(
        self,
        run_id: str,
        timeout: float | None = None,
        *,
        body_deadline: float | None = None,
    ) -> dict | None:
        # the real client reads the run-scoped route, NOT the listing: routing this through
        # `deployments` would let the pre-deploy warning's read land in the rollback-lookup
        # assertions of the `--wait` tests, which instrument that method.
        self.calls.append(("deployed_checkpoint", run_id))
        return None

    def chat(self, run_id: str, messages: list[dict], **_) -> dict:
        self.calls.append(("chat", run_id, messages))
        return {"choices": [{"message": {"content": "42"}}]}

    def chat_stream(self, run_id: str, messages: list[dict], **_):
        self.calls.append(("chat_stream", run_id, messages))
        yield "4"
        yield "2"

    def export(
        self,
        run_id: str,
        *,
        repository: str,
        hf_token: str,
        private: bool = True,
    ) -> dict:
        self.calls.append(("export", run_id, repository, hf_token, private))
        return {
            "run_id": run_id,
            "adapter_id": run_id,
            "repository": repository,
            "url": f"https://huggingface.co/{repository}",
            "source": "org/runs:rl/x/seed0/adapter",
        }


@pytest.fixture(autouse=True)
def project_api(monkeypatch):
    # A Freesolo-HOSTED api_url, because that is the deployment these tests describe. The value
    # used to be an arbitrary placeholder, which was harmless only while nothing read it: commands
    # backed solely by the hosted backend now branch on whether api_url is Freesolo's, so a
    # placeholder domain silently put every one of them on the self-hosted path.
    monkeypatch.setattr(
        "flash.client.config.load_credentials", lambda: ("https://flash.freesolo.co", "fslo-test")
    )
    monkeypatch.setattr(
        account_commands, "load_credentials", lambda: ("https://flash.freesolo.co", "fslo-test")
    )
    monkeypatch.setattr(
        "flash.client.get_project", lambda project_id, api_key: {"id": project_id, "name": "Test"}
    )
    monkeypatch.setattr(
        "flash.client.list_projects",
        lambda api_key: [{"id": "11111111-1111-4111-8111-111111111111", "name": "Test project"}],
    )


@pytest.fixture
def fake_client(monkeypatch) -> _FakeClient:
    client = _FakeClient()
    monkeypatch.setattr(account_commands, "client_from_config", lambda *a, **k: client)
    monkeypatch.setattr(run_commands, "client_from_config", lambda *a, **k: client)
    monkeypatch.setattr(deploy_commands, "client_from_config", lambda *a, **k: client)
    monkeypatch.setattr(train_commands, "client_from_config", lambda *a, **k: client)
    return client


def _run(argv: list[str]) -> int:
    return cli.main(argv)


def test_whoami_prints_identity(fake_client, capsys) -> None:
    assert _run(["whoami"]) == 0
    out = capsys.readouterr().out
    assert "freesolo" in out
    assert "t@example.com" in out
    # Rendered as a human card, not raw JSON.
    assert "{" not in out


def test_project_create_prints_only_returned_id_in_plain_mode(monkeypatch, capsys) -> None:
    seen = {}
    # a Freesolo-hosted api_url: `projects create` only calls the backend on the HOSTED path
    # (a self-hosted plane mints the id locally, see the standalone tests below).
    monkeypatch.setattr(
        account_commands, "load_credentials", lambda: ("https://flash.freesolo.co", "fslo-test")
    )

    def create(name, description, api_key):
        seen.update(name=name, description=description, api_key=api_key)
        return {"id": "33333333-3333-4333-8333-333333333333"}

    monkeypatch.setattr("flash.client.create_project", create)
    assert _run(["projects", "create", " My project ", "--description", " desc "]) == 0
    assert capsys.readouterr().out == "33333333-3333-4333-8333-333333333333\n"
    assert seen == {"name": " My project ", "description": " desc ", "api_key": "fslo-test"}


def test_projects_list_prints_ids_and_names(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "flash.client.list_projects",
        lambda api_key: [
            {
                "id": "11111111-1111-4111-8111-111111111111",
                "name": "Test project",
            }
        ],
    )

    assert _run(["projects", "list"]) == 0
    assert capsys.readouterr().out == ("11111111-1111-4111-8111-111111111111\tTest project\n")


def test_projects_create_uses_plural_group(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "flash.client.create_project",
        lambda name, description, api_key: {"id": "33333333-3333-4333-8333-333333333333"},
    )

    assert _run(["projects", "create", "My project"]) == 0
    assert capsys.readouterr().out == "33333333-3333-4333-8333-333333333333\n"


# --- self-hosted plane: commands backed only by the hosted backend -----------------------------
# All three used to call api.freesolo.co with the operator's plane key, which has no relationship
# with that service -> 401. Same failure `flash env setup` hit on the documented quickstart.

_SELF_HOSTED = ("http://my-plane:8080", "operator-key")


def _self_hosted(monkeypatch) -> None:
    """Put every input the guard reads on a self-hosted plane with no backend configured.

    Each consuming module is patched by name because they `from`-import `load_credentials`,
    binding it at import time -- patching `flash.client.config` alone leaves those bindings on
    the real function, and the command then reads the developer's ambient `~/.flash` config.
    That is environment-dependent, not a test: it passes on a machine already pointed at a local
    plane and fails on a clean checkout, where the default is Freesolo-hosted.

    `FREESOLO_BASE_URL` is cleared for the same reason: the guard reads it as a second input, so
    an operator shell exporting one silently moves every test below onto the configured-backend
    path, where the hosted call is CORRECT and these assertions fail. The tests that want that
    path set the var themselves after calling this.
    """
    monkeypatch.setattr("flash.client.config.load_credentials", lambda: _SELF_HOSTED)
    monkeypatch.setattr(account_commands, "load_credentials", lambda: _SELF_HOSTED)
    monkeypatch.setattr(cli_traces, "load_credentials", lambda: _SELF_HOSTED)
    monkeypatch.delenv("FREESOLO_BASE_URL", raising=False)


def test_projects_create_mints_a_local_id_on_a_self_hosted_plane(monkeypatch, capsys) -> None:
    """The plane accepts any well-shaped uuid under standalone(), so minting one locally IS the
    create. Asserted by CONSUMING the output through the plane's own gate, not by shape alone."""
    _self_hosted(monkeypatch)

    def _unreachable(*a, **k):  # the hosted backend must not be called at all
        raise AssertionError("create_project called on a self-hosted plane")

    monkeypatch.setattr("flash.client.create_project", _unreachable)

    assert _run(["projects", "create", "My project"]) == 0
    minted = capsys.readouterr().out.strip()

    from flash.server.domain.registry.projects import require_project_access

    monkeypatch.setenv("FLASH_STANDALONE", "1")
    assert (
        require_project_access(project_id=minted, key={"auth_kind": "internal"}, authorization=None)
        == minted
    )


def test_projects_create_mints_a_distinct_id_each_time(monkeypatch, capsys) -> None:
    """A fixed id would collide across every project an operator creates, silently merging runs."""
    _self_hosted(monkeypatch)
    monkeypatch.setattr("flash.client.create_project", lambda *a, **k: pytest.fail("hosted call"))

    assert _run(["projects", "create", "one"]) == 0
    assert _run(["projects", "create", "two"]) == 0
    first, second = capsys.readouterr().out.split()
    assert first != second


def test_projects_list_refuses_on_a_self_hosted_plane(monkeypatch, capsys) -> None:
    _self_hosted(monkeypatch)
    monkeypatch.setattr("flash.client.list_projects", lambda *a, **k: pytest.fail("hosted call"))

    assert _run(["projects", "list"]) == 1
    err = capsys.readouterr().err.lower()
    assert "not available on a self-hosted plane" in err
    # names the way forward, so the refusal is actionable rather than a dead end
    assert "projects create" in err


def test_traces_export_refuses_on_a_self_hosted_plane(monkeypatch, capsys) -> None:
    """Unlike `projects create` there is nothing local to substitute: traces are written by the
    freesolo SDK into the hosted backend, so a self-hosted plane has no trace store to read."""
    _self_hosted(monkeypatch)
    # stubbed on `traces`, the binding the command actually calls: patching `flash.client` would
    # leave the real functions in place there, so these fail-fast guards would never fire.
    monkeypatch.setattr(cli_traces, "export_trace_records", lambda *a, **k: pytest.fail("hosted"))
    monkeypatch.setattr(cli_traces, "list_trace_projects", lambda *a, **k: pytest.fail("hosted"))

    assert _run(["traces", "export"]) == 1
    err = capsys.readouterr().err.lower()
    assert "not available on a self-hosted plane" in err
    assert "freesolo sdk" in err


def test_hosted_plane_still_reaches_the_backend(monkeypatch, capsys) -> None:
    """The guard must key on the URL, not disable these commands everywhere."""
    monkeypatch.setattr(
        "flash.client.create_project",
        lambda name, description, api_key: {"id": "44444444-4444-4444-8444-444444444444"},
    )

    assert _run(["projects", "create", "hosted"]) == 0
    assert capsys.readouterr().out == "44444444-4444-4444-8444-444444444444\n"


def test_hosted_plane_still_refuses_when_logged_out(monkeypatch, capsys) -> None:
    """The logged-out guard moved inside the hosted branch when the self-hosted branch was added.
    A hosted caller with no key must still get the login refusal rather than reaching the backend
    with `None` for a bearer token, so assert the relocation kept it on the hosted path.
    """
    # the real logged-out shape: `load_credentials` falls back to DEFAULT_API_URL, so the url is a
    # hosted string and only the key is missing. fabricating a None url would test an unreachable
    # state and keep an impossible branch alive in the guard.
    logged_out = (DEFAULT_API_URL, None)
    monkeypatch.setattr("flash.client.config.load_credentials", lambda: logged_out)
    monkeypatch.setattr(account_commands, "load_credentials", lambda: logged_out)
    monkeypatch.delenv("FREESOLO_BASE_URL", raising=False)

    def _unreachable(*a, **k):
        raise AssertionError("called the backend without a key")

    monkeypatch.setattr("flash.client.create_project", _unreachable)
    monkeypatch.setattr("flash.client.list_projects", _unreachable)

    assert _run(["projects", "create", "no key"]) == 1
    assert f"{cli.CLI_NAME} login" in capsys.readouterr().err
    assert _run(["projects", "list"]) == 1
    assert f"{cli.CLI_NAME} login" in capsys.readouterr().err


def test_configured_backend_keeps_the_hosted_path_on_a_self_hosted_plane(
    monkeypatch, capsys
) -> None:
    """A plane the operator runs can still be pointed at a reachable Freesolo-compatible backend
    via FREESOLO_BASE_URL, which is what these commands actually call. Keying on the plane url
    alone would mint an id that backend's directory never got, turning a clean failure into one
    deferred to project-access validation, and would disable a listing that does work.
    """
    _self_hosted(monkeypatch)
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://freesolo.internal.example")
    monkeypatch.setattr(
        "flash.client.create_project",
        lambda name, description, api_key: {"id": "55555555-5555-4555-8555-555555555555"},
    )

    # the backend is reached, so the id is ITS id -- not a locally minted uuid
    assert _run(["projects", "create", "against a configured backend"]) == 0
    assert capsys.readouterr().out == "55555555-5555-4555-8555-555555555555\n"


def test_configured_backend_keeps_project_listing_available(monkeypatch, capsys) -> None:
    """The refusal is about having no backend, not about who runs the plane."""
    _self_hosted(monkeypatch)
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://freesolo.internal.example")
    sent: list[str] = []

    def _list(api_key):
        sent.append(api_key)
        return [{"id": "66666666-6666-4666-8666-666666666666", "name": "configured"}]

    monkeypatch.setattr("flash.client.list_projects", _list)

    assert _run(["projects", "list"]) == 0
    assert "configured" in capsys.readouterr().out
    # assert WHICH credential travelled: reaching the backend is only correct if the key sent is
    # the one the operator configured. a test that ignores the argument passes a leak too.
    assert sent == ["operator-key"]


def test_a_backend_url_pointing_at_freesolo_does_not_unlock_these_commands(
    monkeypatch, capsys
) -> None:
    """FREESOLO_BASE_URL naming Freesolo's own service is NOT an operator-run backend. Honouring
    it would send the self-hosted plane's operator key to api.freesolo.co as a bearer token --
    the credential disclosure this guard exists to prevent, reopened by the escape hatch added
    for genuinely operator-run backends.
    """
    _self_hosted(monkeypatch)
    monkeypatch.delenv("FREESOLO_API_KEY", raising=False)
    monkeypatch.setattr("flash.client.list_projects", lambda *a, **k: pytest.fail("leaked key"))

    for hosted in ("https://api.freesolo.co", "https://API.Freesolo.CO/", "api.freesolo.co"):
        monkeypatch.setenv("FREESOLO_BASE_URL", hosted)
        assert _run(["projects", "list"]) == 1, hosted
        assert "not available on a self-hosted plane" in capsys.readouterr().err.lower(), hosted


def test_an_ambient_api_key_does_not_unlock_these_commands(monkeypatch, capsys) -> None:
    """FREESOLO_API_KEY cannot prove a hosted account. SELF_HOSTING.md has self-hosters log in
    with the plane-controlling FREESOLO_INTERNAL_KEY, and `cmd_login` reads this same env var as
    the login key, so its value is as likely to be the operator key. Treating its presence as a
    hosted signal ships that credential to api.freesolo.co.
    """
    _self_hosted(monkeypatch)
    monkeypatch.setenv("FREESOLO_API_KEY", "the-plane-operator-key")
    monkeypatch.setattr("flash.client.list_projects", lambda *a, **k: pytest.fail("leaked key"))

    assert _run(["projects", "list"]) == 1
    assert "not available on a self-hosted plane" in capsys.readouterr().err.lower()


def test_blank_backend_url_does_not_count_as_configured(monkeypatch, capsys) -> None:
    """An empty or whitespace value is an unset backend, not a reachable one; treating it as
    configured would restore the 401 this guard exists to prevent."""
    _self_hosted(monkeypatch)
    monkeypatch.setenv("FREESOLO_BASE_URL", "   ")
    # the other hosted-backend signal must be absent, or this asserts nothing about the blank url
    monkeypatch.delenv("FREESOLO_API_KEY", raising=False)
    monkeypatch.setattr("flash.client.list_projects", lambda *a, **k: pytest.fail("hosted call"))

    assert _run(["projects", "list"]) == 1
    assert "not available on a self-hosted plane" in capsys.readouterr().err.lower()


def test_train_cost_requires_explicit_project(tmp_path, capsys) -> None:
    config = tmp_path / "run.toml"
    config.write_text(
        'model = "Qwen/Qwen3.5-9B"\n'
        'algorithm = "grpo"\n'
        "[environment]\n"
        'id = "acme/example-project/example"\n'
        "[train]\n"
        "epochs = 1\n"
        "max_examples = 1\n",
        encoding="utf-8",
    )

    assert _run(["train", str(config), "--cost"]) == 1
    assert "project" in capsys.readouterr().err.lower()


def test_env_setup_maps_inaccessible_project_to_client_error(monkeypatch) -> None:
    from argparse import Namespace

    from flash.cli.commands.env.ops import setup as env_setup
    from flash.client import ApiError, ClientError

    # pinned to a HOSTED url and a key: ownership is only resolved against the backend when the
    # plane is Freesolo's, so leaving this to ambient config would let a self-hosted `~/.flash`
    # take the shape-only branch and pass without ever reaching `get_project`.
    monkeypatch.setattr(
        "flash.client.config.load_credentials", lambda: ("https://flash.freesolo.co", "key-1")
    )
    monkeypatch.setattr(
        "flash.client.get_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ApiError(403, "forbidden")),
    )

    with pytest.raises(ClientError, match="not accessible") as excinfo:
        env_setup._require_setup_project(
            Namespace(project="11111111-1111-4111-8111-111111111111"),
            api_url="https://flash.freesolo.co",
            api_key="key-1",
        )
    assert type(excinfo.value) is ClientError


def test_env_setup_resolves_the_project_locally_on_a_self_hosted_plane(monkeypatch) -> None:
    """A self-hosted plane has no org directory, so the id is validated for shape and accepted.

    Resolving it against ``api.freesolo.co`` sent the operator's plane-root key to a service with
    no relationship to it, which answered 401 -- so `flash env setup`, the first command in the
    SELF_HOSTING.md quickstart, died before writing a file. The plane exposes no project routes at
    all, so there is nothing else to ask; ``flash/server/domain/registry/projects.py`` performs exactly this
    shape-only check under ``standalone()`` when the same run is later submitted.
    """
    from argparse import Namespace

    from flash.cli.commands.env.ops import setup as env_setup

    monkeypatch.setattr(
        "flash.client.config.load_credentials", lambda: ("http://127.0.0.1:8080", "operator-key")
    )
    monkeypatch.setattr(
        "flash.client.get_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not reach the hosted backend from a self-hosted plane")
        ),
    )

    resolved = env_setup._require_setup_project(
        Namespace(project="11111111-1111-4111-8111-111111111111"),
        api_url="http://127.0.0.1:8080",
        api_key="operator-key",
    )
    assert resolved == {"id": "11111111-1111-4111-8111-111111111111"}


_SCAFFOLD_PROJECT = "11111111-1111-4111-8111-111111111111"


def _scaffold(monkeypatch, tmp_path, api_url: str | None, *, turn_mode: str | None = None):
    """Run `flash env setup` in tmp_path against a plane at api_url; return the written files."""
    from argparse import Namespace

    from flash.cli.commands.env.ops import setup as env_setup

    monkeypatch.setattr(
        env_setup,
        "_require_setup_project",
        lambda _args, **_kwargs: {"id": _SCAFFOLD_PROJECT, "name": "Test"},
    )
    monkeypatch.setattr("flash.client.config.load_credentials", lambda: (api_url, "key"))
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    rc = env_setup.cmd_env_setup(
        Namespace(
            project=_SCAFFOLD_PROJECT,
            yes=True,
            turn_mode=turn_mode,
            reasoning=None,
            from_traces=None,
            trace=None,
            force=False,
        )
    )
    assert rc == 0
    return {p.name: p.read_text(encoding="utf-8") for p in tmp_path.rglob("*") if p.is_file()}


def test_env_setup_scaffolds_the_github_form_on_a_self_hosted_plane(monkeypatch, tmp_path) -> None:
    """`flash env push` publishes to Freesolo's managed hub, which a self-hosted plane cannot write.

    The scaffold used to emit `id = ""` plus instructions to run that command regardless of plane,
    so a self-hoster following SELF_HOSTING.md got a config that fails validation and a next step
    that cannot work. `github:owner/repo@ref:path` is the id form their own plane resolves.
    """
    import tomllib

    from flash.cli.commands.env.ops import setup as setup_mod
    from flash.envs.loading.loader import _parse_github_environment_ref

    written = _scaffold(monkeypatch, tmp_path, "https://plane.example.test")

    for name in ("sft.toml", "rl.toml", "opd.toml"):
        assert 'id = "github:OWNER/REPO@REF:environment.py"' in written[name], name
        assert "flash env push --project" not in written[name], name
        # the id must PARSE, or the scaffold just trades one unusable id for another:
        # `github:OWNER/REPO@REF:.` reads naturally and `_normalize_env_path` rejects it.
        env_id = tomllib.loads(written[name])["environment"]["id"]
        assert _parse_github_environment_ref(env_id) is not None, (
            f"{name} scaffolds {env_id!r}, which the environment loader cannot parse"
        )
    # the generated .py files carry the same guidance in their docstrings, with every placeholder
    # filled: one left unrendered is valid python, so nothing raises and the operator reads an
    # internal token where the workflow should be.
    placeholders = set(setup_mod._HOSTED_GUIDANCE) | set(setup_mod._SELF_HOSTED_GUIDANCE)
    assert placeholders, "no guidance placeholders found; the render contract moved"
    for name in ("environment.py", "evaluations.py"):
        assert "flash env push" not in written[name], name
        for placeholder in placeholders | {"PROJECT_UUID"}:
            assert placeholder not in written[name], f"{name} still contains {placeholder}"


def test_scaffolded_ref_is_a_placeholder_not_a_branch_name(monkeypatch, tmp_path) -> None:
    """The scaffolded ref must not name a branch that half of all repos do not have.

    Scaffolding `@main` next to OWNER/REPO makes the ref look like the one token already correct, so
    it survives an edit that replaces the obvious placeholders around it. `gh repo create` still
    publishes `master`, so following the CLI's default and GitHub's default together produces a ref
    that does not exist -- which surfaces as a pinning error naming neither the ref nor the cause.
    """
    written = _scaffold(monkeypatch, tmp_path, "https://plane.example.test")

    for name in ("sft.toml", "rl.toml", "opd.toml", "TRAINING.md"):
        assert "OWNER/REPO@main" not in written[name], (
            f"{name} scaffolds a concrete branch where a placeholder belongs"
        )
        assert "OWNER/REPO@master" not in written[name], name


def test_env_setup_keeps_the_push_workflow_on_the_managed_plane(monkeypatch, tmp_path) -> None:
    """The hosted path is the common one and must be untouched by the self-hosted branch.

    An unset api_url is the managed plane too: it means the built-in default.
    """
    from flash.cli.commands.env.ops import setup as setup_mod

    placeholders = set(setup_mod._HOSTED_GUIDANCE) | set(setup_mod._SELF_HOSTED_GUIDANCE)
    for index, api_url in enumerate(("https://flash.freesolo.co", None)):
        written = _scaffold(monkeypatch, tmp_path / f"case{index}", api_url)

        for name in ("sft.toml", "rl.toml", "opd.toml"):
            assert (
                f"flash env push --project {_SCAFFOLD_PROJECT} --name my-env ." in written[name]
            ), name
            assert 'id = ""' in written[name], name
            assert "github:OWNER" not in written[name], name
        assert "flash env push" in written["environment.py"]
        for name in ("environment.py", "evaluations.py"):
            for placeholder in placeholders | {"PROJECT_UUID"}:
                assert placeholder not in written[name], f"{name} still contains {placeholder}"


def test_env_setup_warns_when_a_hosted_scaffold_is_rerun_on_a_self_hosted_plane(
    monkeypatch, tmp_path, capsys
) -> None:
    """A rerun against the other plane kind leaves every scaffolded file describing the old workflow.

    Setup is idempotent: configs are written only when absent, and the starter .py files only under
    `if not starter_env_exists`. So a hosted-then-self-hosted rerun keeps blank ids and files telling
    the operator to run `flash env push` -- a command their plane cannot use -- while the printed
    next step describes the new plane. The files are deliberately not rewritten, so the warning is
    the whole remedy and has to name all five.
    """
    hosted = _scaffold(monkeypatch, tmp_path, "https://flash.freesolo.co")
    assert "flash env push" in hosted["environment.py"]
    assert "flash env push" in hosted["evaluations.py"]
    capsys.readouterr()

    retained = _scaffold(monkeypatch, tmp_path, "https://plane.example.test")
    warning = capsys.readouterr().err

    assert "still tell you to run `flash env push`, which this plane cannot do" in warning
    assert "no usable [environment] id, which fails validation on any plane" in warning
    assert "`github:OWNER/REPO@REF:PATH`" in warning
    for name in ("environment.py", "evaluations.py"):
        assert name in warning, name
    for name in ("sft.toml", "rl.toml", "opd.toml"):
        assert f"configs/{name}" in warning, name
    # the premise the warning exists for: the files really are retained unrewritten
    assert retained["environment.py"] == hosted["environment.py"]
    assert retained["evaluations.py"] == hosted["evaluations.py"]
    for name in ("sft.toml", "rl.toml", "opd.toml"):
        assert retained[name] == hosted[name], name


def test_env_setup_names_the_self_hosted_id_form_in_both_next_step_renderings(
    monkeypatch, tmp_path, capsys
) -> None:
    """The printed next step is the last thing setup says, and it named the hosted-only command.

    Both branches are asserted because they are separate code paths (`setup.py` and
    `render.env_setup`), and fixing one alone leaves whichever the operator's terminal selects.
    """
    from flash.cli.ui import render

    # force the plain branch: render.styled() decides which of the two runs
    monkeypatch.setattr(render, "styled", lambda: False)
    _scaffold(monkeypatch, tmp_path, "https://plane.example.test")
    plain = capsys.readouterr().out
    assert "next: push this folder to a git repo" in plain
    assert "flash env push" not in plain

    styled = render.env_setup(["environment.py"], "UUID", can_publish=False)
    assert "github:OWNER/REPO@REF:environment.py" in styled
    assert "flash env push" not in styled


def test_env_setup_warns_when_a_self_hosted_scaffold_is_rerun_on_a_hosted_plane(
    monkeypatch, tmp_path, capsys
) -> None:
    """The reverse direction, which carries different retained text and a different remedy.

    `_render_starter` writes distinct guidance into each file, and the retained configs hold
    `github:` ids a hosted plane will not take. Detecting only one of the two would leave the other
    stale beside it -- the same half-fix this warning exists to prevent in the other direction.
    """
    self_hosted = _scaffold(monkeypatch, tmp_path, "https://plane.example.test")
    assert "this plane is self-hosted, so publishing" in self_hosted["environment.py"]
    capsys.readouterr()

    retained = _scaffold(monkeypatch, tmp_path, "https://flash.freesolo.co")
    warning = capsys.readouterr().err

    assert "document a self-hosted plane" in warning
    assert "hosted plane requires managed hub ids" in warning
    assert "Run `flash env push`" in warning
    for name in ("environment.py", "evaluations.py"):
        assert name in warning, name
    for name in ("sft.toml", "rl.toml", "opd.toml"):
        assert f"configs/{name}" in warning, name
        assert retained[name] == self_hosted[name], name


@pytest.mark.parametrize("api_url", ["https://flash.freesolo.co", "https://plane.example.test"])
def test_env_setup_clean_scaffold_has_no_environment_form_warning(
    monkeypatch, tmp_path, capsys, api_url
) -> None:
    """A same-plane rerun stays silent: the detector reads what is on disk, not a plane flag.

    Guards the obvious failure of the two transition tests -- a warning that fired on every rerun
    would satisfy both while making the idempotent path noisy.
    """
    _scaffold(monkeypatch, tmp_path, api_url)
    capsys.readouterr()

    _scaffold(monkeypatch, tmp_path, api_url)

    assert capsys.readouterr().err == ""


def test_env_setup_survives_a_retained_starter_it_cannot_decode(
    monkeypatch, tmp_path, capsys
) -> None:
    """A non-UTF-8 retained starter must not abort setup.

    `# -*- coding: latin-1 -*-` is valid Python, and this path only decides whether to PRINT an
    advisory -- the files are left untouched either way. Reading them strictly turned an operator's
    perfectly runnable environment.py into a UnicodeDecodeError traceback out of `env setup`.
    """
    from argparse import Namespace

    from flash.cli.commands.env.ops import setup as env_setup

    _scaffold(monkeypatch, tmp_path, "https://flash.freesolo.co")
    capsys.readouterr()
    (tmp_path / "environment.py").write_bytes(b"# -*- coding: latin-1 -*-\n# caf\xe9\n")

    # not through `_scaffold`: its read-back of every written file is itself strict UTF-8, so it
    # would raise on the byte this test just wrote and hide whether the CLI survived.
    monkeypatch.setattr(
        env_setup,
        "_require_setup_project",
        lambda _args, **_kwargs: {"id": _SCAFFOLD_PROJECT, "name": "Test"},
    )
    monkeypatch.setattr(
        "flash.client.config.load_credentials", lambda: ("https://plane.example.test", "key")
    )
    monkeypatch.chdir(tmp_path)
    rc = env_setup.cmd_env_setup(
        Namespace(
            project=_SCAFFOLD_PROJECT,
            yes=True,
            turn_mode=None,
            reasoning=None,
            from_traces=None,
            trace=None,
            force=False,
        )
    )

    assert rc == 0  # completed rather than raising UnicodeDecodeError
    # the undecodable file carries no marker, so it is simply not named in the advisory
    assert "environment.py" not in capsys.readouterr().err


def test_env_setup_reports_an_undecodable_config_as_an_error(monkeypatch, tmp_path) -> None:
    """A config that cannot be decoded is a hard error, unlike a starter .py.

    The distinction is deliberate: `[environment] id` is read to classify the plane, so an
    unreadable config cannot be skipped. UnicodeDecodeError is not an OSError, so it needs naming
    explicitly or it escapes as a traceback instead of this message.
    """
    from flash.client import ClientError

    _scaffold(monkeypatch, tmp_path, "https://flash.freesolo.co")
    (tmp_path / "configs" / "sft.toml").write_bytes(b'[environment]\nid = "caf\xe9"\n')

    with pytest.raises(ClientError, match="cannot read existing"):
        _scaffold(monkeypatch, tmp_path, "https://plane.example.test")


def test_env_setup_does_not_let_an_unreadable_env_override_the_turn_flag(
    monkeypatch, tmp_path
) -> None:
    """An environment.py we cannot decode must not silently resolve the turn mode.

    The anchor probe has three states, not two: "carries the marker", "does not", and "cannot be
    read". Collapsing the last into False classified an undecodable multi-turn env as single-turn
    AND then took that guess as authoritative, overriding an explicit --multi-turn without a word.
    """
    from argparse import Namespace
    from pathlib import Path

    from flash.cli.commands.env.ops import setup as env_setup

    starter = tmp_path / "environment.py"
    starter.write_bytes(b"# -*- coding: latin-1 -*-\nclass E(EnvironmentMultiTurn):  # caf\xe9\n")

    assert env_setup._marker_present(starter, "EnvironmentMultiTurn") is None

    multi_turn, _ = env_setup._resolve_turn_mode(
        Namespace(turn_mode="multi", yes=True),
        starter,
        Path(tmp_path / "dataset" / "train.jsonl"),
    )
    assert multi_turn is True  # the flag decides, because the anchor could not be read


def test_env_setup_names_the_plane_side_opd_teacher_setup_when_self_hosted(
    monkeypatch, tmp_path
) -> None:
    """ "nothing to set up" is true only where the platform holds the key.

    `require_teacher_broker_configuration` reads PARASAIL_API_KEY and FLASH_PUBLIC_URL from the
    CONTROL PLANE's environment, so on a self-hosted plane the operator sets both. Telling them
    otherwise moves the discovery to a submit-time failure.
    """
    hosted = _scaffold(monkeypatch, tmp_path / "hosted", "https://flash.freesolo.co")["opd.toml"]
    assert "nothing to set up or export" in hosted

    self_hosted = _scaffold(monkeypatch, tmp_path / "own", "https://plane.example.test")["opd.toml"]
    assert "nothing to set up or export" not in self_hosted
    assert "PARASAIL_API_KEY" in self_hosted
    assert "FLASH_PUBLIC_URL" in self_hosted


def test_env_setup_caveats_that_a_github_id_needs_a_standalone_plane(monkeypatch, tmp_path) -> None:
    """The scaffolded github: id is rejected by an identity-backed self-hosted plane.

    `_require_supported_environment_form` accepts a non-slug id only when `auth.standalone()`, which
    reads the plane's OWN environment. Setup classifies on the API URL and cannot see that, so the
    scaffold writes an id that a non-standalone plane answers with a 400 -- name the requirement
    rather than let it surface as an unexplained submit failure.
    """
    written = _scaffold(monkeypatch, tmp_path, "https://plane.example.test")
    assert "FLASH_STANDALONE=1" in written["sft.toml"]
    # the guide prescribes the same id form, so it carries the same requirement
    assert "FLASH_STANDALONE=1" in written["TRAINING.md"]


def test_training_guide_says_a_private_env_repo_needs_a_plane_side_token(
    monkeypatch, tmp_path
) -> None:
    """The scaffold tells a self-hoster to point at their own repo without naming the token it needs.

    `_github_token` (flash/envs/loading/loader.py) reads `GITHUB_TOKEN` from the resolving process's own
    environment and there is no spec or client field that carries one, so a private repo resolves as
    missing no matter what the operator has exported locally. The guide has to name where the token
    belongs, since the failure surfaces as an unreadable ref rather than an auth error.
    """
    import inspect

    from flash.envs.loading import loader

    # the premise: the token comes from the plane's process env, not from anything the client sends
    source = inspect.getsource(loader._github_token)
    assert 'os.environ.get("GITHUB_TOKEN")' in source

    guide = _scaffold(monkeypatch, tmp_path, "https://plane.example.test")["TRAINING.md"]
    assert "A private repository needs `GITHUB_TOKEN` on the plane, not in your shell" in guide
    assert "forwards no credential of" in guide


def test_training_guide_says_env_eval_rejects_a_github_id(monkeypatch, tmp_path) -> None:
    """The guide prescribes `env eval` further down, but a `github:` id never reaches the suites.

    `_resolve_evaluation_environment` requires `is_managed_environment_slug` and refuses anything
    else before loading anything, so on a standalone plane every `env eval` line in this guide
    deterministically fails. TRAINING.md is static prose and cannot branch on the plane, so the
    caveat has to sit beside the `github:` form it contradicts.
    """
    import inspect

    from flash.cli.commands.env.testing import eval as env_eval

    # the premise: the gate is a managed-slug check, not a soft preference
    source = inspect.getsource(env_eval._resolve_evaluation_environment)
    assert "is_managed_environment_slug" in source

    guide = _scaffold(monkeypatch, tmp_path, "https://plane.example.test")["TRAINING.md"]
    assert "`flash env eval` does not accept a `github:` id" in guide


def test_training_guide_caveats_the_managed_hub_commands(monkeypatch, tmp_path) -> None:
    """The guide says `env push` is unavailable, then later prescribes it without caveat.

    TRAINING.md is static prose read verbatim (only PROJECT_UUID is substituted), so it cannot
    branch on the plane. The self-hosted section near the top is therefore contradicted further down
    by the troubleshooting row and the command reference. Both need the caveat, or a self-hoster who
    lands on either -- which is how a reference gets read -- follows a command their plane cannot run.
    """
    guide = _scaffold(monkeypatch, tmp_path, "https://plane.example.test")["TRAINING.md"]

    assert 'id = "github:OWNER/REPO@REF:environment.py"' in guide
    assert "**Self-hosted:** `env push` targets the managed hub" in guide
    assert "push/pull/delete act on Freesolo's managed hub" in guide


def test_env_setup_still_rejects_a_malformed_project_when_self_hosted(monkeypatch) -> None:
    """Skipping the ownership lookup must not skip the shape check that stands in for it."""
    from argparse import Namespace

    from flash.cli.commands.env.ops import setup as env_setup

    monkeypatch.setattr(
        "flash.client.config.load_credentials", lambda: ("http://127.0.0.1:8080", "operator-key")
    )

    with pytest.raises(ValueError, match="valid UUID"):
        env_setup._require_setup_project(
            Namespace(project="not-a-uuid"),
            api_url="http://127.0.0.1:8080",
            api_key="operator-key",
        )


def test_env_setup_self_hosted_interactive_requires_an_explicit_project(monkeypatch) -> None:
    from argparse import Namespace

    from flash.cli.commands.env.ops import setup as env_setup
    from flash.client import ClientError

    monkeypatch.delenv("FREESOLO_BASE_URL", raising=False)
    monkeypatch.setattr(
        "flash.client.config.load_credentials", lambda: ("http://127.0.0.1:8080", "operator-key")
    )
    monkeypatch.setattr(env_setup, "_setup_interactive", lambda _args: True)
    monkeypatch.setattr(
        "flash.client.list_projects",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a backend-less plane has no project directory to enumerate")
        ),
    )

    with pytest.raises(ClientError, match=r"--project PROJECT_UUID.*no\s+Freesolo backend"):
        env_setup._require_setup_project(
            Namespace(project=""),
            api_url="http://127.0.0.1:8080",
            api_key="operator-key",
        )


def test_env_setup_interactive_lists_projects_from_an_operator_backend(monkeypatch) -> None:
    """A self-hosted plane with its own identity backend HAS a directory, so listing must work.

    Classifying on the control-plane url alone made this topology -- the one SELF_HOSTING.md
    documents for real multi-tenancy: no FLASH_STANDALONE, FREESOLO_BASE_URL pointing at an
    operator-run backend -- indistinguishable from a backend-less plane, so interactive setup
    refused to enumerate a directory that was there the whole time.
    """
    from argparse import Namespace

    from flash.cli.commands.env.ops import setup as env_setup

    project_id = "11111111-1111-4111-8111-111111111111"
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://identity.operator.example")
    monkeypatch.setattr(
        "flash.client.config.load_credentials", lambda: ("http://127.0.0.1:8080", "operator-key")
    )
    monkeypatch.setattr(env_setup, "_setup_interactive", lambda _args: True)
    monkeypatch.setattr(
        "flash.client.list_projects", lambda api_key: [{"id": project_id, "name": "Example"}]
    )
    monkeypatch.setattr(env_setup.render, "select_required", lambda _prompt, _options: project_id)
    monkeypatch.setattr("flash.client.get_project", lambda *_a, **_k: {"id": project_id})

    assert env_setup._require_setup_project(
        Namespace(project=""),
        api_url="http://127.0.0.1:8080",
        api_key="operator-key",
    ) == {"id": project_id}


def test_resolve_project_returns_the_canonical_mapping_and_id(monkeypatch) -> None:
    from flash.client import resolve_project, resolve_project_id

    project_id = "11111111-1111-4111-8111-111111111111"
    project = {"id": project_id, "name": "Canonical Project"}
    calls = []

    def _get_project(selected, api_key):
        calls.append((selected, api_key))
        return project

    monkeypatch.setattr("flash.client.get_project", _get_project)

    assert resolve_project(project_id, "key-1") == project
    assert resolve_project_id(project_id, "key-1") == project_id
    assert calls == [(project_id, "key-1"), (project_id, "key-1")]


def test_supplied_project_is_ownership_checked_against_an_operator_backend(monkeypatch) -> None:
    """An explicit uuid must still be checked for ownership when a backend can answer.

    The url-only branch returned the id after a shape check, so on this topology any well-formed
    uuid -- including another tenant's -- was accepted, and `env eval` could run every generation
    before the upload discovered the project was not reachable.
    """
    from flash.client import ApiError, ClientError, resolve_project_id

    monkeypatch.setenv("FREESOLO_BASE_URL", "https://identity.operator.example")

    def _forbidden(*_args, **_kwargs):
        raise ApiError(403, "forbidden")

    monkeypatch.setattr("flash.client.get_project", _forbidden)

    with pytest.raises(ClientError, match="not accessible"):
        resolve_project_id(
            "11111111-1111-4111-8111-111111111111", "operator-key", "http://127.0.0.1:8080"
        )


def test_env_setup_hosted_interactive_still_selects_a_project(monkeypatch) -> None:
    from argparse import Namespace

    from flash.cli.commands.env.ops import setup as env_setup

    project_id = "11111111-1111-4111-8111-111111111111"
    api_url = "https://flash.freesolo.co"
    seen = {}
    monkeypatch.setattr("flash.client.config.load_credentials", lambda: (api_url, "key-1"))
    monkeypatch.setattr(env_setup, "_setup_interactive", lambda _args: True)
    monkeypatch.setattr(
        "flash.client.list_projects", lambda api_key: [{"id": project_id, "name": "Example"}]
    )
    monkeypatch.setattr(env_setup.render, "select_required", lambda _prompt, _options: project_id)

    def _resolve(selected, api_key, selected_api_url):
        seen.update(selected=selected, api_key=api_key, api_url=selected_api_url)
        return {"id": selected, "name": "Example"}

    monkeypatch.setattr("flash.client.resolve_project", _resolve)

    assert env_setup._require_setup_project(
        Namespace(project=""), api_url=api_url, api_key="key-1"
    ) == {
        "id": project_id,
        "name": "Example",
    }
    assert seen == {"selected": project_id, "api_key": "key-1", "api_url": api_url}


def test_env_setup_interactive_retains_the_selected_project_name(monkeypatch, tmp_path) -> None:
    from argparse import Namespace

    from flash.cli.commands.env.ops import setup as env_setup

    project_id = "11111111-1111-4111-8111-111111111111"
    project_name = "Interactive Project"
    calls = {"load_credentials": 0, "get_project": 0, "fetch_records": 0}

    def _load_credentials():
        calls["load_credentials"] += 1
        return "https://flash.freesolo.co", "key-1"

    def _get_project(selected, _key):
        calls["get_project"] += 1
        return {"id": selected, "name": project_name}

    def _fetch_records(selected, api_key):
        calls["fetch_records"] += 1
        assert (selected, api_key) == (project_id, "key-1")
        return {"records": [{"input": "trace input", "output": "trace output"}]}

    monkeypatch.setattr("flash.client.config.load_credentials", _load_credentials)
    monkeypatch.setattr(env_setup, "_setup_interactive", lambda _args: True)
    monkeypatch.setattr(
        "flash.client.list_projects", lambda _key: [{"id": project_id, "name": "Listed Name"}]
    )
    monkeypatch.setattr(env_setup.render, "select_required", lambda _prompt, _options: project_id)
    monkeypatch.setattr(env_setup.render, "select", lambda *_a, **_k: "yes")
    monkeypatch.setattr("flash.client.get_project", _get_project)
    monkeypatch.setattr(env_setup.traces, "fetch_records", _fetch_records)
    monkeypatch.chdir(tmp_path)

    assert (
        env_setup.cmd_env_setup(
            Namespace(
                project="",
                yes=False,
                turn_mode="single",
                reasoning=False,
                from_traces=None,
                trace=None,
                force=False,
            )
        )
        == 0
    )

    assert calls == {"load_credentials": 1, "get_project": 1, "fetch_records": 1}
    assert json.loads((tmp_path / "dataset/train.jsonl").read_text()) == {
        "input": "trace input",
        "output": "trace output",
    }
    for name in ("sft.toml", "rl.toml", "opd.toml"):
        parsed = tomllib.loads((tmp_path / "configs" / name).read_text())
        assert parsed["wandb"]["project"] == project_name


def test_login_shows_who_you_are(monkeypatch, capsys) -> None:
    # Verify + store are stubbed; login should still surface the identity card itself so the
    # user sees who they are without a separate `flash whoami`. The card is built from the
    # just-verified key via ApiClient, so stub that (not client_from_config).
    monkeypatch.setattr(account_commands, "verify_freesolo_key", lambda *a, **k: None)
    monkeypatch.setattr(account_commands, "save_credentials", lambda *a, **k: None)
    # `kind` mirrors the real /v1/me, which always sends it alongside key_prefix (server/routes/
    # meta.py); only `email` is conditional. Login now checks for that shape before trusting a
    # plane's answer, so a fixture missing it would be asserting on a response no server can send.
    monkeypatch.setattr(
        account_commands,
        "ApiClient",
        lambda *a, **k: type(
            "_C",
            (),
            {
                "me": lambda self: {
                    "kind": "freesolo_api_key",
                    "key_prefix": "freesolo",
                    "email": "t@example.com",
                }
            },
        )(),
    )
    assert _run(["login", "--api-key", "fs-secret-key"]) == 0
    out = capsys.readouterr().out
    assert "logged in to flash" in out
    assert "t@example.com" in out
    assert "fs-secret-key" not in out  # never echo the key


def test_login_failure_is_friendly_and_asks_to_retry(monkeypatch, capsys) -> None:
    from flash.client import ClientError

    def _reject(api_key, base_url=None):
        raise ClientError("freesolo rejected this API key")

    monkeypatch.setattr(account_commands, "verify_freesolo_key", _reject)
    assert _run(["login", "--api-key", "bad-key"]) == 1
    err = capsys.readouterr().err
    assert "login failed" in err
    assert "try again" in err
    assert "founders@freesolo.co" in err
    assert "bad-key" not in err


def test_login_api_key_argument_warns_but_environment_route_does_not(monkeypatch, capsys) -> None:
    monkeypatch.setattr(account_commands, "verify_freesolo_key", lambda *a, **k: None)
    monkeypatch.setattr(account_commands, "save_credentials", lambda *a, **k: None)
    monkeypatch.setattr(account_commands, "_identity_or_none", lambda *a, **k: None)

    assert _run(["login", "--api-key", "fs_secret"]) == 0
    err = capsys.readouterr().err
    assert "visible in process listings" in err
    assert "FREESOLO_API_KEY" in err
    assert "fs_secret" not in err

    monkeypatch.setenv("FREESOLO_API_KEY", "fs_env_secret")
    assert _run(["login"]) == 0
    err = capsys.readouterr().err
    assert "visible in process listings" not in err
    assert "fs_env_secret" not in err


def test_identity_render_is_ascii_locale_safe(monkeypatch) -> None:
    # Under an ASCII / non-UTF-8 stdout, neither a non-ASCII identity value nor our own
    # punctuation may raise UnicodeEncodeError after a login has already succeeded.
    from flash.cli.ui import render

    class _AsciiStdout:
        encoding = "ascii"

        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(render.sys, "stdout", _AsciiStdout())
    card = render.whoami({"key_prefix": "fs", "email": "tëst@example.com"})
    fallback = render.login_ok(None)
    for text in (card, fallback):
        text.encode("ascii")  # raises if any non-ASCII slipped through
    assert f"run `{render.CLI_NAME} whoami`" in fallback


def test_models_table(fake_client, capsys) -> None:
    assert _run(["models", "list"]) == 0
    out = capsys.readouterr().out
    # every catalog model is listed (no experimental/hidden tier)
    assert "Qwen/Qwen3.5-9B" in out
    assert "Qwen/Qwen3.5-9B" in out
    assert "Qwen/Qwen3.8-27B" in out
    assert "Qwen/Qwen3.5-9B" in out
    # only bare model ids, none of the extra per-model detail columns
    assert "2.3B" not in out
    assert "dense" not in out
    assert "(text-only fine-tune)" not in out
    assert "algos=" not in out
    assert "bf16" not in out
    assert "thinking=" not in out


def test_gpus_tip_explains_automatic_default_and_type_pin(fake_client, capsys) -> None:
    assert _run(["gpus"]) == 0
    out = capsys.readouterr().out
    assert "GPU allocation is automatic by default" in out
    assert "cheapest validated class" in out
    assert 'type = "<CLASS>"' in out
    assert "don't pin" not in out
    assert "cannot pin" not in out
    assert "runpod" not in out.lower()
    assert "lambda" not in out.lower()


def _train_config(tmp_path, *, extra_train: str = ""):
    path = tmp_path / "train.toml"
    path.write_text(
        'model = "Qwen/Qwen3.5-9B"\n'
        'project = "11111111-1111-4111-8111-111111111111"\n'
        'algorithm = "sft"\n'
        '[environment]\nid = "owner/project/env"\n'
        f"[train]\nepochs = 1\nmax_examples = 2\n{extra_train}"
    )
    return path


@pytest.mark.parametrize(
    ("compatibility", "expected"),
    [
        (
            {
                "status": "agreement",
                "client_only": [],
                "server_only": [],
                "introduced_in_differences": [],
            },
            "schemas agree exactly",
        ),
        (
            {
                "status": "disagreement",
                "client_only": ["future_knob"],
                "server_only": ["server_knob"],
                "introduced_in_differences": [
                    {"key": "epochs", "client": "0.2.1", "server": "0.2.0"}
                ],
            },
            "client-only keys: future_knob",
        ),
        (None, "unverifiable (legacy server)"),
    ],
)
def test_train_dry_run_keeps_compatibility_on_stderr(
    fake_client, tmp_path, capsys, monkeypatch, compatibility, expected
) -> None:
    monkeypatch.chdir(tmp_path)
    if compatibility is None:
        original_create_run = fake_client.create_run

        def create_run_without_compatibility(*args, **kwargs):
            response = original_create_run(*args, **kwargs)
            response.pop("train_schema_compatibility", None)
            return response

        fake_client.create_run = create_run_without_compatibility
    else:
        original_create_run = fake_client.create_run

        def create_run_with_compatibility(*args, **kwargs):
            response = original_create_run(*args, **kwargs)
            response["train_schema_compatibility"] = compatibility
            return response

        fake_client.create_run = create_run_with_compatibility

    assert _run(["train", str(_train_config(tmp_path)), "--dry-run"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    call = next(call for call in fake_client.calls if call[0] == "create_run")

    assert "train_schema_compatibility" not in payload
    assert expected in captured.err
    assert call[2] is None
    assert call[3] is True
    assert call[4]["authored_keys"] == ["epochs", "max_examples"]
    assert call[1]["train"] == {"epochs": 1, "max_examples": 2}


def test_train_dry_run_attributes_sft_counts_to_the_managed_environment(
    fake_client, tmp_path, capsys, monkeypatch
) -> None:
    original_create_run = fake_client.create_run

    def create_run_with_profile(*args, **kwargs):
        response = original_create_run(*args, **kwargs)
        response["workload_profile"] = {
            "environment_id": "owner/project/env",
            "environment_revision": "a" * 40,
            "source_examples": 125,
        }
        return response

    monkeypatch.setattr(fake_client, "create_run", create_run_with_profile)

    assert _run(["train", str(_train_config(tmp_path)), "--dry-run"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["workload_profile"]["environment_id"] == "owner/project/env"
    assert payload["workload_profile"]["environment_revision"] == "a" * 40
    assert (
        "published environment: owner/project/env @ aaaaaaaaaaaa (125 source rows)" in captured.err
    )
    assert "dataset counts come from this resolved published copy, not local files" in captured.err
    assert "If you expected local dataset edits to be included" in captured.err
    assert (
        "run `flash env push --name NAME --project PROJECT_UUID [path]` again for this managed "
        "environment"
    ) in captured.err


def test_train_dry_run_keeps_inline_records_off_the_published_environment_note(
    fake_client, tmp_path, capsys, monkeypatch
) -> None:
    """inline rows are already authoritative, so the published-copy warning does not apply."""
    original_create_run = fake_client.create_run

    def create_run_with_profile(*args, **kwargs):
        response = original_create_run(*args, **kwargs)
        response["workload_profile"] = {
            "environment_id": "owner/project/env",
            "environment_revision": "a" * 40,
            "source_examples": 2,
        }
        return response

    monkeypatch.setattr(fake_client, "create_run", create_run_with_profile)
    config = tmp_path / "inline.toml"
    config.write_text(
        'model = "Qwen/Qwen3.5-9B"\n'
        'project = "11111111-1111-4111-8111-111111111111"\n'
        'algorithm = "sft"\n'
        '[environment]\nid = "owner/project/env"\n'
        '[environment.params]\nrecords = [{ input = "a", output = "b" }]\n'
        "[train]\nepochs = 1\nmax_examples = 2\n"
    )

    assert _run(["train", str(config), "--dry-run"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["workload_profile"]["source_examples"] == 2
    assert "published environment:" not in captured.err
    assert "SFT dataset counts come from" not in captured.err
    assert "env push" not in captured.err


def test_inline_records_are_not_labelled_a_published_copy_in_cost_rows(monkeypatch) -> None:
    """The cost panel and the provenance note sit within a few lines of each other.

    If only the note learns that the rows came from the request body, the panel above it still
    reads "source rows in published copy" and the quote contradicts itself in one screen.
    """
    from types import SimpleNamespace

    from flash.cli.commands.ops import train

    monkeypatch.setenv("FLASH_STYLE", "0")
    profile = {
        "environment_id": "owner/project/env",
        "environment_revision": "d" * 40,
        "source_examples": 9,
        "retained_examples": 8,
        "selected_examples": 9,
    }
    spec = SimpleNamespace(
        model="Qwen/Qwen3.5-9B",
        environment=SimpleNamespace(params={"records": [{"input": "x"}]}),
    )

    rows = dict(train._sft_cost_rows(spec, profile))

    assert "inline records" in rows["examples"]
    assert "published copy" not in rows["examples"]
    assert rows["env"] == "resolved environment owner/project/env"
    assert "published" not in rows["revision"]


def test_published_rows_keep_their_published_labels(monkeypatch) -> None:
    """The inline branch must not relabel an ordinary published quote."""
    from types import SimpleNamespace

    from flash.cli.commands.ops import train

    monkeypatch.setenv("FLASH_STYLE", "0")
    profile = {
        "environment_id": "owner/project/env",
        "environment_revision": "e" * 40,
        "source_examples": 9,
        "retained_examples": 8,
        "selected_examples": 9,
    }
    spec = SimpleNamespace(model="Qwen/Qwen3.5-9B", environment=SimpleNamespace(params={}))

    rows = dict(train._sft_cost_rows(spec, profile))

    assert "source rows in published copy" in rows["examples"]
    assert rows["env"] == "published environment owner/project/env"
    assert "(published commit)" in rows["revision"]


def test_train_dry_run_sends_declared_runtime_secrets(
    fake_client, tmp_path, capsys, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "train.toml"
    config.write_text(
        'model = "Qwen/Qwen3.5-9B"\n'
        'project = "11111111-1111-4111-8111-111111111111"\n'
        'algorithm = "sft"\n'
        '[environment]\nid = "owner/project/env"\nsecrets = ["SERPAPI_API_KEY"]\n'
        "[train]\nepochs = 1\nmax_examples = 2\n"
    )
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-secret")

    assert _run(["train", str(config), "--dry-run"]) == 0
    capsys.readouterr()
    call = next(call for call in fake_client.calls if call[0] == "create_run")

    assert call[2] == {"SERPAPI_API_KEY": "serp-secret"}
    assert call[3] is True


def test_train_dry_run_authoritative_rejection_keeps_stdout_empty(
    fake_client, tmp_path, capsys, monkeypatch
) -> None:
    from flash.client import ApiError

    def reject(*_args, **_kwargs):
        raise ApiError(
            400,
            "unknown key(s): future_knob. Unsupported authored [train] key(s): "
            "future_knob (minimum released Flash version 0.3.0); "
            "client/server [train] schemas disagree",
        )

    monkeypatch.setattr(fake_client, "create_run", reject)

    assert _run(["train", str(_train_config(tmp_path)), "--dry-run"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "future_knob" in captured.err
    assert "minimum released Flash version 0.3.0" in captured.err
    assert "schemas disagree" in captured.err


def test_train_live_and_dry_run_send_the_same_sparse_spec(fake_client, tmp_path, capsys) -> None:
    config = _train_config(tmp_path)

    assert _run(["train", str(config), "--dry-run"]) == 0
    capsys.readouterr()
    assert _run(["train", str(config), "--background"]) == 0
    capsys.readouterr()
    calls = [call for call in fake_client.calls if call[0] == "create_run"]

    assert calls[0][1] == calls[1][1]
    assert calls[0][1]["train"] == {"epochs": 1, "max_examples": 2}
    assert calls[0][3] is True
    assert calls[0][4] is not None
    assert calls[1][3] is False
    assert calls[1][4] == calls[0][4]


def test_status_runs_and_log_command(fake_client, capsys, monkeypatch) -> None:
    assert _run(["runs", "status", "flash-1"]) == 0
    out = capsys.readouterr().out
    assert "done" in out
    assert "cost_usd" in out

    assert _run(["runs", "status", "flash-1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == "flash-1"

    assert _run(["runs", "list"]) == 0
    out = capsys.readouterr().out
    assert "ALGO" in out
    assert "flash-1" in out
    assert "done" in out
    assert "SFT" in out

    assert _run(["runs", "status", "flash-1", "--follow"]) == 0
    out = capsys.readouterr().out
    assert "cost_usd" in out
    assert "hello from the worker" not in out

    assert _run(["runs", "status", "flash-1", "--follow", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "done"

    statuses = iter(
        [
            {"run_id": "flash-1", "state": "running"},
            {"run_id": "flash-1", "state": "done"},
        ]
    )
    with monkeypatch.context() as patched:
        # scoped: this iterator holds exactly the two statuses this one assertion consumes, so
        # letting it stay installed would starve every later command of a status it needs.
        patched.setattr(fake_client, "get_run", lambda _run_id: next(statuses))
        patched.setattr(run_commands.time, "sleep", lambda _seconds: None)
        assert _run(["runs", "status", "flash-1", "--follow", "--json"]) == 0
        assert [json.loads(line)["state"] for line in capsys.readouterr().out.splitlines()] == [
            "running",
            "done",
        ]

    assert _run(["runs", "log", "flash-1"]) == 0
    out = capsys.readouterr().out
    assert "hello from the worker" in out
    assert "----- console_sft.txt -----" in out
    assert "worker stdout line" in out
    assert "cost_usd" not in out


def test_log_labels_previous_attempt_artifacts_after_the_live_attempt_log(
    fake_client, capsys, monkeypatch
) -> None:
    live_heartbeat = {
        "stage": "rl_step",
        "step": 4,
        "ts": 456.0,
        "attempt": 1,
        "gpu": {"device_name": "NVIDIA B200"},
    }
    fake_client.log_text = _format_heartbeat(live_heartbeat) + "\n"
    monkeypatch.setattr(
        fake_client,
        "get_run",
        lambda _run_id: {
            "run_id": "flash-1",
            "state": "running",
            "remote": {"attempt": 1},
            "last_heartbeat": live_heartbeat,
        },
    )
    monkeypatch.setattr(
        fake_client,
        "get_worker_output",
        lambda _run_id: {
            "console_rl_attempt0.txt": (
                'HEARTBEAT {"stage":"rl_step","step":0,"attempt":0,'
                '"gpu":{"device_name":"NVIDIA H200"}}\n'
            ),
            "error_rl_attempt0.txt": "torch.OutOfMemoryError: CUDA OOM\n",
            "raylogs_rl_attempt0.txt": "raylet exited\n",
        },
    )

    assert _run(["runs", "log", "flash-1"]) == 0
    out = capsys.readouterr().out

    live_line = "worker: stage=rl_step attempt=1 step=4"
    previous_header = (
        "----- console_rl_attempt0.txt (attempt=0, previous attempt; current attempt=1) -----"
    )
    assert live_line in out
    assert previous_header in out
    assert (
        "----- error_rl_attempt0.txt (attempt=0, previous attempt; current attempt=1) -----" in out
    )
    assert (
        "----- raylogs_rl_attempt0.txt (attempt=0, previous attempt; current attempt=1) -----"
        in out
    )
    assert out.index(live_line) < out.index(previous_header)
    # the dead attempt's heartbeat is still printed -- it explains why a retry exists -- but it
    # carries its provenance inline, because a section heading does not survive
    # `grep HEARTBEAT | tail -1`.
    assert 'HEARTBEAT [superseded attempt=0; current attempt=1] {"stage":"rl_step"' in out


def test_log_still_prints_artifacts_when_the_attempt_lookup_fails(
    fake_client, capsys, monkeypatch
) -> None:
    """A failed status read costs the heading, never the artifacts.

    The artifacts hold the traceback the user ran this command to read. Losing them to a lookup
    that only decorates a section header would hide the failure behind an unrelated one.
    """
    from flash.client import ClientError

    def unavailable(_run_id):
        raise ClientError("freesolo is unreachable")

    monkeypatch.setattr(fake_client, "get_run", unavailable)
    monkeypatch.setattr(
        fake_client,
        "get_worker_output",
        lambda _run_id: {"error_rl_attempt0.txt": "torch.OutOfMemoryError: CUDA OOM\n"},
    )

    assert _run(["runs", "log", "flash-1"]) == 0
    out = capsys.readouterr().out
    assert "torch.OutOfMemoryError: CUDA OOM" in out
    # unlabelled against a live attempt, since none could be established -- but still attributed.
    assert "----- error_rl_attempt0.txt (attempt=0) -----" in out


def test_log_reads_no_status_when_there_are_no_artifacts_to_label(
    fake_client, capsys, monkeypatch
) -> None:
    """The status only names a heading, so with no headings it is never worth a request."""
    calls: list[str] = []

    def get_run(_run_id):
        calls.append("get_run")
        return {"run_id": "flash-1", "state": "running", "remote": {"attempt": 0}}

    monkeypatch.setattr(fake_client, "get_run", get_run)
    monkeypatch.setattr(fake_client, "get_worker_output", lambda _run_id: {})

    assert _run(["runs", "log", "flash-1"]) == 0
    capsys.readouterr()
    assert calls == []


def test_log_reads_no_status_and_prints_nothing_when_all_artifacts_are_empty(
    fake_client, capsys, monkeypatch
) -> None:
    """Empty artifact values are not printable sections and must not trigger a status read."""
    fake_client.log_text = ""
    calls: list[str] = []

    def get_run(_run_id):
        calls.append("get_run")
        return {"run_id": "flash-1", "state": "running", "remote": {"attempt": 1}}

    monkeypatch.setattr(fake_client, "get_run", get_run)
    monkeypatch.setattr(
        fake_client,
        "get_worker_output",
        lambda _run_id: {
            "console_rl_attempt1.txt": "",
            "error_rl_attempt1.txt": "",
        },
    )

    assert _run(["runs", "log", "flash-1"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert calls == []


def test_log_labels_artifacts_against_a_retry_that_starts_mid_command(
    fake_client, capsys, monkeypatch
) -> None:
    """Artifact acquisition must precede the status snapshot used for its heading."""
    events: list[str] = []

    def get_worker_output(_run_id):
        events.append("artifacts")
        return {"error_rl_attempt0.txt": "torch.OutOfMemoryError: CUDA OOM\n"}

    def get_run(_run_id):
        events.append("status")
        return {"run_id": "flash-1", "state": "running", "remote": {"attempt": 1}}

    monkeypatch.setattr(fake_client, "get_worker_output", get_worker_output)
    monkeypatch.setattr(fake_client, "get_run", get_run)

    assert _run(["runs", "log", "flash-1"]) == 0
    out = capsys.readouterr().out
    assert events == ["artifacts", "status"]
    assert (
        "----- error_rl_attempt0.txt (attempt=0, previous attempt; current attempt=1) -----" in out
    )


def test_log_prints_partial_log_line_with_newline(fake_client, capsys) -> None:
    fake_client.log_text = "partial log line"
    fake_client.get_worker_output = lambda run_id: {}

    assert _run(["runs", "log", "flash-1"]) == 0
    out = capsys.readouterr().out
    assert out == "partial log line\n"


def test_log_snapshot_reads_one_offset_page_without_status(fake_client, capsys) -> None:
    calls = []
    pages = {
        0: {"run_id": "flash-1", "logs": "first\n", "offset": 6, "state": "running"},
        6: {"run_id": "flash-1", "logs": "second\n", "offset": 13, "state": "done"},
        13: {"run_id": "flash-1", "logs": "", "offset": 13, "state": "done"},
    }

    def get_logs(run_id: str, offset: int = 0) -> dict:
        calls.append(offset)
        return pages[offset]

    fake_client.get_logs = get_logs
    fake_client.get_worker_output = lambda run_id: {}

    assert _run(["runs", "log", "flash-1"]) == 0
    out = capsys.readouterr().out
    assert out == "first\n"
    assert calls == [0]


def _interrupt_the_stream(monkeypatch, fake_client) -> None:
    """Make the next poll raise KeyboardInterrupt, i.e. the user pressed ctrl-c mid-stream."""

    def _boom(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(fake_client, "get_logs", _boom, raising=False)
    monkeypatch.setattr(fake_client, "get_run", _boom, raising=False)


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "CONFIG"],
        ["runs", "log", "flash-1", "--follow"],
        ["runs", "status", "flash-1", "--follow"],
    ],
)
def test_ctrl_c_while_following_says_the_run_is_still_billing(
    argv, fake_client, monkeypatch, tmp_path, capsys
) -> None:
    """Ctrl-C stops the stream, not the run, and the output has to say so.

    The generic handler printed "aborted", which reads as "the run stopped". It did not: the
    remote run keeps going and keeps billing, so the next thing a user does is re-run
    `flash train` and pay for a duplicate. Name the run and the command that actually stops it.
    """
    config = tmp_path / "sft.toml"
    config.write_text(
        "model = 'Qwen/Qwen3.5-9B'\nalgorithm = 'sft'\n"
        "project = '11111111-1111-4111-8111-111111111111'\n"
        "[environment]\nid = 'github:owner/repo@main:env/environment.py'\n"
        "[train]\nepochs = 1\nmax_examples = 8\n"
    )
    monkeypatch.setattr(
        fake_client,
        "create_run",
        lambda *a, **k: {"run_id": "flash-1", "state": "queued"},
        raising=False,
    )
    _interrupt_the_stream(monkeypatch, fake_client)

    assert _run([str(config) if a == "CONFIG" else a for a in argv]) == 130
    err = capsys.readouterr().err
    assert "still going and still billing" in err
    assert f"{cli.CLI_NAME} runs cancel flash-1" in err
    assert f"{cli.CLI_NAME} runs log flash-1 --follow" in err
    # the run was never cancelled on the user's behalf -- detaching is not stopping.
    assert not any(c[0] == "cancel" for c in fake_client.calls)


def test_train_submit_note_warns_that_ctrl_c_keeps_billing(
    fake_client, monkeypatch, tmp_path, capsys
) -> None:
    """The hand-off note has to carry the same warning, before anyone reaches for ctrl-c."""
    config = tmp_path / "sft.toml"
    config.write_text(
        "model = 'Qwen/Qwen3.5-9B'\nalgorithm = 'sft'\n"
        "project = '11111111-1111-4111-8111-111111111111'\n"
        "[environment]\nid = 'github:owner/repo@main:env/environment.py'\n"
        "[train]\nepochs = 1\nmax_examples = 8\n"
    )
    monkeypatch.setattr(
        fake_client,
        "create_run",
        lambda *a, **k: {"run_id": "flash-1", "state": "queued"},
        raising=False,
    )

    assert _run(["train", str(config)]) == 0
    err = capsys.readouterr().err
    assert "keeps billing" in err
    assert f"{cli.CLI_NAME} runs cancel flash-1" in err


def test_follow_logs_shows_tty_spinner_while_waiting(monkeypatch, capsys) -> None:
    class _TTYBuffer(io.StringIO):
        def isatty(self) -> bool:
            return True

    class _WaitingClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.pages = iter(
                [
                    {"run_id": "flash-spin", "logs": "", "offset": 0, "state": "queued"},
                    {
                        "run_id": "flash-spin",
                        "logs": "worker ready\n",
                        "offset": 13,
                        "state": "done",
                    },
                ]
            )
            self.statuses = iter(
                [
                    {"run_id": "flash-spin", "state": "queued"},
                    {"run_id": "flash-spin", "state": "done"},
                ]
            )

        def get_logs(self, run_id: str, offset: int = 0) -> dict:
            return next(self.pages)

        def get_run(self, run_id: str) -> dict:
            return next(self.statuses)

    stderr = _TTYBuffer()
    monkeypatch.setattr(run_commands.sys, "stderr", stderr)
    monkeypatch.setattr(run_commands.time, "sleep", lambda _seconds: None)

    state, printed_any, _ = run_commands._poll_logs(_WaitingClient(), "flash-spin", interval=0.2)

    assert state == "done"
    assert printed_any is True
    assert capsys.readouterr().out == "worker ready\n"
    err = stderr.getvalue()
    assert "following logs for flash-spin (queued)" in err
    assert "\r" in err
    assert err.endswith("\r")


def test_poll_logs_returns_the_live_attempt_from_the_terminal_status(capsys) -> None:
    class _AttemptClient(_FakeClient):
        def get_logs(self, run_id: str, offset: int = 0) -> dict:
            return {"run_id": run_id, "logs": "", "offset": 0, "state": "done"}

        def get_run(self, run_id: str) -> dict:
            return {"run_id": run_id, "state": "done", "remote": {"attempt": 1}}

    result = run_commands._poll_logs(_AttemptClient(), "flash-attempt", interval=0)

    assert result == run_commands._LogPollResult("done", False, 1)
    assert result.live_attempt == 1
    assert capsys.readouterr().out == ""


def test_follow_logs_uses_status_progress_when_log_tail_lags(monkeypatch, capsys) -> None:
    class _TTYBuffer(io.StringIO):
        def isatty(self) -> bool:
            return True

    class _LaggingLogClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.statuses = iter(
                [
                    {
                        "run_id": "flash-lag",
                        "state": "running",
                        "last_heartbeat": {"stage": "rl_step", "step": 42},
                        "realized_cost_usd": 1.23456,
                    },
                    {
                        "run_id": "flash-lag",
                        "state": "done",
                        "last_heartbeat": {"stage": "rl_train_done"},
                        "realized_cost_usd": 1.5,
                    },
                ]
            )

        def get_logs(self, run_id: str, offset: int = 0) -> dict:
            # Stale/lossy log stream: no bytes and a non-terminal page state forever.
            return {"run_id": run_id, "logs": "", "offset": 0, "state": "running"}

        def get_run(self, run_id: str) -> dict:
            return next(self.statuses)

    stderr = _TTYBuffer()
    monkeypatch.setattr(run_commands.sys, "stderr", stderr)
    monkeypatch.setattr(run_commands.time, "sleep", lambda _seconds: None)

    state, printed_any, _ = run_commands._poll_logs(_LaggingLogClient(), "flash-lag", interval=0.2)

    assert state == "done"
    assert printed_any is False
    assert capsys.readouterr().out == ""
    err = stderr.getvalue()
    assert "stage=rl_step" in err
    assert "step=42" in err
    assert "realized_cost=$1.2346" in err


def test_follow_logs_prints_heartbeat_metrics_once_per_step(monkeypatch, capsys) -> None:
    metric_one = {
        "step": 1,
        "reward": 0.75,
        "reward_std": 0.12,
        "grad_norm": 1.5,
        "kl": 0.03,
        "entropy": 0.82,
        "frac_reward_zero_std": 0.25,
        "mean_completion_tokens": 48.5,
        "truncation_rate": 0.125,
        "discarded_rollouts": 1,
        "max_completion_tokens": 256,
    }
    metric_two = {
        "step": 2,
        "reward": 0.8,
        "reward_std": 0.1,
        "grad_norm": 1.25,
        "kl": None,
        "entropy": 0.79,
        "frac_reward_zero_std": 0.0,
        "mean_completion_tokens": 51.0,
        "truncation_rate": 0.25,
        "discarded_rollouts": 2,
        "max_completion_tokens": 256,
    }

    class _MetricClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.statuses = iter(
                [
                    {
                        "run_id": "flash-metrics",
                        "state": "running",
                        "last_heartbeat": {"stage": "rl_step", "metrics_last": [metric_one]},
                    },
                    {
                        "run_id": "flash-metrics",
                        "state": "running",
                        "last_heartbeat": {
                            "stage": "rl_step",
                            "metrics_last": [metric_one, metric_two],
                        },
                    },
                    {
                        "run_id": "flash-metrics",
                        "state": "done",
                        "last_heartbeat": {"stage": "rl_step", "metrics_last": [metric_two]},
                    },
                ]
            )

        def get_logs(self, run_id: str, offset: int = 0) -> dict:
            return {"run_id": run_id, "logs": "", "offset": 0, "state": "running"}

        def get_run(self, run_id: str) -> dict:
            return next(self.statuses)

    monkeypatch.setattr(run_commands.time, "sleep", lambda _seconds: None)

    state, printed_any, _ = run_commands._poll_logs(_MetricClient(), "flash-metrics", interval=0.2)

    assert state == "done"
    assert printed_any is False
    metric_lines = [
        line for line in capsys.readouterr().err.splitlines() if line.startswith("step=")
    ]
    assert metric_lines == [
        (
            "step=1 reward=0.75 reward_std=0.12 grad_norm=1.5 kl=0.03 entropy=0.82 "
            "frac_zero_std=0.25 comp_len=48.5 trunc=0.125 discarded=1 max_comp_tokens=256"
        ),
        (
            "step=2 reward=0.8 reward_std=0.1 grad_norm=1.25 entropy=0.79 frac_zero_std=0 "
            "comp_len=51 trunc=0.25 discarded=2 max_comp_tokens=256"
        ),
    ]


def test_log_follow_metric_dedup_is_attempt_aware() -> None:
    from flash.cli.commands.ops.log_follow import _log_follow_metric_rows

    seen = set()
    attempt_one = {
        "last_heartbeat": {
            "attempt": 1,
            "metrics_last": [{"step": 7, "reward": 0.5}],
        }
    }
    attempt_two = {
        "last_heartbeat": {
            "attempt": 2,
            "metrics_last": [{"step": 7, "reward": 0.6}],
        }
    }

    assert _log_follow_metric_rows(attempt_one, seen) == ["step=7 reward=0.5"]
    assert _log_follow_metric_rows(attempt_one, seen) == []
    assert _log_follow_metric_rows(attempt_two, seen) == ["step=7 reward=0.6"]


def test_cancel_surfaces_surviving_checkpoints(fake_client, capsys) -> None:
    """`state=cancelled` + adapter_ref=null + cost=0 reads as discardable, yet the per-step
    deployable checkpoints streamed before the cancel survive it — the cancel output must say
    so (on stderr in the plain path, keeping the stdout JSON machine-readable)."""
    import json as _json

    assert _run(["runs", "cancel", "flash-1"]) == 0
    assert ("checkpoints", "flash-1") in fake_client.calls
    out, err = capsys.readouterr()
    assert _json.loads(out)["state"] == "cancelled"  # stdout stays pure JSON in the plain path
    assert "2 deployable checkpoint(s) survive this cancel" in err
    assert f"{cli.CLI_NAME} runs checkpoint flash-1" in err
    assert (
        f"{cli.CLI_NAME} models deploy flash-1/step-40" in err
    )  # points at the newest surviving step


def test_cancel_hint_is_best_effort_when_checkpoint_listing_fails(
    fake_client, capsys, monkeypatch
) -> None:
    """The surviving-checkpoints lookup must never break `flash runs cancel` itself."""

    def boom(run_id):
        raise RuntimeError("backend hiccup")

    monkeypatch.setattr(fake_client, "checkpoints", boom)
    assert _run(["runs", "cancel", "flash-1"]) == 0
    out, err = capsys.readouterr()
    assert '"state": "cancelled"' in out
    assert "deployable checkpoint" not in err


def test_cancel_hint_survives_malformed_checkpoint_shape(fake_client, capsys, monkeypatch) -> None:
    """A checkpoint dict missing 'step' (or carrying a non-orderable value) must NOT crash a cancel
    that already succeeded — the max(step) hint is best-effort. A recoverable step still drives the
    deploy example; when none is recoverable the example is simply dropped (no crash, no bogus step)."""
    monkeypatch.setattr(
        fake_client, "checkpoints", lambda run_id: [{"no_step": 1}, {"step": None}, {"step": 7}]
    )
    assert _run(["runs", "cancel", "flash-1"]) == 0  # did not raise on the malformed entries
    out, err = capsys.readouterr()
    assert '"state": "cancelled"' in out
    assert "3 deployable checkpoint(s) survive this cancel" in err
    assert f"{cli.CLI_NAME} models deploy flash-1/step-7" in err  # max of the RECOVERABLE steps

    monkeypatch.setattr(fake_client, "checkpoints", lambda run_id: [{"no_step": 1}])
    assert _run(["runs", "cancel", "flash-1"]) == 0
    _, err2 = capsys.readouterr()
    assert "1 deployable checkpoint(s) survive this cancel" in err2
    assert "flash deploy" not in err2


def test_cancel_deploy_undeploy_deployments(fake_client, capsys) -> None:
    assert _run(["runs", "cancel", "flash-1"]) == 0
    assert ("cancel", "flash-1") in fake_client.calls

    assert _run(["models", "deploy", "flash-1/final"]) == 0
    assert ("deploy", "flash-1/final", {"dry_run": False}) in fake_client.calls

    assert _run(["models", "deploy", "flash-1/step-40"]) == 0
    assert ("deploy", "flash-1/step-40", {"dry_run": False}) in fake_client.calls
    err = capsys.readouterr().err
    assert "flash models undeploy flash-1/final`" in err
    assert "flash models undeploy flash-1/step-40`" in err

    assert _run(["models", "deployments"]) == 0
    deployments_out = capsys.readouterr().out
    assert "flash-1" in deployments_out
    assert "CHECKPOINT ID" in deployments_out
    assert "OPENAI BASE URL" in deployments_out
    assert "https://serve.example/v1" in deployments_out

    assert _run(["models", "undeploy", "flash-1/final"]) == 0
    assert ("undeploy", "flash-1/final") in fake_client.calls


def test_deployments_json_passes_server_rows_through(fake_client, capsys) -> None:
    assert _run(["models", "deployments", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows == [
        {
            "run_id": "flash-1",
            "deployment": {
                "state": "ready",
                "endpoint_name": "https://serve.example",
                "openai_base_url": "https://serve.example/v1",
            },
        }
    ]


def test_deployments_json_empty_list(fake_client, monkeypatch, capsys) -> None:
    monkeypatch.setattr(fake_client, "deployments", list)
    assert _run(["models", "deployments", "--json"]) == 0
    assert capsys.readouterr().out.strip() == "[]"


def test_deployments_without_base_url_renders_placeholder(fake_client, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        fake_client,
        "deployments",
        lambda: [{"run_id": "flash-old", "deployment": {"state": "ready"}}],
    )
    assert _run(["models", "deployments"]) == 0
    out = capsys.readouterr().out
    assert "flash-old" in out
    assert "OPENAI BASE URL" in out
    assert "  -" in out


def test_chat_sends_message_and_prints_reply(fake_client, capsys) -> None:
    assert _run(["models", "chat", "flash-1/final", "-m", "What is 6*7?"]) == 0
    assert "42" in capsys.readouterr().out
    assert fake_client.calls[-1][0] == "chat_stream"


def test_chat_checkpoint_ref_is_forwarded_unchanged(fake_client) -> None:
    target = "flash-1/step-40"

    assert _run(["models", "chat", target, "-m", "What is 6*7?"]) == 0
    assert fake_client.calls[-1][0] == "chat_stream"
    assert fake_client.calls[-1][1] == target


def test_chat_stream_uses_permanent_checkpoint_without_legacy_capability_preflight(
    monkeypatch,
) -> None:
    from flash.client import ApiClient

    class Response:
        def __init__(self):
            self.headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    client = ApiClient("https://flash.test")
    health_calls = 0

    def health():
        nonlocal health_calls
        health_calls += 1
        return {"capabilities": []}

    monkeypatch.setattr(client, "health", health)
    monkeypatch.setattr(
        "flash.client.http.urllib.request.urlopen", lambda *args, **kwargs: Response()
    )

    assert list(client.chat_stream("flash-1/step-3", [])) == ["ok"]
    assert health_calls == 0


def test_chat_accepts_permanent_checkpoint(fake_client) -> None:
    checkpoint_id = "flash-1/step-40"
    assert _run(["models", "chat", checkpoint_id, "-m", "What is 6*7?"]) == 0
    assert fake_client.calls[-1][0] == "chat_stream"
    assert fake_client.calls[-1][1] == checkpoint_id


def test_chat_system_flag_prepends_system_message(fake_client) -> None:
    """--system gives evals training-prompt parity without calling the HTTP API directly."""
    assert (
        _run(["models", "chat", "flash-1/final", "-m", "What is 6*7?", "--system", "be brief"]) == 0
    )
    _, _, messages = fake_client.calls[-1]
    assert messages == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "What is 6*7?"},
    ]


def test_chat_without_system_flag_sends_user_message_only(fake_client) -> None:
    assert _run(["models", "chat", "flash-1/final", "-m", "What is 6*7?"]) == 0
    _, _, messages = fake_client.calls[-1]
    assert messages == [{"role": "user", "content": "What is 6*7?"}]


@pytest.mark.parametrize("flag", ["--enable-thinking", "--disable-thinking"])
def test_chat_does_not_expose_thinking_override_flags(fake_client, flag) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run(["models", "chat", "flash-1/final", "-m", "What is 6*7?", flag])
    assert excinfo.value.code == 2


def test_env_setup_scaffolds_grpo_and_sft_configs(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    assert _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 0

    assert (tmp_path / "environment.py").is_file()
    evaluations = tmp_path / "evaluations.py"
    assert evaluations.is_file()
    evaluations_text = evaluations.read_text()
    assert "load_evaluations(environment=None)" in evaluations_text
    assert "self.environment.reward(response, example)" in evaluations_text
    # the scaffold has to teach the invocation argparse actually accepts. it advertised
    # `env eval TARGET .` back when a local directory was the input, which is now a usage error,
    # so the header would have sent every new user's first evaluation into an exit-2.
    assert "flash env eval TARGET`" in evaluations_text
    assert "flash env eval TARGET ." not in evaluations_text

    class StarterEnvironment:
        def reward(self, response, example):
            assert response == "12"
            assert example["output"] == "12"
            return 0.75

        def grade(self, response, example):
            return True

    from flash.envs.meta.evaluations import load_evaluation_suites

    starter_suite = load_evaluation_suites(tmp_path, environment=StarterEnvironment())[0]
    starter_case = starter_suite.cases()[0]
    starter_scored = starter_suite.score(starter_case, "12")
    assert starter_scored.score == 0.75
    assert starter_scored.passed is True

    dataset = tmp_path / "dataset/train.jsonl"
    assert dataset.is_file()
    assert not (tmp_path / "datasets").exists()
    assert '"input":"What is 2 + 2?"' in dataset.read_text()
    grpo = tmp_path / "configs/rl.toml"
    sft = tmp_path / "configs/sft.toml"
    assert grpo.is_file()
    assert sft.is_file()
    assert not (tmp_path / "configs/endpoints.toml").exists()
    assert 'project = "11111111-1111-4111-8111-111111111111"' in grpo.read_text()
    assert 'algorithm = "grpo"' in grpo.read_text()
    assert "epochs = 1" in grpo.read_text()
    assert "max_examples = 2" in grpo.read_text()
    assert "cheapest fitting managed class" in grpo.read_text()
    assert "private environment-scoped repo" in grpo.read_text()
    assert 'project = "11111111-1111-4111-8111-111111111111"' in sft.read_text()
    assert 'algorithm = "sft"' in sft.read_text()
    assert "epochs = 1" in sft.read_text()
    assert "max_examples = 2" in sft.read_text()
    assert "cheapest fitting managed class" in sft.read_text()
    assert "private environment-scoped repo" in sft.read_text()
    opd = tmp_path / "configs/opd.toml"
    assert opd.is_file()
    opd_text = opd.read_text()
    assert 'project = "11111111-1111-4111-8111-111111111111"' in opd_text
    assert 'algorithm = "opd"' in opd_text
    assert "epochs = 1" in opd_text
    assert "max_examples = 2" in opd_text
    # the teacher key is platform-managed: the scaffold neither declares it as a secret nor tells
    # the user to export it, so the generated config must not mention the provider credential.
    assert "PARASAIL_API_KEY" not in opd_text
    assert "secrets" not in opd_text
    assert "platform-managed" in opd_text
    # single-turn opd runs fine, so it carries NO multi-turn "fails fast" warning
    assert "fail fast" not in opd_text
    for name, suffix in (("sft.toml", "sft"), ("rl.toml", "grpo"), ("opd.toml", "opd")):
        parsed = tomllib.loads((tmp_path / "configs" / name).read_text())
        assert parsed["wandb"] == {
            "project": "Test",
            "run_name": f"{tmp_path.name}-{suffix}",
        }
        assert "wandb" not in parsed.get("train", {}), name
        assert "wandb" not in parsed.get("environment", {}), name
    training = tmp_path / "TRAINING.md"
    assert training.is_file()
    training_text = training.read_text(encoding="utf-8")
    assert "how to actually improve a model with Flash" in training_text
    assert "## Using Flash" in training_text  # end-to-end library usage, not just conventions
    assert "## Common Flash issues and mitigations" in training_text
    assert "GPU selection is not what you expected" in training_text
    assert "response_text.thinking" in training_text
    assert "Qwen3.5 thinking multi-turn SFT" in training_text
    assert "`grades_episodes = True`" in training_text
    assert "`score(case, response, state)`" in training_text
    assert "longest shared token prefix" in training_text
    assert "flash env pull your-org/your-project/my-env" in training_text
    assert "private environment-scoped repo" in training_text
    assert 'project = "11111111-1111-4111-8111-111111111111"' in training_text
    assert "flash runs checkpoint <run-id>" in training_text
    assert "flash models deployments" in training_text
    assert (
        "flash models export --adapter-id <run-id>/final --repository <you>/<repo>" in training_text
    )
    assert (
        "flash env push --project 11111111-1111-4111-8111-111111111111 --name my-env ."
        in training_text
    )
    assert "HF_TOKEN" in training_text
    assert "runpod" not in training_text.lower()
    assert "lambda" not in training_text.lower()
    out = capsys.readouterr().out
    assert "evaluations.py" in out
    assert "dataset/train.jsonl" in out
    assert "configs/rl.toml" in out
    assert "configs/opd.toml" in out
    assert "TRAINING.md" in out


def test_env_setup_wandb_metadata_escapes_project_and_folder_names(monkeypatch, tmp_path) -> None:
    project_name = 'Project "quoted" 🚀\nsecond line'
    folder = tmp_path / 'environment "quoted" 🧪'
    folder.mkdir()
    monkeypatch.setattr(
        "flash.client.get_project",
        lambda project_id, _api_key: {"id": project_id, "name": project_name},
    )
    monkeypatch.chdir(folder)

    assert _run(["env", "setup", "--project", _SCAFFOLD_PROJECT]) == 0

    for name, suffix in (("sft.toml", "sft"), ("rl.toml", "grpo"), ("opd.toml", "opd")):
        config_text = (folder / "configs" / name).read_text()
        assert "🚀" in config_text
        assert "🧪" in config_text
        parsed = tomllib.loads(config_text)
        assert parsed["wandb"] == {
            "project": project_name,
            "run_name": f"{folder.name}-{suffix}",
        }


@pytest.mark.parametrize(
    "project_name",
    [
        None,
        42,
        "",
        "   ",
        "invalid/project",
        "invalid\\project",
        "invalid#project",
        "invalid?project",
        "invalid%project",
        "invalid:project",
        "x" * 129,
    ],
)
def test_env_setup_falls_back_to_project_id_for_unusable_wandb_project_names(
    monkeypatch, tmp_path, project_name
) -> None:
    monkeypatch.setattr(
        "flash.client.get_project",
        lambda project_id, _api_key: {"id": project_id, "name": project_name},
    )
    monkeypatch.chdir(tmp_path)

    assert _run(["env", "setup", "--project", _SCAFFOLD_PROJECT]) == 0

    for name in ("sft.toml", "rl.toml", "opd.toml"):
        parsed = tomllib.loads((tmp_path / "configs" / name).read_text())
        assert parsed["wandb"]["project"] == _SCAFFOLD_PROJECT


def test_env_setup_does_not_backfill_wandb_into_existing_configs(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    project = ["--project", _SCAFFOLD_PROJECT]
    assert _run(["env", "setup", *project]) == 0

    retained = {}
    for name in ("sft.toml", "rl.toml", "opd.toml"):
        path = tmp_path / "configs" / name
        before, rest = path.read_text().split("[wandb]\n", 1)
        _, after = rest.split("[environment]\n", 1)
        retained[name] = before + "[environment]\n" + after
        path.write_text(retained[name])

    assert _run(["env", "setup", *project]) == 0

    for name, expected in retained.items():
        path = tmp_path / "configs" / name
        assert path.read_text() == expected
        assert "wandb" not in tomllib.loads(expected)


def test_env_setup_does_not_overwrite_existing_evaluations(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "evaluations.py"
    existing.write_text("# keep this evaluation sidecar\n")

    assert _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 0

    assert existing.read_text() == "# keep this evaluation sidecar\n"


def test_env_setup_does_not_add_starter_evaluations_to_a_custom_environment(
    monkeypatch, tmp_path, capsys
) -> None:
    # the arithmetic starter scorer calls the neighboring environment's reward with its own
    # example, so adding it on a rerun makes an unrelated custom environment fail or score nonsense.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "environment.py").write_text("def load_environment(): return object()\n")

    assert _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 0

    assert not (tmp_path / "evaluations.py").exists()
    assert "evaluations.py" not in capsys.readouterr().out
    # the rest of the scaffold still lands: this is about the suite, not about refusing to run.
    assert (tmp_path / "configs/rl.toml").is_file()


def test_env_setup_multi_turn_scaffolds_opd_for_multi_turn(monkeypatch, tmp_path, capsys) -> None:
    """`flash env setup --multi-turn` scaffolds all three configs (sft/rl/opd). opd now supports
    multi-turn (it rolls out each episode and distils every assistant turn), so the multi-turn opd.toml
    and the starter env docstring must NOT warn it is single-turn only / fails fast."""
    monkeypatch.chdir(tmp_path)

    assert (
        _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111", "--multi-turn"])
        == 0
    )

    env_py = (tmp_path / "environment.py").read_text()
    assert "EnvironmentMultiTurn" in env_py  # genuinely a multi-turn scaffold
    evaluations_text = (tmp_path / "evaluations.py").read_text()
    assert "load_evaluations(environment=None)" in evaluations_text
    # the multi-turn scaffold gets its own suite. `reward(response, example)` with no episode state
    # sends `_score_one` down the single-turn branch (flash/envs/loading/adapter.py:237-243), which grades an
    # EMPTY transcript -- the arithmetic suite scored this guess-the-number env 1.0 on "12".
    assert "self.environment.reward(response, example)" not in evaluations_text
    assert "`score(self, case, response, state)`" in evaluations_text
    assert "passes the resulting transcript in `state`" in evaluations_text
    assert "step_episode" in evaluations_text
    assert "`messages` already ends with this action" in env_py
    assert "`messages[:-1]` and apply `assistant_response` once" in env_py
    # the docstring documents all three algorithms train off the multi-turn env (no opd carve-out)
    assert "distils EVERY assistant turn" in env_py
    assert "single-turn only" not in env_py
    # all three algorithm configs are scaffolded in multi-turn mode too
    for name in ("configs/sft.toml", "configs/rl.toml", "configs/opd.toml"):
        assert (tmp_path / name).is_file(), name
    opd_text = (tmp_path / "configs/opd.toml").read_text()
    assert 'algorithm = "opd"' in opd_text
    # ...and the multi-turn opd.toml notes it distils every assistant turn, with NO fail-fast warning
    assert "distils every assistant turn" in opd_text
    assert "SINGLE-TURN only" not in opd_text
    assert "fail fast" not in opd_text
    assert "configs/opd.toml" in capsys.readouterr().out


def test_env_setup_reasoning_emits_parseable_opd_config(monkeypatch, tmp_path) -> None:
    """The reasoning opd.toml must still be valid TOML with the key at top level.

    `thinking` is a root key, not a `[train]` one, so it has to be written before the first table
    header. Asserting only on the substring would pass just as happily with the line stranded under
    `[train]` or glued to the `algorithm` line.
    """
    monkeypatch.chdir(tmp_path)
    assert (
        _run(
            [
                "env",
                "setup",
                "--project",
                "11111111-1111-4111-8111-111111111111",
                "--multi-turn",
                "--reasoning",
            ]
        )
        == 0
    )

    for name in ("configs/opd.toml", "configs/rl.toml", "configs/sft.toml"):
        parsed = tomllib.loads((tmp_path / name).read_text())
        assert parsed["thinking"] is True, name
        assert "thinking" not in parsed.get("train", {}), name


def test_env_setup_reasoning_conflict_names_every_stale_config(
    monkeypatch, tmp_path, capsys
) -> None:
    """Deleting exactly what the conflict warning names must not leave a config behind.

    All three configs persist `thinking`, so a user who follows this warning literally has to end up
    with three consistent configs. If the warning named only a subset, the unnamed one would keep the
    old setting while the others were rewritten -- the same silent cross-algorithm mismatch the
    `thinking` emission fixes, just moved one step later.
    """
    monkeypatch.chdir(tmp_path)
    project = ["--project", "11111111-1111-4111-8111-111111111111"]
    assert _run(["env", "setup", *project, "--reasoning"]) == 0
    capsys.readouterr()

    # a rerun that disagrees warns instead of rewriting, and must name every config holding the state.
    assert _run(["env", "setup", *project, "--no-reasoning"]) == 0
    warning = capsys.readouterr().err
    assert "ignoring --no-reasoning" in warning
    for name in ("configs/rl.toml", "configs/opd.toml", "configs/sft.toml"):
        assert name in warning, f"{name} missing from: {warning}"

    # follow the instruction exactly: delete what it named, nothing more.
    named = [w.strip(" ,.") for w in warning.split() if w.startswith("configs/")]
    for name in named:
        (tmp_path / name).unlink()
    assert _run(["env", "setup", *project, "--no-reasoning"]) == 0

    for name in ("configs/rl.toml", "configs/opd.toml", "configs/sft.toml"):
        parsed = tomllib.loads((tmp_path / name).read_text())
        assert "thinking" not in parsed, f"{name} kept reasoning after a --no-reasoning re-scaffold"


def test_existing_reasoning_ignores_thinking_text_in_comments(tmp_path) -> None:
    from flash.cli.commands.env.ops import setup as env_setup

    sft = tmp_path / "sft.toml"
    rl = tmp_path / "rl.toml"
    sft.write_text(
        'model = "Qwen/Qwen3.5-9B"\n'
        "# reasoning is on (thinking = true): gold outputs need think tags\n"
    )
    rl.write_text('model = "Qwen/Qwen3.5-9B"\n')

    assert env_setup._existing_reasoning((sft, rl)) is False


def _drop_thinking(path) -> None:
    """Rewrite a config the way the pre-#824 release left opd.toml: no `thinking` key at all."""
    path.write_text(path.read_text().replace("thinking = true\n", ""))


def test_env_setup_refuses_a_scaffold_whose_configs_disagree_about_reasoning(
    monkeypatch, tmp_path, capsys
) -> None:
    """Anchoring on the first config found accepts a project whose configs contradict each other.

    This is the exact shape a scaffold from before #824 has: that release wrote `thinking` into rl
    and sft but never into opd. `configs/rl.toml` is visited first, reports reasoning, and the stale
    opd config is neither reported nor rewritten -- configs are only written when absent, so the
    rerun leaves the project training GRPO with reasoning and OPD without it, silently.

    There is no anchor to pick here: either side leaves a mismatch, so setup has to refuse.
    """
    monkeypatch.chdir(tmp_path)
    project = ["--project", "11111111-1111-4111-8111-111111111111"]
    assert _run(["env", "setup", *project, "--reasoning"]) == 0
    _drop_thinking(tmp_path / "configs/opd.toml")
    capsys.readouterr()

    # a user upgrading re-runs setup with no flag at all -- the case that used to pass silently.
    assert _run(["env", "setup", *project]) == 1
    err = capsys.readouterr().err
    assert "disagree about reasoning" in err
    # the message has to name which side is which, or the user cannot tell what state they are in.
    assert "configs/rl.toml" in err, err
    assert "configs/sft.toml" in err, err
    assert "configs/opd.toml" in err, err
    # and it must name every config to delete, so following it literally cannot leave one behind.
    named = {w.strip(" ,.`") for w in err.split() if w.startswith("configs/")}
    assert named == {"configs/rl.toml", "configs/sft.toml", "configs/opd.toml"}, err

    # refusing must not have half-written anything: the disagreeing scaffold is left exactly as it
    # was, so the user's own files are still theirs to inspect before deleting.
    assert "thinking" not in tomllib.loads((tmp_path / "configs/opd.toml").read_text())
    assert tomllib.loads((tmp_path / "configs/rl.toml").read_text())["thinking"] is True


def test_env_setup_refusal_is_not_dodged_by_an_explicit_reasoning_flag(
    monkeypatch, tmp_path, capsys
) -> None:
    # a flag cannot repair the mismatch either: existing configs are never rewritten, so
    # `--no-reasoning` over a disagreeing scaffold would warn about ignoring the flag and then leave
    # the same split state. the refusal has to come first, whatever the user asked for.
    monkeypatch.chdir(tmp_path)
    project = ["--project", "11111111-1111-4111-8111-111111111111"]
    assert _run(["env", "setup", *project, "--reasoning"]) == 0
    _drop_thinking(tmp_path / "configs/opd.toml")
    capsys.readouterr()

    for flag in ("--reasoning", "--no-reasoning"):
        assert _run(["env", "setup", *project, flag]) == 1, flag
        assert "disagree about reasoning" in capsys.readouterr().err, flag


def test_env_setup_still_accepts_a_scaffold_whose_configs_agree(monkeypatch, tmp_path) -> None:
    # the guard must fire on disagreement only. a consistent rerun is the overwhelmingly common
    # case, and a check that rejects it would break every re-scaffold rather than the broken ones.
    monkeypatch.chdir(tmp_path)
    project = ["--project", "11111111-1111-4111-8111-111111111111"]
    for flag in ("--reasoning", "--no-reasoning"):
        target = tmp_path / flag.strip("-")
        target.mkdir()
        monkeypatch.chdir(target)
        assert _run(["env", "setup", *project, flag]) == 0
        assert _run(["env", "setup", *project]) == 0, f"a consistent {flag} rerun was rejected"
        expected = True if flag == "--reasoning" else None
        for name in ("configs/rl.toml", "configs/sft.toml", "configs/opd.toml"):
            parsed = tomllib.loads((target / name).read_text())
            assert parsed.get("thinking") is expected, name


def test_env_setup_reasoning_anchor_survives_a_partially_scaffolded_directory(
    monkeypatch, tmp_path
) -> None:
    # only some configs existing is NOT a disagreement -- a config that is absent is about to be
    # written from the anchor. treating "missing" as "reasoning off" would reject a directory whose
    # single existing config is perfectly coherent, which is a normal partial scaffold.
    monkeypatch.chdir(tmp_path)
    project = ["--project", "11111111-1111-4111-8111-111111111111"]
    assert _run(["env", "setup", *project, "--reasoning"]) == 0
    (tmp_path / "configs/opd.toml").unlink()
    (tmp_path / "configs/sft.toml").unlink()

    assert _run(["env", "setup", *project]) == 0
    for name in ("configs/rl.toml", "configs/sft.toml", "configs/opd.toml"):
        parsed = tomllib.loads((tmp_path / name).read_text())
        assert parsed["thinking"] is True, f"{name} was rewritten off the surviving anchor"


def test_env_setup_default_omits_reasoning(monkeypatch, tmp_path) -> None:
    # Non-interactive (pytest stdin is not a tty) with no flags stays on today's scaffold: no
    # reasoning knobs land in either config.
    monkeypatch.chdir(tmp_path)
    assert _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 0
    rl = (tmp_path / "configs/rl.toml").read_text()
    sft = (tmp_path / "configs/sft.toml").read_text()
    opd = (tmp_path / "configs/opd.toml").read_text()
    assert "thinking = true" not in rl
    assert "thinking = true" not in sft
    assert "thinking = true" not in opd
    assert "max_completion_tokens" not in rl
    assert "EnvironmentSingleTurn" in (tmp_path / "environment.py").read_text()


def test_env_setup_reasoning_flag_enables_thinking(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    assert (
        _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111", "--reasoning"])
        == 0
    )
    rl = (tmp_path / "configs/rl.toml").read_text()
    sft = (tmp_path / "configs/sft.toml").read_text()
    opd = (tmp_path / "configs/opd.toml").read_text()
    assert "thinking = true" in rl
    assert "thinking = true" in sft
    # opd too. `thinking` is algorithm-agnostic in the spec and the opd worker reads it to pick a
    # reasoning parser, so a scaffold that emitted it for two of the three algorithms would hand a
    # user asking for reasoning an opd config that silently trains without it.
    assert "thinking = true" in opd
    # GRPO raises the generation budget so reasoning does not truncate the answer.
    assert "max_completion_tokens = 2048" in rl
    # SFT can't share a token budget it doesn't generate; it gets the gold think-tag guidance instead.
    assert "warn_missing_think_tags" in sft
    assert "max_completion_tokens" not in sft
    # nor opd -- not because the knob is inert there (opd honors it, via `_resolve_opd_knobs` ->
    # `opd_completion_len` at flash/engine/plan/vram.py:97, which feeds verl's max_response_length) but
    # because opd ALREADY raises its own budget under thinking: 512 -> 1536. Writing a literal would
    # pin what the recipe should choose, and would go stale the moment that default moves. GRPO needs
    # the line only because its non-thinking default is 320, too tight to leave to a scaffold reader.
    assert "max_completion_tokens" not in opd


def test_env_setup_no_reasoning_flag_is_explicit_off(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    assert (
        _run(
            ["env", "setup", "--project", "11111111-1111-4111-8111-111111111111", "--no-reasoning"]
        )
        == 0
    )
    assert "thinking = true" not in (tmp_path / "configs/rl.toml").read_text()


def test_env_setup_multi_turn_flag_scaffolds_multiturn(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    assert (
        _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111", "--multi-turn"])
        == 0
    )
    assert "EnvironmentMultiTurn" in (tmp_path / "environment.py").read_text()
    assert "secret whole number" in (tmp_path / "dataset/train.jsonl").read_text()


def test_env_setup_multi_turn_scaffolds_runnable_evaluations(monkeypatch, tmp_path) -> None:
    # the multi-turn scaffold ships its own evaluations.py, and `env eval` sends one prompt and
    # grades one reply. so the starter grades the FIRST action's format rather than delegating to
    # the environment, which with no episode state would score an empty transcript. asserting the
    # file exists would not catch a sidecar that raises on every case, which reads as the model
    # failing rather than as a broken template.
    monkeypatch.chdir(tmp_path)
    assert (
        _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111", "--multi-turn"])
        == 0
    )
    assert (tmp_path / "evaluations.py").is_file()

    from flash.envs.loading.loader import load_freesolo_environment
    from flash.envs.meta.evaluations import load_evaluation_suites

    environment = load_freesolo_environment(str(tmp_path / "environment.py"))
    suite = load_evaluation_suites(tmp_path / "environment.py", environment=environment)[0]
    case = suite.cases()[0]

    # a reply `step_episode` can read: a single in-range integer scores full marks.
    passing = suite.score(case, "50")
    assert passing.passed is True
    assert passing.score == 1.0

    # and one it cannot is graded, not raised: the template reports why rather than erroring out.
    failing = suite.score(case, "somewhere in the middle")
    assert failing.passed is False
    assert failing.score == 0.0
    assert "single integer" in failing.reason

    # the environment is multi-turn but this suite grades one reply, so it must NOT opt into
    # episode play. driving it would score the last turn instead of the opening action, and these
    # cases carry no `output` for `step_episode` to advance from, so the case would error out.
    from flash.cli.commands.env.testing.episode import _grades_episodes

    assert environment.multi_turn is True
    assert _grades_episodes(suite) is False


def test_env_setup_multi_turn_eval_case_does_not_duplicate_the_episode_prompt(
    monkeypatch, tmp_path
) -> None:
    # `env eval` builds the request through environment.prompt_messages(), so the scaffolded case's
    # `input` is a dataset row, not a finished prompt. spelling the reply-instructions block into
    # the case as well sent it twice and evaluated a prompt training never used, defeating the fix
    # that made eval match training in the first place.
    monkeypatch.chdir(tmp_path)
    assert (
        _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111", "--multi-turn"])
        == 0
    )

    from flash.envs.loading.loader import load_freesolo_environment
    from flash.envs.meta.evaluations import load_evaluation_suites

    environment = load_freesolo_environment(str(tmp_path / "environment.py"))
    case = load_evaluation_suites(tmp_path / "environment.py", environment=environment)[0].cases()[
        0
    ]
    prompt = "\n".join(
        str(message.get("content") or "")
        for message in environment.prompt_messages({"input": case.input})
    )

    # the instructions reach the model exactly once, and they come from the environment.
    assert prompt.count("Reply with a single integer per turn") == 1
    assert "Reply with a single integer per turn" not in case.input
    # and the case still carries the part of the prompt only the dataset row knows.
    assert "secret whole number between 1 and 100" in case.input


def test_starter_evaluator_fails_a_near_miss_the_environment_rejects(monkeypatch, tmp_path) -> None:
    """A shaped reward's partial credit is not a pass.

    The starter sidecar delegates to the environment, and a shaped reward pays partial credit for a
    wrong answer -- the multi-turn starter's `score_episode` scores a near miss as `closeness * 0.5`
    with `success=False`. Returning that bare float let `normalize_eval_result` mark every positive
    score as passed, so an incorrect answer was reported as a passing evaluation case: a graded
    failure reading as model success, which is the one thing the suite exists to detect.

    The scaffolded single-turn environment happens to reward 1.0/0.0, where "positive" and "the
    environment says this succeeded" coincide -- so the two rules are only distinguishable against
    an environment that actually shapes its reward, which is what any real one does.
    """
    monkeypatch.chdir(tmp_path)
    assert _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 0

    from flash.envs.meta.evaluations import load_evaluation_suites

    class ShapedEnvironment:
        """Pays partial credit for a wrong answer, exactly as the multi-turn starter does."""

        def reward(self, response, example):
            return 0.5

        def grade(self, response, example):
            return False

    suite = load_evaluation_suites(tmp_path / "environment.py", environment=ShapedEnvironment())[0]
    case = suite.cases()[0]

    scored = suite.score(case, "the wrong answer, but close")

    assert scored.score == 0.5, "the shaped reward must reach the report unchanged"
    assert scored.passed is False


def test_env_setup_interactive_survey_picks_multi_and_reasoning(monkeypatch, tmp_path) -> None:
    # A real terminal: the two survey questions are asked and answered "2" (multi-turn) then
    # "2" (reasoning). FLASH_STYLE forces the themed path; a fake stdin reports a tty; input()
    # is stubbed so no real keypress is read. CI is cleared so the guard doesn't force defaults.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        "flash.client.config.load_credentials", lambda: ("https://flash.freesolo.co", "fslo-test")
    )
    answers = iter(["1", "", "2", "2"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    assert _run(["env", "setup"]) == 0
    assert "EnvironmentMultiTurn" in (tmp_path / "environment.py").read_text()
    rl = (tmp_path / "configs/rl.toml").read_text()
    assert "thinking = true" in rl
    assert "max_completion_tokens = 2048" in rl


def test_env_setup_interactive_enter_takes_defaults(monkeypatch, tmp_path) -> None:
    # Pressing enter (empty answer) at both questions selects the marked defaults: single-turn,
    # no reasoning — identical to the non-interactive scaffold.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        "flash.client.config.load_credentials", lambda: ("https://flash.freesolo.co", "fslo-test")
    )
    answers = iter(["1", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    assert _run(["env", "setup"]) == 0
    env_py = (tmp_path / "environment.py").read_text()
    assert "EnvironmentSingleTurn" in env_py
    assert "EnvironmentMultiTurn" not in env_py
    assert "thinking = true" not in (tmp_path / "configs/rl.toml").read_text()


def test_env_setup_under_ci_never_prompts(monkeypatch, tmp_path) -> None:
    # Even with a tty and themed output, CI=true must fall back to defaults instead of blocking on
    # a prompt. input() raises so a regression (actually prompting) fails loudly.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))

    def _boom(*a, **k):
        raise AssertionError("prompted under CI")

    monkeypatch.setattr("builtins.input", _boom)
    assert _run(["env", "setup"]) == 1
    assert not (tmp_path / "environment.py").exists()
    assert _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 0
    assert "EnvironmentSingleTurn" in (tmp_path / "environment.py").read_text()
    assert "thinking = true" not in (tmp_path / "configs/rl.toml").read_text()


def test_env_setup_yes_requires_project_before_creating_files(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    assert _run(["env", "setup", "--yes"]) == 1
    assert "--project PROJECT_UUID is required" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "canonical_fields",
    [{}, {"name": None}, {"name": 7}, {"name": ""}, {"name": "   "}],
)
def test_env_setup_uses_project_id_when_canonical_name_is_unusable(
    monkeypatch, tmp_path, canonical_fields
) -> None:
    monkeypatch.setattr(
        "flash.client.get_project",
        lambda project_id, _api_key: {"id": project_id, **canonical_fields},
    )
    monkeypatch.chdir(tmp_path)

    assert _run(["env", "setup", "--project", _SCAFFOLD_PROJECT]) == 0

    for name in ("sft.toml", "rl.toml", "opd.toml"):
        parsed = tomllib.loads((tmp_path / "configs" / name).read_text())
        assert parsed["wandb"]["project"] == _SCAFFOLD_PROJECT


def test_env_setup_rejects_blank_folder_name_before_writes(monkeypatch, tmp_path, capsys) -> None:
    from flash.cli.commands.env.ops import setup as env_setup

    class _BlankCwd:
        name = "   "

    class _Path:
        @staticmethod
        def cwd():
            return _BlankCwd()

    monkeypatch.setattr(env_setup, "Path", _Path)
    monkeypatch.chdir(tmp_path)

    assert _run(["env", "setup", "--project", _SCAFFOLD_PROJECT]) == 1

    assert "environment folder name must be nonblank" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_env_setup_rejects_projectless_or_conflicting_existing_configs(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "sft.toml").write_text('model = "Qwen/Qwen3.5-9B"\n')
    assert _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 1
    assert "has no valid top-level project UUID" in capsys.readouterr().err
    assert not (tmp_path / "environment.py").exists()

    (configs / "sft.toml").write_text('project = "22222222-2222-4222-8222-222222222222"\n')
    assert _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 1
    assert "does not match selected project" in capsys.readouterr().err
    assert not (tmp_path / "environment.py").exists()


def test_env_setup_reasoning_flag_warns_when_configs_exist(monkeypatch, tmp_path, capsys) -> None:
    # First scaffold (no reasoning), then re-run with --reasoning: the existing configs win and the
    # command warns instead of silently ignoring the flag or writing mismatched configs.
    monkeypatch.chdir(tmp_path)
    assert _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 0
    capsys.readouterr()
    assert (
        _run(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111", "--reasoning"])
        == 0
    )
    err = capsys.readouterr().err
    assert "existing configs are no reasoning" in err
    assert "ignoring --reasoning" in err
    assert "thinking = true" not in (tmp_path / "configs/rl.toml").read_text()


def test_unknown_run_errors_surface_as_nonzero_exit(monkeypatch, capsys) -> None:
    from flash.client import ApiError

    class _Erroring(_FakeClient):
        def get_run(self, run_id: str) -> dict:
            raise ApiError(404, "unknown run")

    monkeypatch.setattr(run_commands, "client_from_config", lambda *a, **k: _Erroring())
    assert _run(["runs", "status", "nope"]) != 0
    assert "unknown run" in capsys.readouterr().err


def test_submit_payload_carries_authored_pip_and_the_worker_appends_it(
    monkeypatch, tmp_path
) -> None:
    """pip is authorable: it travels on the wire, and the submit path adds the worker baseline.

    Both halves matter. Carrying the key without the provider still supplying the baseline would
    ship a worker with no Freesolo SDK, and the failure would surface only on a real GPU.
    """
    from flash.client.specs import spec_payload
    from flash.core.spec import EnvironmentSpec, JobSpec
    from flash.envs.loading.base import worker_pip_with_extras

    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        project="11111111-1111-4111-8111-111111111111",
        environment=EnvironmentSpec(id="owner/project/env", pip=("pymongo>=4.6",)),
    )

    # the author's scorer dependency reaches the server rather than being stripped on the client.
    assert tuple(spec_payload(spec)["environment"]["pip"]) == ("pymongo>=4.6",)
    # and the submit paths install it after the worker requirement, not instead of it.
    assert worker_pip_with_extras(spec.environment.id, spec.environment.pip) == [
        "freesolo>=0.4.2",
        "pymongo>=4.6",
    ]


def test_export_uses_api_key_flag_and_forwards_args(fake_client, capsys, monkeypatch) -> None:
    # The --api-key flag is the destination HF token; checkpoint refs and --public are forwarded.
    from flash.cli.commands.ops import deploy as cli_deploy

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(cli_deploy, "_hf_identity_and_write_access", lambda *_: "me")
    assert (
        _run(
            [
                "models",
                "export",
                "--adapter-id",
                "flash-1/step-40",
                "--repository",
                "me/adapters",
                "--api-key",
                "hf_flag",
                "--public",
            ]
        )
        == 0
    )
    assert ("export", "flash-1/step-40", "me/adapters", "hf_flag", False) in fake_client.calls
    # The destination repo / url are reported back to the user.
    captured = capsys.readouterr()
    assert "me/adapters" in captured.out
    assert "visible in process listings" in captured.err
    assert "hf_flag" not in captured.err


def test_export_reads_hf_token_from_env_and_defaults_private(
    fake_client, monkeypatch, capsys
) -> None:
    # No --api-key: the token resolves from HF_TOKEN, and the repo defaults to private.
    from flash.cli.commands.ops import deploy as cli_deploy

    monkeypatch.setenv("HF_TOKEN", "hf_env")
    monkeypatch.setattr(cli_deploy, "_hf_identity_and_write_access", lambda *_: "me")
    assert _run(["models", "export", "--adapter-id", "flash-1", "--repository", "me/adapters"]) == 0
    assert ("export", "flash-1", "me/adapters", "hf_env", True) in fake_client.calls
    err = capsys.readouterr().err
    assert "visible in process listings" not in err
    assert "hf_env" not in err


def test_export_without_token_errors_cleanly(fake_client, monkeypatch, capsys, tmp_path) -> None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    # A clean cwd so a stray local .env can't supply a token.
    monkeypatch.chdir(tmp_path)
    assert _run(["models", "export", "--adapter-id", "flash-1", "--repository", "me/adapters"]) == 1
    err = capsys.readouterr().err
    assert "HuggingFace token" in err
    # The control plane is never contacted when there's no token to send.
    assert not any(call[0] == "export" for call in fake_client.calls)


def test_export_rejects_unwritable_namespace_before_control_plane(
    fake_client, monkeypatch, capsys
) -> None:
    import sys
    import types

    class FakeHfApi:
        def whoami(self, token):
            return {"name": "alice", "orgs": [], "auth": {"accessToken": {"role": "write"}}}

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeHfApi))
    assert (
        _run(
            [
                "models",
                "export",
                "--adapter-id",
                "flash-1",
                "--repository",
                "bob/adapters",
                "--api-key",
                "hf_secret",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "account alice" in err
    assert "namespace bob" in err
    assert "hf_secret" not in err
    assert not any(call[0] == "export" for call in fake_client.calls)


def test_export_allows_an_org_contributor_who_can_write_the_exact_repo(
    fake_client, monkeypatch, capsys
) -> None:
    """The exact-repo permission outranks the coarse org role, so it has to be asked first.

    A `contributor` in `acme` is not `write`/`admin` org-wide, but auth_check says this token can
    write this very repo. Consulting the role first would refuse an export the Hub allows.
    """
    import sys
    import types

    reached: dict[str, bool] = {"auth_check": False}

    class FakeHfApi:
        def whoami(self, token):
            return {
                "name": "alice",
                "orgs": [{"name": "acme", "role": "contributor"}],
                "auth": {"accessToken": {"role": "write"}},
            }

        def auth_check(self, repo_id, *, repo_type=None, token=None, write=False):
            reached["auth_check"] = True
            return

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeHfApi))
    assert (
        _run(
            [
                "models",
                "export",
                "--adapter-id",
                "flash-1",
                "--repository",
                "acme/model",
                "--api-key",
                "hf_secret",
            ]
        )
        == 0
    )
    assert reached["auth_check"], "the exact-repo check must run before the org role decides"
    assert any(call[0] == "export" for call in fake_client.calls)
    assert "hf_secret" not in capsys.readouterr().err


def test_export_allows_an_org_contributor_on_a_hub_too_old_for_the_write_probe(
    fake_client, monkeypatch, capsys
) -> None:
    """`auth_check(..., write=True)` landed in huggingface-hub 1.5, but this package supports >=1.2.

    On 1.2-1.4 there is no exact write probe, so the guard skips it. What follows are CREATION
    rules, and they must not run for a destination that already exists: an `acme` contributor who
    can write `acme/model` would be refused on a supported hub version.
    """
    import sys
    import types

    class FakeHfApi:
        def whoami(self, token):
            return {
                "name": "alice",
                "orgs": [{"name": "acme", "role": "contributor"}],
                "auth": {"accessToken": {"role": "write"}},
            }

        # the pre-1.5 signature: present, but with no `write` parameter to ask for write access.
        def auth_check(self, repo_id, *, repo_type=None, token=None):
            return

        def repo_exists(self, repo_id, *, repo_type=None, token=None):
            return True

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=FakeHfApi, __version__="1.4.0"),
    )
    assert (
        _run(
            [
                "models",
                "export",
                "--adapter-id",
                "flash-1",
                "--repository",
                "acme/model",
                "--api-key",
                "hf_secret",
            ]
        )
        == 0
    )
    assert any(call[0] == "export" for call in fake_client.calls)
    err = capsys.readouterr().err
    assert "1.4.0" in err, "the unverified export must say which hub could not check it"
    assert "hf_secret" not in err


def test_export_still_applies_creation_rules_on_an_old_hub_when_the_repo_is_absent(
    fake_client, monkeypatch, capsys
) -> None:
    """The old-hub bypass is scoped to an EXISTING destination.

    A repo that is not there yet is genuinely being created, so the namespace rules still decide.
    Skipping them here would turn a missing exact probe into a way to create a repo in an org this
    token has no write access to -- the wrong-namespace export this preflight exists to stop.
    """
    import sys
    import types

    class FakeHfApi:
        def whoami(self, token):
            return {
                "name": "alice",
                "orgs": [{"name": "acme", "role": "contributor"}],
                "auth": {"accessToken": {"role": "write"}},
            }

        def auth_check(self, repo_id, *, repo_type=None, token=None):
            return

        def repo_exists(self, repo_id, *, repo_type=None, token=None):
            return False

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=FakeHfApi, __version__="1.4.0"),
    )
    assert (
        _run(
            [
                "models",
                "export",
                "--adapter-id",
                "flash-1",
                "--repository",
                "acme/model",
                "--api-key",
                "hf_secret",
            ]
        )
        == 1
    )
    assert not any(call[0] == "export" for call in fake_client.calls)
    err = capsys.readouterr().err
    assert "cannot create" in err
    assert "hf_secret" not in err


def test_export_keeps_the_creation_rules_when_existence_is_inconclusive(
    fake_client, monkeypatch, capsys
) -> None:
    """An unanswered existence lookup must not buy a bypass of the namespace rules.

    The old-hub branch skips the creation rules on the grounds that an EXISTING repo is not being
    created. Extending that to a lookup that merely failed would let any transient Hub error carry
    an export into a namespace nothing verified -- the wrong-namespace export this preflight exists
    to stop. Being denied a bypass costs a `contributor` one retry; granting it wrongly is silent.
    """
    import sys
    import types

    class FakeHfApi:
        def whoami(self, token):
            return {
                "name": "alice",
                "orgs": [{"name": "acme", "role": "contributor"}],
                "auth": {"accessToken": {"role": "write"}},
            }

        # the pre-1.5 signature: no `write` parameter, so there is no exact write probe.
        def auth_check(self, repo_id, *, repo_type=None, token=None):
            return

        def repo_exists(self, repo_id, *, repo_type=None, token=None):
            raise OSError("hub unreachable")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=FakeHfApi, __version__="1.4.0"),
    )
    assert (
        _run(
            [
                "models",
                "export",
                "--adapter-id",
                "flash-1",
                "--repository",
                "acme/model",
                "--api-key",
                "hf_secret",
            ]
        )
        == 1
    )
    assert not any(call[0] == "export" for call in fake_client.calls)
    err = capsys.readouterr().err
    assert "cannot create" in err
    assert "hf_secret" not in err


def test_export_proceeds_when_the_exact_repo_probe_cannot_reach_the_hub(
    fake_client, monkeypatch, capsys
) -> None:
    """A timeout or DNS failure from `auth_check` is not a verdict on the token's permissions.

    Reporting it as "cannot write to <repo>" blames the user's access for a network fault, and the
    upload runs on the control plane regardless. The whoami path above already degrades this way.
    """
    import sys
    import types

    class FakeHfApi:
        def whoami(self, token):
            return {
                "name": "alice",
                "orgs": [{"name": "acme", "role": "write"}],
                "auth": {"accessToken": {"role": "write"}},
            }

        def auth_check(self, repo_id, *, repo_type=None, token=None, write=False):
            raise OSError("temporary failure in name resolution")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=FakeHfApi, __version__="1.27.0"),
    )
    assert (
        _run(
            [
                "models",
                "export",
                "--adapter-id",
                "flash-1",
                "--repository",
                "acme/model",
                "--api-key",
                "hf_secret",
            ]
        )
        == 0
    )
    assert any(call[0] == "export" for call in fake_client.calls)
    err = capsys.readouterr().err
    assert "cannot write to" not in err, "an unreachable Hub must not read as a permission verdict"
    assert "hf_secret" not in err


def test_export_warning_does_not_echo_the_token_from_a_hub_exception(
    fake_client, monkeypatch, capsys
) -> None:
    """The degrade warnings quote the exception, and a local credential rejection quotes the token.

    httpx raises `Illegal header value b'Bearer <token>'` for a token with an internal newline, so
    interpolating the exception verbatim prints the credential into stderr, logs and bug reports.

    The token here holds a REAL newline and the exception renders the ESCAPED form, which is what
    httpx actually does -- it builds that message from the header bytes. An earlier version of this
    test passed a token with no newline against a message containing a literal backslash-n, so the
    two happened to match and a redaction that only replaced the exact token looked correct.
    """
    import sys
    import types

    token = "hf_secret\nINJECT"

    class FakeHfApi:
        def whoami(self, token):
            return {
                "name": "alice",
                "orgs": [{"name": "acme", "role": "write"}],
                "auth": {"accessToken": {"role": "write"}},
            }

        def auth_check(self, repo_id, *, repo_type=None, token=None, write=False):
            # the bytes repr of the header, escapes and all, exactly as httpx renders it.
            raise ValueError(f"Illegal header value {b'Bearer ' + token.encode()!r}")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=FakeHfApi, __version__="1.27.0"),
    )
    _run(
        [
            "models",
            "export",
            "--adapter-id",
            "flash-1",
            "--repository",
            "acme/model",
            "--api-key",
            token,
        ]
    )
    err = capsys.readouterr().err
    assert "hf_secret" not in err, "the escaped rendering must be redacted too"
    assert "INJECT" not in err
    # the exception is still shown: it is the only clue to what actually failed.
    assert "Illegal header value" in err


def test_export_treats_a_rate_limit_or_outage_as_no_answer_but_still_obeys_a_verdict(
    fake_client, monkeypatch, capsys
) -> None:
    """429 and 5xx describe the Hub; 401/403 describe this token. Only the latter may block.

    Rate limiting and outages are transport facts, so refusing the export on them blames the user's
    access for the Hub being busy. The distinction is the whole reason the check keys on the status
    rather than on "an exception happened", so both sides of the boundary are pinned here.
    """
    import sys
    import types

    class Response:
        def __init__(self, status_code):
            self.status_code = status_code

    def run_with(status):
        class FakeHfApi:
            def whoami(self, token):
                return {
                    "name": "alice",
                    "orgs": [{"name": "acme", "role": "write"}],
                    "auth": {"accessToken": {"role": "write"}},
                }

            def auth_check(self, repo_id, *, repo_type=None, token=None, write=False):
                error = RuntimeError(f"hub said {status}")
                error.response = Response(status)
                raise error

        monkeypatch.setitem(
            sys.modules,
            "huggingface_hub",
            types.SimpleNamespace(HfApi=FakeHfApi, __version__="1.27.0"),
        )
        fake_client.calls.clear()
        code = _run(
            [
                "models",
                "export",
                "--adapter-id",
                "flash-1",
                "--repository",
                "acme/model",
                "--api-key",
                "hf_secret",
            ]
        )
        capsys.readouterr()
        return code

    # 408 and 425 carry a response, so they arrive here with a status rather than through the
    # statusless path -- but a request timeout and a too-early replay are transport outcomes, not
    # statements about this token's access.
    for transient in (408, 425, 429, 500, 503):
        assert run_with(transient) == 0, f"{transient} is about the Hub, not about this token"
    for verdict in (401, 403):
        assert run_with(verdict) == 1, f"{verdict} answers the permission question and must block"


def test_export_refuses_a_gated_destination_instead_of_treating_it_as_creatable(
    fake_client, monkeypatch, capsys
) -> None:
    """GatedRepoError subclasses RepositoryNotFoundError, so "gated" must be subtracted from "missing".

    A gated repo exists and this token may not write it. Reading it as absent would hand it to the
    create-permission paths, which are weaker.
    """
    import sys
    import types

    from huggingface_hub.utils import GatedRepoError

    # a real GatedRepoError, because the point of this test is that the hub makes it a SUBCLASS of
    # RepositoryNotFoundError. it needs a response carrying `request`/`headers` or its own
    # constructor raises, and that AttributeError would block for the wrong reason -- passing the
    # test while never exercising the gated branch at all.
    response = type(
        "R",
        (),
        {"status_code": 403, "headers": {}, "request": type("Q", (), {"headers": {}})()},
    )()

    class FakeHfApi:
        def whoami(self, token):
            return {"name": "alice", "orgs": [], "auth": {"accessToken": {"role": "write"}}}

        def auth_check(self, repo_id, *, repo_type=None, token=None, write=False):
            raise GatedRepoError("gated", response=response)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeHfApi))
    assert (
        _run(
            [
                "models",
                "export",
                "--adapter-id",
                "flash-1",
                "--repository",
                "someone/gated-model",
                "--api-key",
                "hf_secret",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "cannot write" in err
    assert "hf_secret" not in err
    assert not any(call[0] == "export" for call in fake_client.calls)


def test_export_proceeds_when_the_hub_cannot_be_reached(fake_client, monkeypatch, capsys) -> None:
    """An unreachable Hub is not a verdict. The copy runs on the control plane, not on this host.

    Blocking here would make a CLI host without Hub egress unable to export at all, while the same
    command skips the check entirely when huggingface_hub is merely absent.
    """
    import sys
    import types

    class FakeHfApi:
        def whoami(self, token):
            raise OSError("[Errno -3] Temporary failure in name resolution")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeHfApi))
    assert (
        _run(
            [
                "models",
                "export",
                "--adapter-id",
                "flash-1",
                "--repository",
                "alice/adapters",
                "--api-key",
                "hf_secret",
            ]
        )
        == 0
    )
    err = capsys.readouterr().err
    assert "could not reach HuggingFace" in err
    assert "hf_secret" not in err
    assert any(call[0] == "export" for call in fake_client.calls)


def test_export_still_refuses_a_token_the_hub_rejected(fake_client, monkeypatch, capsys) -> None:
    """The other half of the same branch: a real answer from the Hub must still block."""
    import sys
    import types

    class Rejected(Exception):
        """A hub error shape carrying a real status: the Hub answered, and the answer was no."""

        def __init__(self) -> None:
            super().__init__("unauthorized")
            self.response = type("R", (), {"status_code": 401, "headers": {}})()

    class FakeHfApi:
        def whoami(self, token):
            raise Rejected

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeHfApi))
    assert (
        _run(
            [
                "models",
                "export",
                "--adapter-id",
                "flash-1",
                "--repository",
                "alice/adapters",
                "--api-key",
                "hf_secret",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "rejected the token" in err
    assert "hf_secret" not in err
    assert not any(call[0] == "export" for call in fake_client.calls)


def test_export_rejects_fine_grained_scope_for_a_different_namespace(
    fake_client, monkeypatch, capsys
) -> None:
    """A write scope on the token's own repo says nothing about the org it is exporting into.

    alice can write in `acme` by org role, so the namespace gate passes, but her only fine-grained
    grant covers `alice/*`. Crediting it because it is user-typed would wave through exactly the
    wrong-namespace export this preflight exists to stop.
    """
    import sys
    import types

    class FakeHfApi:
        def whoami(self, token):
            return {
                "name": "alice",
                "orgs": [{"name": "acme", "role": "write"}],
                "auth": {
                    "accessToken": {
                        "role": "fineGrained",
                        "fineGrained": {
                            "global": [],
                            "scoped": [
                                {
                                    "entity": {"name": "alice", "type": "user"},
                                    "permissions": ["repo.write"],
                                }
                            ],
                        },
                    }
                },
            }

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeHfApi))
    assert (
        _run(
            [
                "models",
                "export",
                "--adapter-id",
                "flash-1",
                "--repository",
                "acme/secret-model",
                "--api-key",
                "hf_secret",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "acme/secret-model" in err
    assert "hf_secret" not in err
    assert not any(call[0] == "export" for call in fake_client.calls)


def test_export_accepts_fine_grained_scope_naming_the_destination(
    fake_client, monkeypatch, capsys
) -> None:
    """The same shape, scoped to the destination org, is the case that must still export."""
    import sys
    import types

    class FakeHfApi:
        def whoami(self, token):
            return {
                "name": "alice",
                "orgs": [{"name": "acme", "role": "write"}],
                "auth": {
                    "accessToken": {
                        "role": "fineGrained",
                        "fineGrained": {
                            "global": [],
                            "scoped": [
                                {
                                    "entity": {"name": "acme", "type": "org"},
                                    "permissions": ["repo.write"],
                                }
                            ],
                        },
                    }
                },
            }

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeHfApi))
    assert (
        _run(
            [
                "models",
                "export",
                "--adapter-id",
                "flash-1",
                "--repository",
                "acme/secret-model",
                "--api-key",
                "hf_secret",
            ]
        )
        == 0
    )
    assert any(call[0] == "export" for call in fake_client.calls)
    assert "hf_secret" not in capsys.readouterr().err


def test_export_matching_namespace_proceeds_after_preflight(
    fake_client, monkeypatch, capsys
) -> None:
    import sys
    import types

    class FakeHfApi:
        def whoami(self, token):
            return {"name": "alice", "orgs": [], "auth": {"accessToken": {"role": "write"}}}

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeHfApi))
    assert (
        _run(
            [
                "models",
                "export",
                "--adapter-id",
                "flash-1",
                "--repository",
                "alice/adapters",
                "--api-key",
                "hf_secret",
            ]
        )
        == 0
    )
    assert ("export", "flash-1", "alice/adapters", "hf_secret", True) in fake_client.calls
    err = capsys.readouterr().err
    assert "account alice" in err
    assert "hf_secret" not in err


def test_export_without_huggingface_hub_skips_preflight(fake_client, monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def missing_hub(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ModuleNotFoundError(name="huggingface_hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_hub)
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert _run(["models", "export", "--adapter-id", "flash-1", "--repository", "me/adapters"]) == 0
    assert ("export", "flash-1", "me/adapters", "hf_env", True) in fake_client.calls


def test_deploy_enqueues_server_side_verification(fake_client, capsys) -> None:
    assert _run(["models", "deploy", "flash-1/final"]) == 0
    assert ("deploy", "flash-1/final", {"dry_run": False}) in fake_client.calls
    assert not any(c[0] == "chat" for c in fake_client.calls)
    err = capsys.readouterr().err
    assert "flash models deployments" in err
    assert "OpenAI-compatible base URL" in err


def test_deploy_checkpoint_enqueues_base_run_deployment(fake_client) -> None:
    assert _run(["models", "deploy", "flash-1/step-40"]) == 0
    assert ("deploy", "flash-1/step-40", {"dry_run": False}) in fake_client.calls
    assert not any(c[0] == "chat" for c in fake_client.calls)


def test_deploy_no_verify_flag_is_removed(fake_client) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run(["models", "deploy", "flash-1/final", "--no-verify"])
    assert excinfo.value.code == 2
    assert not any(call[0] == "deploy" for call in fake_client.calls)


def test_deploy_dry_run_skips_active_deployment_note(fake_client, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {"run_id": run_id, "state": "dry_run"},
        raising=False,
    )
    assert _run(["models", "deploy", "flash-1/final", "--dry-run"]) == 0
    assert not any(c[0] == "chat" for c in fake_client.calls)
    assert "flash models deployments" not in capsys.readouterr().err


def test_deploy_failed_state_exits_nonzero(fake_client, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {
            "run_id": run_id,
            "state": "failed",
            "error": "smoke generation failed",
        },
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1/final"]) == 1
    err = capsys.readouterr().err
    assert "deployment failed: smoke generation failed" in err
    assert "once it is ready" not in err


def _queued_deploy(monkeypatch, fake_client) -> None:
    """Make POST deploy return what the control plane really returns: a queued record."""
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {"run_id": run_id, "state": "queued"},
        raising=False,
    )
    monkeypatch.setattr(run_commands.time, "sleep", lambda _s: None)


def test_deploy_without_wait_returns_while_still_queued(fake_client, monkeypatch, capsys) -> None:
    """No --wait keeps the old behaviour: return immediately, do not poll."""
    _queued_deploy(monkeypatch, fake_client)

    assert _run(["models", "deploy", "flash-1/final"]) == 0
    assert not any(c[0] == "deployment_for" for c in fake_client.calls)
    assert "deployment state is 'queued'" in capsys.readouterr().err


def test_deploy_wait_polls_until_the_revision_is_servable(fake_client, monkeypatch, capsys) -> None:
    """--wait must not return while the requested revision is still queued.

    deploy returns as soon as the record is persisted, which is normally before the new revision
    can serve a token, so a caller that starts evaluating on that return hits the old revision or
    an error. The wait is what makes the printed record mean "ready".
    """
    _queued_deploy(monkeypatch, fake_client)
    states = iter([{"state": "smoke_testing"}, {"state": "reconciling"}, {"state": "ready"}])
    monkeypatch.setattr(
        fake_client, "deployment_for", lambda run_id, timeout=None: next(states), raising=False
    )

    assert _run(["models", "deploy", "flash-1/final", "--wait"]) == 0
    out, err = capsys.readouterr()
    assert "ready" in out
    assert "queued" not in err
    assert "ctrl-c stops waiting, not the deployment" in err


def test_deploy_wait_stops_on_a_failed_revision(fake_client, monkeypatch, capsys) -> None:
    _queued_deploy(monkeypatch, fake_client)
    monkeypatch.setattr(
        fake_client,
        "deployment_for",
        lambda run_id, timeout=None: {"state": "failed", "error": "smoke generation failed"},
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1/final", "--wait"]) == 1
    assert "deployment failed: smoke generation failed" in capsys.readouterr().err


def test_deploy_wait_gives_up_at_the_timeout_without_claiming_success(
    fake_client, monkeypatch, capsys
) -> None:
    _queued_deploy(monkeypatch, fake_client)
    monkeypatch.setattr(
        fake_client,
        "deployment_for",
        lambda run_id, timeout=None: {"state": "smoke_testing"},
        raising=False,
    )

    # exit 1, not 0: --wait's contract is "the revision is servable when i return", and a timeout
    # is precisely the case where it is not. exiting 0 here is what lets
    # `deploy --wait && evaluate` proceed against a revision that never became servable.
    assert _run(["models", "deploy", "flash-1/final", "--wait", "0.01"]) == 1
    err = capsys.readouterr().err
    assert "still 'smoke_testing' after 0.01s" in err
    assert "flash models deployments" in err


def test_deploy_wait_ends_when_the_deployment_stops_being_listed(
    fake_client, monkeypatch, capsys
) -> None:
    """A run drops out of the listing once its deployment is gone, so that is terminal."""
    _queued_deploy(monkeypatch, fake_client)
    monkeypatch.setattr(
        fake_client, "deployment_for", lambda run_id, timeout=None: None, raising=False
    )

    # the last record seen was still queued, so the requested revision never became servable.
    assert _run(["models", "deploy", "flash-1/final", "--wait"]) == 1
    assert "no longer an active deployment" in capsys.readouterr().err


def test_deploy_wait_reports_a_rollback_to_a_different_checkpoint_step(
    fake_client, monkeypatch, capsys
) -> None:
    """An absent revision is not always a deleted one.

    `deployment_for` matches the checkpoint step, so a failed `deploy RUN/step-40` that the plane
    rolled back to step-20 reads as absent exactly like a deletion does. Reporting it as "no longer
    an active deployment" named the wrong event and threw away `last_deploy_error`, which is the
    only record of why step-40 did not take -- the restored record carries no `failed` state.
    """
    _queued_deploy(monkeypatch, fake_client)
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {"run_id": run_id, "state": "queued", "requested_at": "T1"},
        raising=False,
    )
    # the requested revision is gone from the listing; the predecessor is what is serving.
    monkeypatch.setattr(
        fake_client, "deployment_for", lambda run_id, timeout=None: None, raising=False
    )
    monkeypatch.setattr(
        fake_client,
        "deployments",
        lambda timeout=None: [
            {
                "run_id": "flash-1",
                "deployment": {
                    "run_id": "flash-1",
                    "checkpoint_step": 20,
                    "state": "ready",
                    "requested_at": "T0",
                    "last_deploy_error": "smoke test failed",
                },
            }
        ],
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1/step-40", "--wait", "5"]) == 1
    err = capsys.readouterr().err
    assert "smoke test failed" in err, err
    assert "previously deployed revision is still serving" in err, err
    # the wrong explanation must be gone, not merely accompanied by the right one.
    assert "no longer an active deployment" not in err, err


def test_deploy_wait_reports_a_rollback_from_the_final_adapter(
    fake_client, monkeypatch, capsys
) -> None:
    """A failed final-checkpoint redeploy reports the restored sibling and its error.

    `deploy flash-1/final` asks for the permanent final checkpoint. A run already serving
    `step-20` whose final-checkpoint redeploy fails is restored to step-20, and the exact final
    checkpoint lookup reads as absent. The rollback lookup must retain `last_deploy_error`.
    """
    _queued_deploy(monkeypatch, fake_client)
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {"run_id": run_id, "state": "queued", "requested_at": "T1"},
        raising=False,
    )
    monkeypatch.setattr(
        fake_client, "deployment_for", lambda run_id, timeout=None: None, raising=False
    )
    monkeypatch.setattr(
        fake_client,
        "deployments",
        lambda timeout=None: [
            {
                "run_id": "flash-1",
                "deployment": {
                    "run_id": "flash-1",
                    "checkpoint_step": 20,
                    "state": "ready",
                    "requested_at": "T0",
                    "last_deploy_error": "adapter merge failed",
                },
            }
        ],
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1/final", "--wait", "5"]) == 1
    err = capsys.readouterr().err
    assert "adapter merge failed" in err, err
    assert "previously deployed revision is still serving" in err, err
    assert "no longer an active deployment" not in err, err


def test_deploy_wait_rollback_lookup_stays_inside_the_deadline(fake_client, monkeypatch) -> None:
    """The rollback read is one more read inside the wait, not a second full-length one.

    It runs after a poll that has already spent part of the budget, so bounding it by the
    remainder computed BEFORE that poll hands it time the wait no longer has: a `--wait 5` whose
    poll consumed nearly all five seconds could block for close to ten.
    """
    _queued_deploy(monkeypatch, fake_client)
    clock = {"t": 0.0}
    monkeypatch.setattr(run_commands.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(
        run_commands.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s)
    )

    def _poll(run_id, timeout=None):
        # a stalled plane answers at its bound, which is what leaves nothing for the next read.
        clock["t"] += timeout if timeout is not None else 0.0

    monkeypatch.setattr(fake_client, "deployment_for", _poll, raising=False)
    reads: list[tuple[float, float | None]] = []

    def _listing(timeout=None):
        reads.append((clock["t"], timeout))
        return []

    monkeypatch.setattr(fake_client, "deployments", _listing, raising=False)

    assert _run(["models", "deploy", "flash-1/step-40", "--wait", "5"]) == 1
    assert reads, "the vanished branch issued no rollback lookup"
    for start, bound in reads:
        assert bound is not None, reads
        # the expired case is allowed the zero-wait one-shot bound and nothing wider.
        assert bound <= max(5.0 - start, deploy_commands._DEPLOY_ZERO_WAIT_READ_SECONDS) + 0.001, (
            reads
        )


def test_deploy_wait_rollback_lookup_gets_a_usable_bound_near_the_deadline(
    fake_client, monkeypatch, capsys
) -> None:
    """A remainder can be positive and still be too small to read in.

    The bound above keeps this lookup inside the deadline; this one keeps it able to finish. The
    poll before it lands the wait a hair short of the deadline, so the remainder is positive and
    tiny -- and passing it through means the listing read times out, `_rollback_record` swallows the
    error, and the CLI reports the run as vanished instead of printing `last_deploy_error`. The
    zero-wait floor covers the expired case only, so it does not apply.
    """
    _queued_deploy(monkeypatch, fake_client)
    clock = {"t": 0.0}
    monkeypatch.setattr(run_commands.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(
        run_commands.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s)
    )
    # stop a hair inside the deadline: positive remainder, far too little to complete a read.
    monkeypatch.setattr(
        fake_client,
        "deployment_for",
        lambda run_id, timeout=None: clock.__setitem__("t", 5.0 - 0.002),
        raising=False,
    )
    rolled_back = {
        "run_id": "flash-1",
        "state": "deployed",
        "checkpoint_step": 20,
        "last_deploy_error": "adapter load failed",
    }

    def _listing(timeout=None):
        # a real client cannot answer inside a bound this small; it raises at it.
        if timeout is not None and timeout < deploy_commands._DEPLOY_ZERO_WAIT_READ_SECONDS:
            raise deploy_commands.ClientError("read timed out")
        return [{"run_id": "flash-1", "deployment": rolled_back}]

    monkeypatch.setattr(fake_client, "deployments", _listing, raising=False)

    assert _run(["models", "deploy", "flash-1/step-40", "--wait", "5"]) == 1
    err = capsys.readouterr().err
    assert "adapter load failed" in err, err
    assert "no longer an active deployment" not in err, err


@pytest.mark.parametrize(
    ("rows", "why"),
    [
        ([], "genuinely deleted"),
        (
            [{"run_id": "other", "deployment": {"run_id": "other", "last_deploy_error": "x"}}],
            "another run entirely",
        ),
        (
            [{"run_id": "flash-1", "deployment": {"run_id": "flash-1", "checkpoint_step": 20}}],
            "same run, no recorded error, so nothing ties it to this request",
        ),
    ],
)
def test_deploy_wait_still_reports_a_vanished_deployment_as_vanished(
    fake_client, monkeypatch, capsys, rows, why
) -> None:
    """The rollback lookup must not swallow the deletion case it was added beside.

    Matching on the run id alone is deliberately wider than `deployment_for`, so each of these has
    to stay out: without the `last_deploy_error` requirement this would report an unrelated or
    concurrently-deployed revision as this command's rollback.
    """
    _queued_deploy(monkeypatch, fake_client)
    monkeypatch.setattr(
        fake_client, "deployment_for", lambda run_id, timeout=None: None, raising=False
    )
    monkeypatch.setattr(fake_client, "deployments", lambda timeout=None: rows, raising=False)

    assert _run(["models", "deploy", "flash-1/step-40", "--wait", "5"]) == 1
    assert "no longer an active deployment" in capsys.readouterr().err, why


def test_deploy_wait_survives_a_transient_control_plane_error(
    fake_client, monkeypatch, capsys
) -> None:
    """One failed poll must not fail a deploy that is progressing fine."""
    _queued_deploy(monkeypatch, fake_client)
    results = iter([deploy_commands.ClientError("503"), {"state": "ready"}])

    def _next(run_id, timeout=None):
        value = next(results)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(fake_client, "deployment_for", _next, raising=False)

    assert _run(["models", "deploy", "flash-1/final", "--wait"]) == 0
    assert "ready" in capsys.readouterr().out


def test_deploy_wait_zero_polls_once_instead_of_being_treated_as_no_wait(
    fake_client, monkeypatch, capsys
) -> None:
    """`--wait 0` is an explicit bound, not an absent flag.

    Gating on truthiness makes 0.0 indistinguishable from None, so the one value that means
    "check, but do not block" silently became "do not check at all" and exited 0 on a queued
    record.
    """
    _queued_deploy(monkeypatch, fake_client)
    monkeypatch.setattr(
        fake_client,
        "deployment_for",
        lambda run_id, timeout=None: {"state": "smoke_testing"},
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1/final", "--wait", "0"]) == 1
    assert "waiting up to 0s" in capsys.readouterr().err


def test_deploy_wait_rejects_a_restored_previous_revision(fake_client, monkeypatch, capsys) -> None:
    """A failed redeploy leaves a `ready` record for the PREVIOUS revision.

    mark_deployment_failed restores the old deployment verbatim and records the failure only in
    last_deploy_error, so trusting the state word reports success while the requested checkpoint
    is not the one serving.
    """
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {
            "run_id": run_id,
            "state": "queued",
            "requested_at": "2026-07-29T02:00:00Z",
        },
        raising=False,
    )
    monkeypatch.setattr(run_commands.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        fake_client,
        "deployment_for",
        lambda run_id, timeout=None: {
            "state": "ready",
            "requested_at": "2026-07-29T01:00:00Z",
            "last_deploy_error": "adapter load failed",
        },
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1/final", "--wait"]) == 1
    err = capsys.readouterr().err
    assert "did not become servable" in err
    assert "adapter load failed" in err


def test_deploy_wait_accepts_a_ready_revision_carrying_a_stale_error(
    fake_client, monkeypatch
) -> None:
    """A last_deploy_error from an EARLIER attempt must not fail the attempt that succeeded.

    The stamps match here, so this record is the revision that was just asked for; treating any
    recorded error as failure would make every retry-after-failure report failure forever.
    """
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {
            "run_id": run_id,
            "state": "queued",
            "requested_at": "2026-07-29T02:00:00Z",
        },
        raising=False,
    )
    monkeypatch.setattr(run_commands.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        fake_client,
        "deployment_for",
        lambda run_id, timeout=None: {
            "state": "ready",
            "requested_at": "2026-07-29T02:00:00Z",
            "last_deploy_error": "a previous attempt failed",
        },
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1/final", "--wait"]) == 0


def test_deploy_wait_bounds_each_poll_by_the_remaining_time(
    fake_client, monkeypatch, capsys
) -> None:
    """An unbounded read inside a short wait overshoots the deadline the user set.

    The client default is 60s, so `--wait 5` could block roughly a minute inside a single stalled
    request while reporting that it waited five seconds.
    """
    _queued_deploy(monkeypatch, fake_client)
    seen: list[float | None] = []

    def _poll(run_id, timeout=None):
        seen.append(timeout)
        return {"state": "ready"}

    monkeypatch.setattr(fake_client, "deployment_for", _poll, raising=False)

    assert _run(["models", "deploy", "flash-1/final", "--wait", "5"]) == 0
    assert seen == [pytest.approx(5.0, abs=0.5)]


def test_deploy_wait_zero_actually_reads_the_current_state(fake_client, monkeypatch) -> None:
    """`--wait 0` means "check once, do not block" -- it must issue that one read.

    The deadline was evaluated before the first poll, so a zero budget was already expired on entry
    and deployment_for never ran. Readiness was then judged from the POST body, which is `queued` on
    every normal async deploy, so `--wait 0` could not succeed even against a ready revision.
    """
    _queued_deploy(monkeypatch, fake_client)
    polls: list[str] = []

    def _poll(run_id, timeout=None):
        polls.append(run_id)
        return {"state": "ready"}

    monkeypatch.setattr(fake_client, "deployment_for", _poll, raising=False)

    assert _run(["models", "deploy", "flash-1/final", "--wait", "0"]) == 0
    assert polls == ["flash-1/final"], polls


def test_deploy_wait_does_not_start_a_read_after_the_deadline_expires(
    fake_client, monkeypatch
) -> None:
    """No read may still be running past the deadline the caller set.

    The remaining time was computed once before sleeping, so the post-sleep request still went out
    with the 1.0s floor: `--wait 0.1` against a stalled plane blocked for over a second past the
    bound it advertised.

    Asserted as "every read finishes by the deadline" rather than as a read COUNT. The count was a
    proxy for it under the original behaviour, where the only way to be late was an extra read; it
    stopped tracking the invariant once the final window began funding a read of its own, which is
    a bounded read strictly inside the deadline rather than an overshoot. Keeping the count would
    have made this test forbid the fix to the blind spot it shares a loop with.
    """
    _queued_deploy(monkeypatch, fake_client)
    clock = {"t": 0.0}
    monkeypatch.setattr(run_commands.time, "monotonic", lambda: clock["t"])
    # the sleep is what burns the budget, exactly as a real one would.
    monkeypatch.setattr(
        run_commands.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s)
    )
    reads: list[tuple[float, float | None]] = []

    def _poll(run_id, timeout=None):
        reads.append((clock["t"], timeout))
        return {"state": "queued"}

    monkeypatch.setattr(fake_client, "deployment_for", _poll, raising=False)

    assert _run(["models", "deploy", "flash-1/final", "--wait", "0.1"]) == 1
    assert reads, "the wait issued no read at all"
    # a read starting at `start` and bounded by `bound` occupies the plane until start+bound, so
    # that sum is what has to stay within the advertised wait. this is the assertion the 1.0s floor
    # violated: it put a 0.1s wait on the hook for a full second.
    for start, bound in reads:
        assert bound is not None, reads
        assert start + bound <= 0.1 + 0.001, reads


@pytest.mark.parametrize("state", ["revocation_failed", "some_state_a_newer_plane_added"])
def test_deploy_wait_fails_closed_on_a_terminal_state_that_is_not_ready(
    fake_client, monkeypatch, capsys, state
) -> None:
    """Leaving the busy set is not the same as being servable.

    `revocation_failed` is a real persisted state (a concurrent undeploy whose backend cleanup
    failed), and an unknown state arrives on any client/server skew. Both are non-busy, so gating
    success on "not busy" exited 0 with nothing actually serving.
    """
    _queued_deploy(monkeypatch, fake_client)
    monkeypatch.setattr(
        fake_client,
        "deployment_for",
        lambda run_id, timeout=None: {"state": state},
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1/final", "--wait"]) == 1
    err = capsys.readouterr().err
    assert "not\nservable" in err or "not servable" in err, err
    assert "once it is ready" not in err, err


def test_deploy_wait_rejects_a_superseding_deploy_that_carries_no_error(
    fake_client, monkeypatch, capsys
) -> None:
    """A concurrent deploy for the same run reaches ready on ITS checkpoint, with no error at all.

    Returning early whenever last_deploy_error was absent meant the stamps were never compared on
    exactly the case that needs them, so `deploy --wait && evaluate` reported success and then
    evaluated the other shell's revision.
    """
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {
            "run_id": run_id,
            "state": "queued",
            "requested_at": "2026-07-29T02:00:00Z",
        },
        raising=False,
    )
    monkeypatch.setattr(run_commands.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        fake_client,
        "deployment_for",
        # ready, no error, different attempt: someone else's deploy.
        lambda run_id, timeout=None: {
            "state": "ready",
            "requested_at": "2026-07-29T03:00:00Z",
        },
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1/final", "--wait"]) == 1
    assert "once it is ready" not in capsys.readouterr().err


def test_deploy_wait_observes_readiness_inside_a_short_window(fake_client, monkeypatch) -> None:
    """Sleeping the entire remainder spends the budget without ever looking again.

    With a `--wait 5` and a 5s poll interval, the first read saw `queued`, the sleep consumed all
    five seconds, and the deadline check exited: a revision that became ready one second in was
    still reported as queued and the command exited 1.
    """
    # _queued_deploy stubs sleep to a no-op, so install the clock AFTER it: a frozen monotonic with
    # a non-advancing sleep is an infinite poll, which is a broken test rather than a caught defect.
    _queued_deploy(monkeypatch, fake_client)
    clock = {"t": 0.0}
    monkeypatch.setattr(run_commands.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(
        run_commands.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s)
    )

    def _poll(run_id, timeout=None):
        # ready one second into the five-second window.
        return {"state": "ready" if clock["t"] >= 1.0 else "queued"}

    monkeypatch.setattr(fake_client, "deployment_for", _poll, raising=False)

    assert _run(["models", "deploy", "flash-1/final", "--wait", "5"]) == 0


def test_deploy_wait_observes_readiness_inside_the_final_window(fake_client, monkeypatch) -> None:
    """The last second of a wait must still be watched, not slept through.

    The per-sleep reserve was subtracted only when the slice EXCEEDED it, so a remainder at or under
    the reserve was slept whole and the deadline check ended the wait with no further read. `--wait
    1` therefore could not succeed at all against an async deploy -- one read at t=0, then a full
    second of sleep -- and every longer wait was blind through its final second.
    """
    _queued_deploy(monkeypatch, fake_client)
    clock = {"t": 0.0}
    monkeypatch.setattr(run_commands.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(
        run_commands.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s)
    )

    def _poll(run_id, timeout=None):
        # ready half a second into the one-second window.
        return {"state": "ready" if clock["t"] >= 0.5 else "queued"}

    monkeypatch.setattr(fake_client, "deployment_for", _poll, raising=False)

    assert _run(["models", "deploy", "flash-1/final", "--wait", "1"]) == 0


def test_deploy_wait_watches_the_final_window_to_its_deadline(fake_client, monkeypatch) -> None:
    """The final window must be watched to its END, not only to its midpoint.

    Splitting that window put its one read halfway through and then stopped, so the wait returned
    with half its advertised budget unspent: `--wait 1` read at t=0 and t=0.5 and reported a timeout
    for a revision that went ready at t=0.75. Sleeping the window whole
    and reading at the deadline covers it without adding a read.
    """
    _queued_deploy(monkeypatch, fake_client)
    clock = {"t": 0.0}
    monkeypatch.setattr(run_commands.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(
        run_commands.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s)
    )

    def _poll(run_id, timeout=None):
        # ready in the second half of the window -- past a midpoint read, inside the deadline.
        return {"state": "ready" if clock["t"] >= 0.75 else "queued"}

    monkeypatch.setattr(fake_client, "deployment_for", _poll, raising=False)

    assert _run(["models", "deploy", "flash-1/final", "--wait", "1"]) == 0


def test_deploy_wait_final_window_does_not_poll_unboundedly(fake_client, monkeypatch) -> None:
    """Splitting the final window must be the wait's last sleep, not a converging series.

    Reserving a FRACTION of the remainder rather than a fixed slice never drives the remainder to
    zero, so the loop terminates only on the clock's granularity. Against a stalled plane that is an
    unbounded burst of reads inside the last second -- the failure mode the fixed reserve was chosen
    to avoid, reintroduced at the one point the reserve does not apply.
    """
    _queued_deploy(monkeypatch, fake_client)
    clock = {"t": 0.0}
    monkeypatch.setattr(run_commands.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(
        run_commands.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s)
    )
    reads = []

    def _poll(run_id, timeout=None):
        reads.append(clock["t"])
        # never settles: the wait has to end on its own budget, not on the plane's answer.
        return {"state": "queued"}

    monkeypatch.setattr(fake_client, "deployment_for", _poll, raising=False)

    assert _run(["models", "deploy", "flash-1/final", "--wait", "1"]) == 1
    # the up-front read plus the one the split funds. a fractional reserve makes this grow without
    # bound; asserting the exact count is what keeps the split from silently becoming that.
    assert len(reads) == 2, reads


def test_deploy_wait_zero_does_not_block_past_its_own_bound(fake_client, monkeypatch) -> None:
    """`--wait 0` advertises "check once, do not block", so its one read must be bounded tightly.

    A ten-second fixed budget let a stalled plane hold a zero-second wait for ten seconds, which is
    the same overshoot the per-poll bound exists to prevent, just smaller.
    """
    _queued_deploy(monkeypatch, fake_client)
    seen: list[float | None] = []

    def _poll(run_id, timeout=None):
        seen.append(timeout)
        return {"state": "ready"}

    monkeypatch.setattr(fake_client, "deployment_for", _poll, raising=False)

    assert _run(["models", "deploy", "flash-1/final", "--wait", "0"]) == 0
    assert seen == [pytest.approx(1.0, abs=0.001)], seen


def test_deploy_wait_rejects_a_synchronous_failure_that_returns_the_restored_revision(
    fake_client, monkeypatch, capsys
) -> None:
    """A synchronous deploy returns the FINISHED record, never the queued attempt.

    Under FLASH_DEPLOY_SYNC the POST answers after the job ran, so on failure it returns the
    restored previous `ready` revision. requested and final are then the same row, their stamps
    match by construction, and comparing identity accepted a deploy that never happened.
    """
    settled = {
        "run_id": "flash-1",
        "state": "ready",
        "requested_at": "2026-07-29T01:00:00Z",
        "last_deploy_error": "adapter load failed",
    }
    monkeypatch.setattr(fake_client, "deploy", lambda run_id, **_: dict(settled), raising=False)
    # non-busy on arrival, so _await_deployment returns it without polling at all.
    monkeypatch.setattr(
        fake_client, "deployment_for", lambda run_id, timeout=None: dict(settled), raising=False
    )

    assert _run(["models", "deploy", "flash-1/final", "--wait"]) == 1
    err = capsys.readouterr().err
    assert "did not become servable" in err, err
    assert "adapter load failed" in err, err


def test_deploy_wait_accepts_a_synchronous_success(fake_client, monkeypatch) -> None:
    """The synchronous check keys on a recorded error, so a clean sync deploy still succeeds."""
    settled = {"run_id": "flash-1", "state": "ready", "requested_at": "2026-07-29T02:00:00Z"}
    monkeypatch.setattr(fake_client, "deploy", lambda run_id, **_: dict(settled), raising=False)
    monkeypatch.setattr(
        fake_client, "deployment_for", lambda run_id, timeout=None: dict(settled), raising=False
    )

    assert _run(["models", "deploy", "flash-1/final", "--wait"]) == 0


def test_deploy_notes_name_this_channels_executable(fake_client, monkeypatch, capsys) -> None:
    """The dev channel installs `flash-dev`; a hardcoded `flash ...` hint is not runnable there."""
    monkeypatch.setattr(deploy_commands, "CLI_NAME", "flash-dev")
    _queued_deploy(monkeypatch, fake_client)

    assert _run(["models", "deploy", "flash-1/final"]) == 0
    err = capsys.readouterr().err
    assert "flash-dev models deployments" in err, err
    assert "`flash models" not in err, err


def test_deploy_wait_stops_retrying_a_rejected_key(fake_client, monkeypatch, capsys) -> None:
    """401/403 answers the same way every time, so polling through it just burns the timeout.

    The broad ClientError catch treated a permanent rejection as a transient blip and retried to
    the full default 30-minute deadline before reporting "still queued".
    """
    _queued_deploy(monkeypatch, fake_client)
    calls: list[int] = []

    def _denied(run_id, timeout=None):
        calls.append(1)
        raise deploy_commands.ApiError(403, "forbidden")

    monkeypatch.setattr(fake_client, "deployment_for", _denied, raising=False)

    assert _run(["models", "deploy", "flash-1/final", "--wait"]) == 1
    assert len(calls) == 1
    assert "cannot check flash-1" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-1"])
def test_deploy_wait_rejects_a_timeout_that_would_never_expire(fake_client, value, capsys) -> None:
    """`float` accepts nan and inf, and a NaN deadline makes every `remaining <= 0` false.

    The loop then polls forever while the user believes they set a bound, which is worse than the
    unbounded default because the printed timeout says otherwise. Written as `--wait=VALUE` so a
    leading-dash value reaches the validator instead of being read as another option.
    """
    with pytest.raises(SystemExit) as excinfo:
        _run(["models", "deploy", "flash-1/final", f"--wait={value}"])
    assert excinfo.value.code == 2
    assert "--wait" in capsys.readouterr().err


def test_deploy_wait_before_the_run_id_names_the_real_mistake(fake_client, capsys) -> None:
    """`--wait` takes an optional value, so `deploy --wait flash-1` eats the run id.

    argparse cannot hand the token back, so the error has to say which argument was swallowed;
    the bare "invalid float value: 'flash-1'" reads as if the run id itself were malformed.
    """
    with pytest.raises(SystemExit) as excinfo:
        _run(["models", "deploy", "--wait", "flash-1"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "if 'flash-1' is the run id" in err
    assert "deploy flash-1 --wait" in err


def test_deploy_wait_skips_polling_for_a_dry_run(fake_client, monkeypatch, capsys) -> None:
    """A dry run creates no deployment, so there is nothing to wait on."""
    monkeypatch.setattr(
        fake_client,
        "deploy",
        lambda run_id, **_: {"run_id": run_id, "state": "dry_run"},
        raising=False,
    )

    assert _run(["models", "deploy", "flash-1/final", "--dry-run", "--wait"]) == 0
    assert not any(c[0] == "deployment_for" for c in fake_client.calls)
    assert "ctrl-c stops waiting" not in capsys.readouterr().err


def test_log_follow_progress_includes_heartbeat_age() -> None:
    """The follow spinner must show a live heartbeat age so a long quiet phase reads as
    "alive, throttled" instead of a frozen line."""
    import time as _time

    from flash.cli.commands.ops.runs import _log_follow_progress

    status = {
        "state": "running",
        "last_heartbeat": {"stage": "sft_initializing", "step": 3, "ts": _time.time() - 41},
    }
    state, progress = _log_follow_progress(status, "unknown")
    assert state == "running"
    assert "stage=sft_initializing" in progress
    assert "step=3" in progress
    assert "hb=<1m" in progress

    status["last_heartbeat"]["ts"] = _time.time() - 500
    _, progress = _log_follow_progress(status, "unknown")
    assert "hb=8m" in progress

    state, progress = _log_follow_progress({"state": "running"}, "unknown")
    assert "hb=" not in progress  # no heartbeat yet -> no fabricated age

    malformed = {"state": "running", "last_heartbeat": {"stage": "sft_step", "ts": "oops"}}
    _, progress = _log_follow_progress(malformed, "unknown")
    assert "hb=" not in progress  # non-numeric ts -> no fabricated age


def test_log_follow_progress_names_the_attempt_after_a_relaunch() -> None:
    """A preemption relaunch rewinds the step counter while the state stays "running", so the
    follow line must name the attempt or the rewind reads as lost progress with no cause."""
    import time as _time

    from flash.cli.commands.ops.runs import _log_follow_progress

    # attempts are 0-based, so a first attempt must stay unannotated.
    first = {
        "state": "running",
        "last_heartbeat": {"stage": "sft_step", "step": 455, "ts": _time.time(), "attempt": 0},
    }
    _, progress = _log_follow_progress(first, "unknown")
    assert "step=455" in progress
    assert "attempt=" not in progress

    # relaunched on fresh hardware: same state, step back to 0, attempt incremented.
    relaunched = {
        "state": "running",
        "remote": {"attempt": 1},
        "last_heartbeat": {"stage": "boot", "step": 0, "ts": _time.time(), "attempt": 1},
    }
    state, progress = _log_follow_progress(relaunched, "unknown")
    assert state == "running"
    assert "step=0" in progress
    assert "attempt=1" in progress

    malformed = {
        "state": "running",
        "last_heartbeat": {"stage": "sft_step", "ts": _time.time(), "attempt": "two"},
    }
    _, progress = _log_follow_progress(malformed, "unknown")
    assert "attempt=" not in progress  # non-integer attempt -> no fabricated identity

    # the relaunch window this line exists to explain: `remote.attempt` has advanced but the
    # replacement worker has not published a heartbeat yet, so `last_heartbeat` is still the
    # superseded attempt's ping. reading the heartbeat here leaves the first preemption entirely
    # unlabelled (its stale attempt is 0) and names the previous attempt on every later one.
    mid_relaunch = {
        "state": "running",
        "remote": {"attempt": 1},
        "last_heartbeat": {"stage": "sft_step", "step": 455, "ts": _time.time(), "attempt": 0},
    }
    _, progress = _log_follow_progress(mid_relaunch, "unknown")
    assert "attempt=1" in progress
    # ...and the step it is printed next to belongs to attempt 0, so it must not read as attempt
    # 1's progress. see test_log_follow_progress_marks_a_stale_heartbeats_fields.
    assert "(prev attempt)" in progress

    # `remote` is absent on planes that do not surface it, so the heartbeat still has to answer.
    no_remote = {
        "state": "running",
        "last_heartbeat": {"stage": "sft_step", "step": 12, "ts": _time.time(), "attempt": 2},
    }
    _, progress = _log_follow_progress(no_remote, "unknown")
    assert "attempt=2" in progress


@pytest.mark.parametrize("heartbeat_attempt", [0, 1])
def test_log_follow_progress_does_not_trust_a_ping_left_by_a_cleared_remote(
    heartbeat_attempt: int,
) -> None:
    """A supervised retry publishes `remote: null` for its whole allocation window.

    `flash/runner/supervise/lifecycle.py` clears `remote` before reserving the replacement attempt and does
    not persist the new one until the provider handle lands, so throughout that window flash serves
    a running record whose only attempt identity is the superseded worker's ping. Falling back to it
    there reintroduced exactly what preferring `remote` was meant to fix: the first retry unlabelled
    (its stale attempt is 0), later ones naming the *previous* attempt.

    Both parametrizations are the same window one retry apart, and both must refuse the ping.

    Keyed on an explicit null rather than a missing key, which is what makes it decidable:
    `on_handle` persists `remote` in the same `_update` that sets `running`, so a running flash
    record with a heartbeat and no remote is a worker already torn down. An ABSENT `remote` still
    falls back -- that is a plane which never surfaces the field, covered above.
    """
    import time as _time

    from flash.cli.commands.ops.runs import _log_follow_progress

    _, progress = _log_follow_progress(
        {
            "state": "running",
            "remote": None,
            "last_heartbeat": {
                "stage": "sft_step",
                "step": 455,
                "ts": _time.time(),
                "attempt": heartbeat_attempt,
            },
        },
        "unknown",
    )
    # no identity is claimed: the replacement is not reserved yet, and a wrong number is worse than
    # none for the one field that exists to explain the rewind.
    assert "attempt=" not in progress, progress
    # ...and the ping it did print belongs to the torn-down worker, so it must not read as the
    # replacement's progress.
    assert "step=455" in progress, progress
    assert "(prev attempt)" in progress, progress
    assert progress.index("step=455") < progress.index("(prev attempt)"), progress


def test_log_follow_progress_names_the_attempt_before_the_first_heartbeat() -> None:
    """The relaunch has to be named while the replacement worker is still cold.

    An attempt preempted before it published a ping leaves `last_heartbeat` absent while
    `remote.attempt` has already advanced. Resolving the attempt inside the heartbeat block made
    the line print a bare `running` for the whole cold start -- silent through exactly the window
    the attempt counter exists to explain, and the window a user is most likely to be watching.
    """
    from flash.cli.commands.ops.runs import _log_follow_progress

    state, progress = _log_follow_progress(
        {"state": "running", "remote": {"attempt": 1}, "last_heartbeat": None},
        "running",
    )
    assert state == "running"
    assert "attempt=1" in progress, progress
    # nothing heartbeat-sourced exists to qualify, so the marker would have nothing to cover.
    assert "(prev attempt)" not in progress, progress

    # still 0-based with no heartbeat: a first attempt stays unannotated.
    _, first = _log_follow_progress(
        {"state": "running", "remote": {"attempt": 0}, "last_heartbeat": None},
        "running",
    )
    assert "attempt=" not in first, first


def test_log_follow_progress_marks_a_stale_heartbeats_fields() -> None:
    """stage/step come from the heartbeat, attempt from `remote` -- during a relaunch those differ.

    Printed unqualified, `stage=sft_step step=455 attempt=1` says the replacement worker has run
    455 steps. It has not: it restarted from zero, and 455 is the superseded worker's last ping.
    That is the exact rewind the attempt counter was added to explain, reported as if it never
    happened.

    Marked rather than suppressed: the run did reach step 455, and dropping the fields entirely
    would read as a worker that has produced nothing.
    """
    import time as _time

    from flash.cli.commands.ops.runs import _log_follow_progress

    _, progress = _log_follow_progress(
        {
            "state": "running",
            "remote": {"attempt": 1},
            "last_heartbeat": {"stage": "sft_step", "step": 455, "ts": _time.time(), "attempt": 0},
        },
        "unknown",
    )
    assert "step=455" in progress
    assert "attempt=1" in progress
    assert "(prev attempt)" in progress, progress

    # the marker's scope is positional, so ordering is the contract: everything before it came
    # from the superseded ping, everything after is live. `hb=` has to sit inside that span --
    # the age is the old worker's ping too, so a fresh `hb=<1m` printed past the marker reads as
    # the replacement worker being alive when nothing has been heard from it at all.
    assert progress.index("step=455") < progress.index("(prev attempt)"), progress
    assert progress.index("hb=") < progress.index("(prev attempt)"), progress
    assert progress.index("(prev attempt)") < progress.index("attempt=1"), progress

    # a heartbeat from the live attempt is not stale, so nothing is marked.
    _, current = _log_follow_progress(
        {
            "state": "running",
            "remote": {"attempt": 1},
            "last_heartbeat": {"stage": "sft_step", "step": 455, "ts": _time.time(), "attempt": 1},
        },
        "unknown",
    )
    assert "step=455" in current
    assert "(prev attempt)" not in current, current

    # no `remote` means the live attempt is unknown, so staleness is unprovable -- marking there
    # would label every ordinary heartbeat on a plane that does not surface `remote`.
    _, no_remote = _log_follow_progress(
        {
            "state": "running",
            "last_heartbeat": {"stage": "sft_step", "step": 12, "ts": _time.time(), "attempt": 2},
        },
        "unknown",
    )
    assert "(prev attempt)" not in no_remote, no_remote


@pytest.mark.parametrize("stage", ["rl_train_start", "rl_initializing"])
def test_log_follow_progress_explains_rl_warmup(stage: str) -> None:
    import time as _time

    from flash.cli.commands.ops.runs import _log_follow_progress

    status = {"state": "running", "last_heartbeat": {"stage": stage, "ts": _time.time()}}
    _, progress = _log_follow_progress(status, "unknown")

    assert f"warming up (stage={stage})" in progress
    assert "typically several minutes, sometimes 15-20 min" in progress
    assert "setup is not billed" in progress
    assert "do not cancel" in progress

    status["last_heartbeat"]["stage"] = "rl_step"
    _, progress = _log_follow_progress(status, "unknown")
    assert "warming up" not in progress
    assert "not billed" not in progress


@pytest.mark.parametrize("stage", ["rl_train_start", "rl_initializing"])
def test_log_follow_progress_omits_warmup_claim_for_stale_heartbeat(stage: str) -> None:
    import time as _time

    from flash.cli.commands.ops.runs import _log_follow_progress

    status = {
        "state": "running",
        "last_heartbeat": {"stage": stage, "ts": _time.time() - 1201},
    }
    _, progress = _log_follow_progress(status, "unknown")

    assert f"stage={stage}" in progress
    assert "hb=20m" in progress
    assert "warming up" not in progress
    assert "do not cancel" not in progress


@pytest.mark.parametrize("stage", ["rl_train_start", "rl_initializing"])
def test_log_follow_progress_omits_warmup_claim_for_prior_attempt_heartbeat(stage: str) -> None:
    import time as _time

    from flash.cli.commands.ops.runs import _log_follow_progress

    # remote is on attempt 1 while last_heartbeat is the previous attempt's fresh setup ping: the
    # warmup reassurance must not fire against a superseded attempt before the new worker publishes.
    status = {
        "state": "running",
        "remote": {"attempt": 1},
        "last_heartbeat": {"stage": stage, "ts": _time.time(), "attempt": 0},
    }
    _, progress = _log_follow_progress(status, "unknown")

    assert f"stage={stage}" in progress
    assert "warming up" not in progress
    assert "do not cancel" not in progress


@pytest.mark.parametrize("stage", ["rl_train_start", "rl_initializing"])
def test_log_follow_progress_explains_warmup_when_heartbeat_matches_attempt(stage: str) -> None:
    import time as _time

    from flash.cli.commands.ops.runs import _log_follow_progress

    status = {
        "state": "running",
        "remote": {"attempt": 3},
        "last_heartbeat": {"stage": stage, "ts": _time.time(), "attempt": 3},
    }
    _, progress = _log_follow_progress(status, "unknown")

    assert f"warming up (stage={stage})" in progress
    assert "do not cancel" in progress


# --------------------------------------------------------------------------- live-run cost (MP-022/LS-016)


def test_runs_listing_flags_a_live_cost_as_an_estimate(fake_client, monkeypatch, capsys) -> None:
    """A queued/running run reports cost_usd 0.0 until it settles; showing that bare reads as free."""
    monkeypatch.setattr(cli.render, "styled", lambda: False)
    monkeypatch.setattr(
        fake_client,
        "list_runs",
        lambda: [
            {
                "run_id": "flash-live",
                "state": "running",
                "cost_usd": 0.0,
                "estimated_cost_usd": 2.5,
                "updated_at": 1700000000.0,
                "spec": {"model": "Qwen/Qwen3.5-9B", "algorithm": "grpo"},
            }
        ],
    )
    assert _run(["runs", "list"]) == 0
    out = capsys.readouterr().out
    assert "~2.5000" in out
    assert "0.0000" not in out


def test_runs_listing_shows_the_settled_charge_unflagged(fake_client, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.render, "styled", lambda: False)
    assert _run(["runs", "list"]) == 0
    out = capsys.readouterr().out
    assert "0.2500" in out  # the fixture run is `done`
    assert "~" not in out


def test_log_follow_progress_reports_the_quote_while_live() -> None:
    _state, progress = run_commands._log_follow_progress(
        {"state": "running", "cost_usd": 0.0, "estimated_cost_usd": 2.5}, "running"
    )
    assert "cost=~$2.5000" in progress


def test_log_follow_progress_reports_the_settled_charge_when_done() -> None:
    _state, progress = run_commands._log_follow_progress(
        {"state": "done", "cost_usd": 1.25, "estimated_cost_usd": 2.5}, "done"
    )
    assert "cost=$1.2500" in progress
    assert "~" not in progress


def test_log_follow_progress_omits_cost_when_there_is_nothing_to_show() -> None:
    # no quote and no measured spend: don't print a misleading "cost=$0.0000".
    _state, progress = run_commands._log_follow_progress({"state": "queued"}, "queued")
    assert "cost=" not in progress


def test_log_follow_progress_shows_a_settled_zero_like_the_other_surfaces() -> None:
    """A terminal $0.0000 is an answer, and the three surfaces have to give the same one.

    `runs list` and `runs status` both print $0.0000 for a settled zero because run_cost returns
    (0.0, False) there. Suppressing it only in follow made the same finished run read as costed in
    one place and uncosted in another, which is the inconsistency, not the zero.
    """
    for state in sorted(cost_ui.SETTLED_COST_STATES):
        _state, progress = run_commands._log_follow_progress(
            {"state": state, "cost_usd": 0.0}, state
        )
        assert "cost=$0.0000" in progress, state
        assert "~" not in progress, state


@pytest.mark.parametrize(
    "api_url",
    [
        "http://your-plane:8080",
        "http://10.0.0.5:8080",
        "http://plane.internal",
    ],
)
def test_login_warns_before_sending_the_key_over_plaintext_http(monkeypatch, capsys, api_url):
    """The warning must reach the user BEFORE the key is transmitted.

    On a standalone plane FREESOLO_INTERNAL_KEY is the entire authorization boundary and owns every
    run, and it rides as a bearer header on login and every command after it. SELF_HOSTING.md used
    to present a plaintext remote URL as fine, so following the quickstart put a root-equivalent
    credential on the wire in clear text.
    """
    warned_before_request = {}

    def _verify(api_key, url):
        warned_before_request["stderr"] = capsys.readouterr().err
        return {"email": "operator@example.com"}

    monkeypatch.setattr(account_commands, "_verify_key_against_plane", _verify)
    monkeypatch.setattr(account_commands, "save_credentials", lambda *a, **k: None)
    monkeypatch.setattr(account_commands, "_identity_or_none", lambda *a, **k: None)
    monkeypatch.delenv("FREESOLO_API_KEY", raising=False)

    args = types.SimpleNamespace(api_url=api_url, api_key="operator-key", debug=False)
    assert account_commands.cmd_login(args) == 0
    assert "plaintext HTTP" in warned_before_request["stderr"]


@pytest.mark.parametrize(
    "api_url",
    [
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "https://your-plane.example",
    ],
)
def test_login_stays_quiet_for_loopback_and_tls(monkeypatch, capsys, api_url):
    """Local development over http is the one safe plaintext case, and https is the fix."""
    monkeypatch.setattr(
        account_commands, "_verify_key_against_plane", lambda *a, **k: {"email": "op@example.com"}
    )
    monkeypatch.setattr(account_commands, "verify_freesolo_key", lambda *a, **k: None)
    monkeypatch.setattr(account_commands, "save_credentials", lambda *a, **k: None)
    monkeypatch.setattr(account_commands, "_identity_or_none", lambda *a, **k: None)
    monkeypatch.delenv("FREESOLO_API_KEY", raising=False)

    args = types.SimpleNamespace(api_url=api_url, api_key="operator-key", debug=False)
    assert account_commands.cmd_login(args) == 0
    assert "plaintext HTTP" not in capsys.readouterr().err


def test_login_warns_about_a_plaintext_freesolo_url_behind_an_https_plane(monkeypatch, capsys):
    """The identity backend is a SECOND destination for the key, and it must be checked too.

    `cmd_login` sends the key to whichever of the two urls the branch below it selects. Warning only
    on `api_url` left an https plane paired with an `http://` --freesolo-url completely silent --
    the configuration where the operator has most reason to believe the key is protected.
    """
    warned_before_request = {}

    def _verify(api_key, *, base_url):
        warned_before_request["stderr"] = capsys.readouterr().err

    monkeypatch.setattr(account_commands, "verify_freesolo_key", _verify)
    monkeypatch.setattr(account_commands, "_verifies_against_freesolo", lambda *a, **k: True)
    monkeypatch.setattr(account_commands, "save_credentials", lambda *a, **k: None)
    monkeypatch.setattr(account_commands, "_identity_or_none", lambda *a, **k: None)
    monkeypatch.delenv("FREESOLO_API_KEY", raising=False)

    args = types.SimpleNamespace(
        api_url="https://your-plane.example",
        freesolo_url="http://identity.internal:8000",
        api_key="operator-key",
        debug=False,
    )
    assert account_commands.cmd_login(args) == 0
    stderr = warned_before_request["stderr"]
    assert "plaintext HTTP" in stderr, stderr
    # it names the offending url, not the https plane that is already fine.
    assert "identity.internal" in stderr
    assert "your-plane.example is plaintext" not in stderr


def test_login_does_not_warn_twice_for_one_url():
    """A plane that is also its own identity backend is one destination, so it warns once."""
    warnings = account_commands._plaintext_login_warnings(
        "http://plane.example:8080", "http://plane.example:8080"
    )
    assert len(warnings) == 1, warnings


def test_login_warns_about_a_plaintext_freesolo_base_url_from_the_environment(monkeypatch, capsys):
    """The identity url is RESOLVED, not passed, so the env var is a real destination for the key.

    `verify_freesolo_key` calls `freesolo_base_url(override)`, which falls back to
    FREESOLO_BASE_URL when no --freesolo-url is given. Reading only the CLI arg therefore left the
    env-var spelling of the exact same destination silent, while the key still went to it.
    """
    warned_before_request = {}

    def _verify(api_key, *, base_url):
        warned_before_request["stderr"] = capsys.readouterr().err

    monkeypatch.setattr(account_commands, "verify_freesolo_key", _verify)
    monkeypatch.setattr(account_commands, "save_credentials", lambda *a, **k: None)
    monkeypatch.setattr(account_commands, "_identity_or_none", lambda *a, **k: None)
    monkeypatch.delenv("FREESOLO_API_KEY", raising=False)
    monkeypatch.setenv("FREESOLO_BASE_URL", "http://identity.internal.test")

    # a hosted plane, so login really does route the key to the identity backend.
    args = types.SimpleNamespace(
        api_url="https://api.freesolo.co", api_key="operator-key", debug=False
    )
    assert account_commands.cmd_login(args) == 0
    stderr = warned_before_request["stderr"]
    assert "plaintext HTTP" in stderr, stderr
    assert "identity.internal.test" in stderr


def test_login_ignores_the_identity_url_when_the_key_never_goes_there(monkeypatch):
    """A self-hosted plane verifies its own key, so an http identity url receives nothing.

    Warning about a destination this login will not contact would train operators to ignore the
    warning that matters, so the resolution is gated on the same predicate that does the routing.
    """
    monkeypatch.setenv("FREESOLO_BASE_URL", "http://identity.internal.test")

    warnings = account_commands._plaintext_login_warnings("https://plane.example", None)
    assert warnings == [], warnings


def test_hosted_key_rejection_names_the_url_that_rejected_it(monkeypatch):
    """A 401 must name the service that answered, not only the key.

    The same 401 is what a perfectly VALID key gets when the request reached the wrong issuer --
    a leftover localhost from a self-hosted experiment, an overridden FREESOLO_BASE_URL. A message
    that names only the key accuses the one input the user just copied correctly and never shows
    the one that was actually wrong.

    The remedy must name the knobs this URL is actually read from. ``verify_freesolo_key`` resolves
    through ``freesolo_base_url`` (--freesolo-url / FREESOLO_BASE_URL); --api-url selects the Flash
    control plane and cannot change who answered here, so naming it sends the user to a flag that
    does nothing.
    """
    import urllib.error

    from flash.client.http import ClientError, verify_freesolo_key

    def _reject(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", _reject)
    monkeypatch.setenv("FREESOLO_BASE_URL", "http://localhost:9999")

    with pytest.raises(ClientError) as exc:
        verify_freesolo_key("fs-key")
    assert "localhost:9999" in str(exc.value)
    assert "--freesolo-url" in str(exc.value)
    assert "--api-url" not in str(exc.value)


def test_key_rejection_survives_a_url_whose_port_is_not_a_number(monkeypatch):
    """A malformed port must not turn the login failure into a traceback.

    ``urlsplit`` defers validation to its accessors, so a bad port raises on the ``.port`` read
    rather than at parse time. This helper builds the ERROR message, so a raise inside it replaces
    the friendly ClientError with a ValueError from the reporting path -- the user then sees a
    stack trace instead of being told their key was rejected.
    """
    import urllib.error

    from flash.client.http import ClientError, verify_freesolo_key

    def _reject(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", _reject)
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://identity.example:notaport")

    with pytest.raises(ClientError) as exc:
        verify_freesolo_key("fs-key")
    assert "rejected this API key" in str(exc.value)


def test_displayable_url_keeps_ipv6_brackets_and_rejects_bad_ports():
    """An IPv6 host must stay bracketed, and an unreadable authority must degrade, not raise.

    ``hostname`` strips the brackets an IPv6 literal needs, so appending a port yields
    ``2001:db8::1:8443`` -- an address the reader cannot split back into host and port. A URL
    printed in an error is meant to be copied, so it has to survive the round trip.
    """
    from flash.serve.contract.urls import displayable_url

    assert displayable_url("https://[2001:db8::1]:8443") == "https://[2001:db8::1]:8443"
    assert displayable_url("https://[2001:db8::1]") == "https://[2001:db8::1]"
    assert displayable_url("https://[::1]:80") == "https://[::1]:80"
    # userinfo is still dropped when the host is IPv6.
    assert displayable_url("https://user:pw@[2001:db8::1]:443") == "https://[2001:db8::1]:443"
    # every unreadable authority degrades to the placeholder rather than raising.
    for bad in ("https://identity.example:notaport", "https://host:99999999", "https://[bad::ipv6"):
        assert displayable_url(bad) == "(unparseable url)", bad

    # port 0 is a real configured value and an invalid endpoint. dropping it on truthiness renders
    # the default port instead, hiding the exact setting the reader has to correct.
    assert displayable_url("http://localhost:0") == "http://localhost:0"
    assert displayable_url("http://localhost") == "http://localhost"


def test_hosted_key_rejection_does_not_echo_credentials_from_the_base_url(monkeypatch):
    """The URL named in the error must never carry the secret it was configured with.

    A base URL is user-supplied and can hold credentials in its authority or a token in its query.
    This message is printed to stderr and pasted into bug reports, so echoing the configured URL
    verbatim turns a login failure into credential disclosure.
    """
    import urllib.error

    from flash.client.http import ClientError, verify_freesolo_key

    def _reject(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", _reject)
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://admin:hunter2@api.example.co/?token=t0ps3cret")

    with pytest.raises(ClientError) as exc:
        verify_freesolo_key("fs-key")
    message = str(exc.value)
    assert "api.example.co" in message  # the service that answered is still named
    for secret in ("hunter2", "admin", "t0ps3cret"):
        assert secret not in message, message


def test_unreachable_backend_error_does_not_echo_credentials_from_the_base_url(monkeypatch):
    """The connection-failure path prints the same URL and needs the same redaction.

    Fixing only the 401 leaves the secret to escape through whichever error the user actually hits;
    an unreachable host is the more common one.
    """
    import urllib.error

    from flash.client.http import ClientError, verify_freesolo_key

    def _unreachable(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _unreachable)
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://admin:hunter2@api.example.co")

    with pytest.raises(ClientError) as exc:
        verify_freesolo_key("fs-key")
    assert "api.example.co" in str(exc.value)
    assert "hunter2" not in str(exc.value)


def test_follow_survives_a_transient_502_instead_of_reporting_a_failed_submit(
    monkeypatch, capsys
) -> None:
    """The reported bug: one 502 on the polling endpoints aborted the whole follow.

    `flash train` submits, then follows. The run existed and trained to completion, but the
    CLI exited nonzero with a bare 502, which reads as "the submit failed" -- so the user
    submits again and pays for two GPUs.
    """

    class _FlakyClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.log_calls = 0

        def get_logs(self, run_id: str, offset: int = 0) -> dict:
            self.log_calls += 1
            if self.log_calls == 1:
                raise deploy_commands.ApiError(502, "HTTP Error 502: Bad Gateway")
            return {"run_id": run_id, "logs": "trained\n", "offset": 8, "state": "done"}

        def get_run(self, run_id: str) -> dict:
            return {"run_id": run_id, "state": "done"}

    monkeypatch.setattr(run_commands.time, "sleep", lambda _seconds: None)
    client = _FlakyClient()

    assert run_commands._follow_run(client, "flash-502") == 0

    captured = capsys.readouterr()
    assert "trained" in captured.out
    # the retry is announced, so a paused stream does not read as a stalled run.
    assert "retrying" in captured.err
    assert "the run is unaffected" in captured.err


def test_follow_does_not_reprint_the_log_page_it_retried(monkeypatch, capsys) -> None:
    """A retry must resume at the same offset, not replay bytes already on screen.

    `get_run` fails after `get_logs` already returned a page. If the retry advanced the offset
    or re-emitted that page, the user sees the same worker output twice.
    """

    class _StatusFlakyClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.offsets: list[int] = []
            self.run_calls = 0

        def get_logs(self, run_id: str, offset: int = 0) -> dict:
            self.offsets.append(offset)
            if offset == 0:
                return {"run_id": run_id, "logs": "step 1\n", "offset": 7, "state": "running"}
            return {"run_id": run_id, "logs": "", "offset": offset, "state": "done"}

        def get_run(self, run_id: str) -> dict:
            self.run_calls += 1
            if self.run_calls == 1:
                raise deploy_commands.ApiError(503, "HTTP Error 503: Service Unavailable")
            return {"run_id": run_id, "state": "running" if self.run_calls == 2 else "done"}

    monkeypatch.setattr(run_commands.time, "sleep", lambda _seconds: None)
    client = _StatusFlakyClient()

    result = run_commands._poll_logs(client, "flash-offset", interval=0)

    assert result.state == "done"
    assert capsys.readouterr().out.count("step 1") == 1
    # the failed round re-reads offset 0; it advances only once the pair of calls both land.
    assert client.offsets == [0, 0, 7]


def test_follow_gives_up_after_the_retry_window_and_names_the_run(monkeypatch, capsys) -> None:
    """A genuinely dead plane must still end -- naming the run, never as a failed submit."""

    class _DeadClient(_FakeClient):
        def get_logs(self, run_id: str, offset: int = 0) -> dict:
            raise deploy_commands.ApiError(502, "HTTP Error 502: Bad Gateway")

        def get_run(self, run_id: str) -> dict:
            raise deploy_commands.ApiError(502, "HTTP Error 502: Bad Gateway")

    # the retry budget reads the clock in `log_follow`; patch it there rather than relying on
    # `run_commands.time` happening to be the same module object.
    clock = iter([0.0] + [1000.0] * 20)
    monkeypatch.setattr(log_follow.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(log_follow.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(run_commands.time, "sleep", lambda _seconds: None)

    assert run_commands._follow_run(_DeadClient(), "flash-dead") == 1

    err = capsys.readouterr().err
    assert "still going and still billing" in err
    # both recovery commands, so the next move is never a duplicate submit.
    assert "runs log flash-dead --follow" in err
    assert "runs cancel flash-dead" in err


def test_follow_surfaces_a_client_error_immediately_without_retrying(monkeypatch) -> None:
    """A 4xx is a real answer about this request, not a blip. Retrying it just delays the truth."""

    class _UnauthorizedClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.log_calls = 0

        def get_logs(self, run_id: str, offset: int = 0) -> dict:
            self.log_calls += 1
            raise deploy_commands.ApiError(401, "invalid or missing API key")

        def get_run(self, run_id: str) -> dict:
            return {"run_id": run_id, "state": "running"}

    monkeypatch.setattr(run_commands.time, "sleep", lambda _seconds: None)
    client = _UnauthorizedClient()

    with pytest.raises(deploy_commands.ApiError) as excinfo:
        run_commands._poll_logs(client, "flash-401", interval=0)

    assert excinfo.value.status == 401
    assert client.log_calls == 1


def test_status_follow_survives_a_transient_502(monkeypatch, capsys) -> None:
    """`runs status --follow` polls the same endpoint and froze the same way."""

    class _FlakyStatusClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def get_run(self, run_id: str) -> dict:
            self.calls += 1
            if self.calls == 1:
                raise deploy_commands.ApiError(502, "HTTP Error 502: Bad Gateway")
            return {"run_id": run_id, "state": "done"}

    monkeypatch.setattr(run_commands.time, "sleep", lambda _seconds: None)

    assert run_commands._follow_status(_FlakyStatusClient(), "flash-st", interval=0) == 0
    assert "done" in capsys.readouterr().out


def test_follow_reports_a_finished_run_even_if_the_final_status_fetch_blips(
    monkeypatch, capsys
) -> None:
    """The poll already returned the terminal state; a blip on the last render is not a failure."""

    class _FinalBlipClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.run_calls = 0

        def get_logs(self, run_id: str, offset: int = 0) -> dict:
            return {"run_id": run_id, "logs": "", "offset": 0, "state": "done"}

        def get_run(self, run_id: str) -> dict:
            self.run_calls += 1
            if self.run_calls == 1:
                return {"run_id": run_id, "state": "done"}
            raise deploy_commands.ApiError(502, "HTTP Error 502: Bad Gateway")

    monkeypatch.setattr(run_commands.time, "sleep", lambda _seconds: None)

    assert run_commands._follow_run(_FinalBlipClient(), "flash-final") == 0

    err = capsys.readouterr().err
    assert "could not fetch the final status" in err
    assert "runs status flash-final" in err


def test_follow_warns_once_per_outage_not_once_per_attempt(monkeypatch, capsys) -> None:
    """A five-minute outage must not bury the logs already printed under identical warnings."""

    class _RepeatFlakyClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.log_calls = 0

        def get_logs(self, run_id: str, offset: int = 0) -> dict:
            self.log_calls += 1
            if self.log_calls <= 3:
                raise deploy_commands.ApiError(502, "HTTP Error 502: Bad Gateway")
            return {"run_id": run_id, "logs": "", "offset": 0, "state": "done"}

        def get_run(self, run_id: str) -> dict:
            return {"run_id": run_id, "state": "done"}

    monkeypatch.setattr(run_commands.time, "sleep", lambda _seconds: None)

    assert run_commands._poll_logs(_RepeatFlakyClient(), "flash-noisy", interval=0).state == "done"
    assert capsys.readouterr().err.count("retrying") == 1


def test_follow_surfaces_a_wrong_api_url_instead_of_retrying_it(monkeypatch, capsys) -> None:
    """A proxy answering 200 with a non-JSON body is permanent, not a blip.

    `ClientError` is the base class, so classifying every one of them as transient would retry a
    wrong `--api-url` for five minutes and bury the hint that says how to fix it -- then claim the
    run "may still be going and still billing". That is the misleading message this fix exists to
    remove, so it must not reappear here.
    """
    from flash.client.http import _unexpected_response

    class _ProxyClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.log_calls = 0

        def get_logs(self, run_id: str, offset: int = 0) -> dict:
            self.log_calls += 1
            raise _unexpected_response(
                "https://proxy.example", f"/v1/runs/{run_id}/logs", "returned a non-JSON body"
            )

        def get_run(self, run_id: str) -> dict:
            return {"run_id": run_id, "state": "running"}

    monkeypatch.setattr(run_commands.time, "sleep", lambda _seconds: None)
    client = _ProxyClient()

    with pytest.raises(deploy_commands.ClientError) as excinfo:
        run_commands._poll_logs(client, "flash-proxy", interval=0)

    assert "Check that --api-url points at your Flash control plane" in str(excinfo.value)
    assert client.log_calls == 1
    assert "retrying" not in capsys.readouterr().err


def test_follow_retries_a_genuinely_unreachable_service(monkeypatch, capsys) -> None:
    """Nobody answering IS transient: a plane restart mid-run must not end the follow."""
    from flash.client.http import ServiceUnreachableError

    class _UnreachableClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.log_calls = 0

        def get_logs(self, run_id: str, offset: int = 0) -> dict:
            self.log_calls += 1
            if self.log_calls == 1:
                raise ServiceUnreachableError(
                    "cannot reach the Flash service at https://plane.example (Connection refused)"
                )
            return {"run_id": run_id, "logs": "trained\n", "offset": 8, "state": "done"}

        def get_run(self, run_id: str) -> dict:
            return {"run_id": run_id, "state": "done"}

    monkeypatch.setattr(run_commands.time, "sleep", lambda _seconds: None)

    assert run_commands._poll_logs(_UnreachableClient(), "flash-down", interval=0).state == "done"
    assert "the service was unreachable" in capsys.readouterr().err
