"""The hosted Modal app's on-disk paths must survive the move into flash.

`flash/serving/` used to be the top-level `serving/` of another repo, so every path it derives by
walking up from its own file shifted by one level, and its own `pyproject.toml` did not come with it.
Both breakages are invisible to the offline suite -- the app imports fine and every route test still
passes -- and only surface at `modal deploy`, which is the worst place to find them. These assert on
the source text because importing `modal_app` requires live Modal configuration.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODAL_APP = ROOT / "flash" / "serving" / "app" / "modal_app.py"
SERVING_README = ROOT / "flash" / "serving" / "app" / "README.md"
DEPLOY_WORKFLOWS = ("deploy-modal.yml", "deploy-modal-dev.yml")


def _module() -> ast.Module:
    return ast.parse(MODAL_APP.read_text(encoding="utf-8"))


def _assigned_value(name: str) -> ast.expr:
    for node in ast.walk(_module()):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node.value
    raise AssertionError(f"{name} is not assigned in modal_app.py")


def test_repo_dir_walks_up_to_the_actual_repo_root() -> None:
    # SERVING_DIR is flash/serving/app/, so the repo root is three parents up. too few resolves
    # inside the flash package, where load_dotenv finds no .env and silently returns -- a production
    # deploy then runs with none of its secrets and fails at request time, not at deploy time.
    assert MODAL_APP.parent.parent.parent.parent == ROOT
    assert (ROOT / "pyproject.toml").is_file()

    # derived from the file's real depth rather than hardcoded, so a later move of the app updates
    # the expectation with it instead of leaving this passing against a stale literal.
    depth = len(MODAL_APP.parent.relative_to(ROOT).parts)
    source = ast.unparse(_assigned_value("REPO_DIR"))
    assert source == "SERVING_DIR" + ".parent" * depth, source


def test_image_installs_from_a_pyproject_that_exists() -> None:
    # the app's own pyproject.toml did not survive the move; its dependency bounds live in flash's
    # `serve-runtime` and `serving` extras now. pip_install_from_pyproject on a missing file fails
    # the image build outright.
    call = next(
        node
        for node in ast.walk(_module())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pip_install_from_pyproject"
    )

    target = ast.unparse(call.args[0])
    assert "REPO_DIR" in target, target
    assert "pyproject.toml" in target, target
    assert "SERVING_DIR" not in target, target

    extras = next(kw for kw in call.keywords if kw.arg == "optional_dependencies")
    assert ast.literal_eval(extras.value) == ["serve-runtime", "serving"]

    # and those extras must genuinely exist, or the build resolves an empty dependency set.
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    for extra in ("serve-runtime", "serving"):
        assert declared[extra], extra


def test_modal_image_applies_and_verifies_the_shared_vllm_repair() -> None:
    source = MODAL_APP.read_text(encoding="utf-8")
    copy_call = next(
        node
        for node in ast.walk(_module())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_local_file"
        and "patch_vllm_moe_lora.py" in ast.unparse(node.args[0])
    )
    remote_keyword = next(keyword for keyword in copy_call.keywords if keyword.arg == "remote_path")
    assert ast.literal_eval(remote_keyword.value) == "/root/patch_vllm_moe_lora.py"
    copy_keyword = next(keyword for keyword in copy_call.keywords if keyword.arg == "copy")
    assert ast.literal_eval(copy_keyword.value) is True

    install_index = source.index(".pip_install_from_pyproject(")
    copy_index = source.index(".add_local_file(", install_index)
    run_index = source.index(".run_commands(", copy_index)
    env_index = source.index(".env(", run_index)
    mount_index = source.index('.add_local_python_source("flash")', env_index)
    assert install_index < copy_index < run_index < env_index < mount_index
    command = source[run_index:env_index]
    apply = "python /root/patch_vllm_moe_lora.py &&"
    verify = "python /root/patch_vllm_moe_lora.py --verify &&"
    cleanup = "rm /root/patch_vllm_moe_lora.py"
    assert command.index(apply) < command.index(verify) < command.index(cleanup)


def test_hosted_deploy_docs_and_workflows_point_at_paths_that_exist() -> None:
    """The documented deploy commands and the workflows that run them must match the move.

    The README's blocks were written when the app was a top-level `serving/` directory, so they said
    `cd serving` and `modal deploy modal_app.py`. Both are now wrong in a way that fails only when an
    operator follows them during an incident: there is no repo-root `serving/`, and deploying from
    the package directory raises `ModuleNotFoundError: No module named 'flash'` because modal's CLI
    puts the *working directory* on sys.path.
    """
    readme = SERVING_README.read_text(encoding="utf-8")
    assert "cd serving" not in readme
    # both documented deploys must name the app by its repo-root-relative path.
    assert readme.count("modal deploy flash/serving/app/modal_app.py") == 1
    assert readme.count("modal deploy --env dev flash/serving/app/modal_app.py") == 1
    assert readme.count('export FREESOLO_DEPLOYMENT_ID="manual-production-') == 1
    assert readme.count('export FREESOLO_DEPLOYMENT_ID="manual-development-') == 1

    for workflow in DEPLOY_WORKFLOWS:
        path = ROOT / ".github" / "workflows" / workflow
        # the README cites these by name; a citation that resolves to nothing stranded the hosted
        # fleet with no redeploy path, which is why they were ported alongside the app.
        assert path.is_file(), workflow
        assert workflow in readme, workflow
        body = path.read_text(encoding="utf-8")
        assert "modal deploy" in body
        # run from the repo root, for the sys.path reason above.
        assert "working-directory: ." in body


def test_image_extras_cover_every_directly_imported_third_party_package() -> None:
    """The image resolves from these extras, not uv.lock, so a dropped bound is a production break.

    The app used to carry its own `pyproject.toml` listing its full dependency set; splitting that
    across flash's `serve-runtime` + `serving` extras silently dropped `httpx`, which
    `src/persistence.py` and `src/supabase_rest.py` import at module scope for the durable adapter
    registry and usage billing. Nothing offline catches that: the test suite installs the dev extra,
    which brings httpx in anyway.
    """
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    named = set()
    for extra in ("serve-runtime", "serving"):
        for spec in declared[extra]:
            named.add(re.split(r"[<>=!\[;]", spec, maxsplit=1)[0].strip().replace("_", "-"))

    imported: set[str] = set()
    for path in (ROOT / "flash" / "serving").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                root = module.split(".")[0]
                if root != "flash" and root not in sys.stdlib_module_names:
                    imported.add(root)

    # import name -> distribution name, where they differ.
    distribution = {
        "PIL": "pillow",
        "huggingface_hub": "huggingface-hub",
        "dotenv": "python-dotenv",
    }
    # guaranteed by a hard dependency of something already named above, so they need no own bound:
    # anyio is a required starlette dep (via fastapi); torch/PIL/pydantic ship with vllm.
    transitive = {"anyio", "torch", "PIL", "pydantic"}

    missing = {
        module
        for module in imported - transitive
        if distribution.get(module, module).replace("_", "-") not in named
    }
    assert not missing, f"imported by flash/serving but absent from the image extras: {missing}"


def test_image_ships_the_package_under_its_real_import_path() -> None:
    """The container must be able to `import flash.serving.src.X`.

    Before the move these modules imported each other as `src.X`, so shipping the bare directory
    to `/root/src` was enough. They import each other as `flash.serving.src.X` now, and a `/root/src`
    tree cannot satisfy that: the container raises `ModuleNotFoundError: No module named
    'flash.serving'` on the first engine call -- *after* `modal deploy` has already reported success,
    which is why no offline test or deploy-time import catches it.
    """
    tree = _module()
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    # a bare directory mount cannot reconstruct the `flash.serving.src` package path.
    assert "add_local_dir" not in calls
    assert "add_local_python_source" in calls

    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_local_python_source"
    )
    assert [ast.literal_eval(arg) for arg in call.args] == ["flash"]

    # every module the engine imports must actually live under that package, or the mount ships a
    # tree that still cannot satisfy the imports.
    src = ROOT / "flash" / "serving" / "src"
    assert (ROOT / "flash" / "__init__.py").is_file()
    assert (ROOT / "flash" / "serving" / "__init__.py").is_file()
    assert (src / "__init__.py").is_file()

    # and modal_app's own module-scope flash imports must resolve inside that same package.
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("flash."):
            relative = Path(node.module.replace(".", "/"))
            assert (ROOT / f"{relative}.py").is_file() or (
                ROOT / relative / "__init__.py"
            ).is_file(), node.module


def test_serving_workflow_gates_include_the_shared_repair_script() -> None:
    path = "docker/patch_vllm_moe_lora\\.py$"
    for workflow in (*DEPLOY_WORKFLOWS, "publish-serving-image.yml"):
        source = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        gate_lines = [line for line in source.splitlines() if "grep -qE" in line]
        assert len(gate_lines) == 1, workflow
        assert path in gate_lines[0], workflow


def test_self_hosting_docs_name_the_image_the_workflow_actually_publishes() -> None:
    """The documented `--image` must be the repository the publish workflow pushes to.

    These are two independent strings with no shared constant: the docs said
    `ghcr.io/freesolo-co/flash-serve` while the workflow publishes
    `ghcr.io/freesolo-co/freesolo-flash-serve`. A wrong image name is not caught by the deploy
    dry-run, which only validates that the digest is syntactically well formed -- it fails at
    provisioning, against the customer's own provider account.
    """
    workflow = (ROOT / ".github" / "workflows" / "publish-serving-image.yml").read_text(
        encoding="utf-8"
    )
    published = re.search(r"IMAGE:\s*(\S+)", workflow)
    assert published is not None, "the publish workflow must state the image it pushes"
    image = published.group(1)

    docs = (ROOT / "SELF_HOSTING.md").read_text(encoding="utf-8")
    cited = re.findall(r"ghcr\.io/[\w./-]+(?=@sha256:)", docs)
    assert cited, "SELF_HOSTING.md must show a pinned --image to deploy"
    assert set(cited) == {image}, (cited, image)
