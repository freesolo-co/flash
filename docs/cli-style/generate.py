"""Generate a before/after gallery of every `flash` CLI command's output.

Drives the real cmd_* handlers with a fake control-plane client, capturing the plain
(FLASH_STYLE=0) and themed (FLASH_STYLE=1) rendering of each, then writes a self-contained
dark-terminal HTML page plus a full-page PNG. ANSI is converted to inline-styled spans
(xterm-256 -> hex), so the page needs no external assets and renders anywhere.

usage:
    uv run python docs/cli-style/generate.py            # writes docs/cli-style/index.html
    uv run python docs/cli-style/generate.py out.html
"""

from __future__ import annotations

import contextlib
import html as _html
import io
import os
import re
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from flash.cli import main as cli
from flash.cli.main import render


class _Utf8(io.StringIO):
    encoding = "utf-8"

    def isatty(self) -> bool:
        return False


class FakeClient:
    """A stand-in control plane with realistic data for every read/manage command."""

    def me(self):
        return {
            "kind": "freesolo_api_key",
            "key_prefix": "fs_live_8Kd2",
            "email": "ml-team@acme.ai",
            "org_id": "org_acme",
            "user_id": "usr_4192",
        }

    def list_runs(self):
        return [
            _run("flash-1718903000-e5f6a7b8", "running", 0.4210, "Qwen/Qwen3.5-9B", "sft", "vast", "L40S", 1718903000.0),
            _run("flash-1718900000-a1b2c3d4", "done", 1.8421, "Qwen/Qwen3.5-4B", "grpo", "runpod", "RTX 5090", 1718900000.0),
            _run("flash-1718800000-99887766", "failed", 0.0712, "Qwen/Qwen3.5-2B", "grpo", None, None, 1718800000.0),
            _run("flash-1718700000-12ab34cd", "queued", 0.0, "openbmb/MiniCPM5-1B", "sft", None, None, 1718700000.0),
        ]

    def get_run(self, run_id):
        return {
            "run_id": run_id,
            "state": "done",
            "spec": {"model": "Qwen/Qwen3.5-4B", "algorithm": "grpo", "gpu": {"type": "RTX 5090"}},
            "remote": {"provider": "runpod", "gpu": "RTX 5090"},
            "cost_usd": 1.8421,
            "realized_cost_usd": 1.7740,
            "created_at": 1718896400.0,
            "updated_at": 1718900000.0,
            "error": None,
            "artifacts_dir": "acme/qwen3.5-4b-grpo-runs",
        }

    def cancel_run(self, run_id):
        return {"run_id": run_id, "state": "cancelled"}

    def deploy(self, run_id, **_):
        return {
            "run_id": run_id,
            "state": "deployed",
            "mode": "dev",
            "openai_model": run_id,
            "endpoint_name": "https://api.freesolo.co/v1",
            "gpu": "RTX 4090",
        }

    def undeploy(self, run_id):
        return {"run_id": run_id, "deleted_endpoints": ["flash-a1b2c3d4-serve"]}

    def deployments(self):
        return [
            {
                "run_id": "flash-1718900000-a1b2c3d4",
                "deployment": {"gpu": "RTX 4090", "endpoint_name": "https://api.freesolo.co/v1"},
            }
        ]

    def chat(self, run_id, messages, **_):
        return {
            "choices": [
                {
                    "message": {
                        "content": "The capital of France is Paris. It has been the country's "
                        "political and cultural center since the 12th century."
                    }
                }
            ]
        }


def _run(run_id, state, cost, model, algo, provider, gpu, ts):
    remote = {"provider": provider, "gpu": gpu} if provider else {}
    return {
        "run_id": run_id,
        "state": state,
        "cost_usd": cost,
        "updated_at": ts,
        "spec": {"model": model, "algorithm": algo},
        "remote": remote,
    }


FAKE = FakeClient()


def _set_style(styled: bool) -> None:
    os.environ["FLASH_STYLE"] = "1" if styled else "0"
    os.environ.pop("NO_COLOR", None)
    os.environ["COLUMNS"] = "76"


def _capture_argv(argv, *, styled, cwd=None) -> str:
    """Run a real `flash` invocation against the fake client and capture stdout."""
    _set_style(styled)
    commands = sys.modules["flash.cli.main.commands"]
    saved = commands.client_from_config
    commands.client_from_config = lambda *a, **k: FAKE
    out, err = _Utf8(), _Utf8()
    prev = os.getcwd()
    try:
        if cwd:
            os.chdir(cwd)
        with redirect_stdout(out), redirect_stderr(err), contextlib.suppress(SystemExit):
            cli.main(argv)
    finally:
        os.chdir(prev)
        commands.client_from_config = saved
    return out.getvalue().rstrip("\n")


def _capture_text(styled_text: str, plain_text: str, *, styled: bool) -> str:
    """For flows that need extra stubbing, supply the styled/plain strings directly."""
    _set_style(styled)
    return (styled_text if styled else plain_text).rstrip("\n")


# ---- xterm-256 -> hex --------------------------------------------------------------
def _xterm_hex(n: int) -> str:
    if n < 16:
        base = [
            "000000", "800000", "008000", "808000", "000080", "800080", "008080", "c0c0c0",
            "808080", "ff0000", "00ff00", "ffff00", "0000ff", "ff00ff", "00ffff", "ffffff",
        ]
        return base[n]
    if n >= 232:
        v = 8 + (n - 232) * 10
        return f"{v:02x}{v:02x}{v:02x}"
    n -= 16
    r, g, b = n // 36, (n % 36) // 6, n % 6
    steps = [0, 95, 135, 175, 215, 255]
    return f"{steps[r]:02x}{steps[g]:02x}{steps[b]:02x}"


_ANSI = re.compile(r"\x1b\[([0-9;]*)m")


def ansi_to_html(text: str) -> str:
    out, idx, open_span = [], 0, False
    bold = dim = False
    fg = None

    def style() -> str:
        parts = []
        if fg:
            parts.append(f"color:#{fg}")
        if bold:
            parts.append("font-weight:700")
        if dim:
            parts.append("opacity:.62")
        return ";".join(parts)

    for m in _ANSI.finditer(text):
        out.append(_html.escape(text[idx : m.start()]))
        idx = m.end()
        codes = [c for c in m.group(1).split(";") if c != ""] or ["0"]
        i = 0
        while i < len(codes):
            c = codes[i]
            if c == "0":
                bold = dim = False
                fg = None
            elif c == "1":
                bold = True
            elif c == "2":
                dim = True
            elif c == "38" and i + 2 < len(codes) and codes[i + 1] == "5":
                fg = _xterm_hex(int(codes[i + 2]))
                i += 2
            i += 1
        if open_span:
            out.append("</span>")
            open_span = False
        s = style()
        if s:
            out.append(f'<span style="{s}">')
            open_span = True
    out.append(_html.escape(text[idx:]))
    if open_span:
        out.append("</span>")
    return "".join(out)


def build_cases(tmp: Path):
    setup_dir = tmp / "scaffold"
    setup_dir.mkdir()
    list_dir = tmp / "listing"
    (list_dir / "environments" / "math-grader").mkdir(parents=True)
    (list_dir / "environments" / "math-grader" / "environment.py").write_text("# env\n")
    (list_dir / "environment.py").write_text("# env\n")

    cost_cfg = tmp / "rl.toml"
    cost_cfg.write_text(
        'model = "Qwen/Qwen3.5-4B"\nalgorithm = "grpo"\n'
        '[environment]\nid = "acme/math-grader"\n'
        '[train]\nsteps = 150\nseeds = [0]\nhf_repo = "acme/runs"\n'
        '[gpu]\ntype = "RTX 5090"\n'
    )
    dry_cfg = tmp / "sft.toml"
    dry_cfg.write_text(
        'model = "Qwen/Qwen3.5-4B"\nalgorithm = "sft"\n'
        '[environment]\nid = "acme/math-grader"\n'
        '[train]\nepochs = 1\nseeds = [0]\nhf_repo = "acme/runs"\n'
    )

    install_plain = (
        "installed acme/math-grader; recorded in /Users/you/.flash/environments.json\n"
        'use it via:  [environment]\\nid = "acme/math-grader"'
    )
    push_plain = (
        "published acme/math-grader\nreference it in your config:\n\n"
        '  [environment]\n  id = "acme/math-grader"'
    )

    cases = []

    def add_argv(title, sub, argv, cwd=None):
        cases.append((title, sub, _capture_argv(argv, styled=False, cwd=cwd),
                      _capture_argv(argv, styled=True, cwd=cwd)))

    def add_text(title, sub, styled_fn, plain_text):
        _set_style(False)
        plain = _capture_text("", plain_text, styled=False)
        _set_style(True)
        styled = _capture_text(styled_fn(), "", styled=True)
        cases.append((title, sub, plain, styled))

    add_argv("flash version", "package version", ["version"])
    add_text("flash login", "verify + store your freesolo key",
             lambda: render.login_ok(FAKE.me()), render.login_ok(FAKE.me()))
    add_argv("flash whoami", "identity behind the stored key", ["whoami"])
    add_argv("flash models", "supported base models", ["models"])
    add_argv("flash gpus", "managed GPU classes + $/hr", ["gpus"])
    add_argv("flash env setup", "scaffold a starter environment", ["env", "setup"], cwd=setup_dir)
    add_argv("flash env list", "installed + local environments", ["env", "list"], cwd=list_dir)
    add_text("flash env install acme/math-grader", "record a published environment",
             lambda: render.env_installed("acme/math-grader", "/Users/you/.flash/environments.json"),
             install_plain)
    add_text("flash env push --name math-grader .", "package + upload a local environment",
             lambda: render.env_published("acme/math-grader"), push_plain)
    add_argv("flash train --cost rl.toml", "pre-flight cost estimate", ["train", str(cost_cfg), "--cost"])
    add_argv("flash train --dry-run sft.toml", "validate a config locally", ["train", str(dry_cfg), "--dry-run"])
    add_argv("flash status <run>", "a run's full status", ["status", "flash-1718900000-a1b2c3d4"])
    add_argv("flash runs", "all runs and their state/cost", ["runs"])
    add_argv("flash cancel <run>", "cancel a run", ["cancel", "flash-1718903000-e5f6a7b8"])
    add_argv("flash deploy <run>", "serve a trained adapter", ["deploy", "flash-1718900000-a1b2c3d4"])
    add_argv("flash undeploy <run>", "tear down a serving endpoint", ["undeploy", "flash-1718900000-a1b2c3d4"])
    add_argv("flash deployments", "active serving deployments", ["deployments"])
    add_argv("flash chat <run> -m ...", "chat with a deployed adapter",
             ["chat", "flash-1718900000-a1b2c3d4", "-m", "What is the capital of France?"])
    return cases


HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flash CLI - output style</title>
<style>
:root{--bg:#0b0e14;--ink:#c8d3e0;--mut:#7a8699;--line:#1c2330;--accent:#00afff;--accent2:#5fd7d7;--green:#5fd787}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 600px at 50% -10%,#10213a33,transparent),var(--bg);
  color:var(--ink);font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Inter,sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:56px 24px 80px}
.brand{display:inline-flex;align-items:center;gap:10px;font-weight:700}
.brand .dot{width:9px;height:9px;border-radius:50%;background:var(--accent);box-shadow:0 0 14px var(--accent)}
.brand b{color:var(--accent)}
h1{font-size:30px;margin:18px 0 8px;letter-spacing:-.4px}
.lede{color:var(--mut);max-width:760px}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:22px;color:var(--mut);font-size:13px}
.legend span{display:inline-flex;align-items:center;gap:7px}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
.card{border:1px solid var(--line);border-radius:14px;margin:22px 0;overflow:hidden;
  box-shadow:0 1px 0 #ffffff0a inset,0 18px 40px -28px #000}
.card>.head{display:flex;align-items:baseline;gap:12px;padding:15px 18px;border-bottom:1px solid var(--line)}
.card>.head code{font:600 14px/1 ui-monospace,SFMono-Regular,Menlo,monospace}
.card>.head .sub{color:var(--mut);font-size:13px}
.cols{display:grid;grid-template-columns:1fr 1fr}
.col+.col{border-left:1px solid var(--line)}
.col .tag{display:flex;align-items:center;gap:8px;padding:9px 16px;color:var(--mut);font-size:11px;
  letter-spacing:.14em;text-transform:uppercase;border-bottom:1px solid var(--line);background:#0c0f17}
.col .tag .pill{width:7px;height:7px;border-radius:50%}
.before .pill{background:#3b4252}
.after .pill{background:var(--green);box-shadow:0 0 10px #5fd78788}
pre{margin:0;padding:18px;overflow-x:auto;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,"Cascadia Code",monospace;
  color:#aeb9c7;background:#0c0f17}
.after pre{background:#0a0d14}
.before pre{color:#8b93a3}
@media(max-width:760px){.cols{grid-template-columns:1fr}.col+.col{border-left:0;border-top:1px solid var(--line)}}
footer{color:var(--mut);font-size:13px;margin-top:48px;border-top:1px solid var(--line);padding-top:20px}
footer code{color:var(--accent2)}
</style></head><body><div class="wrap">
<header>
  <div class="brand"><span class="dot"></span><span><b>flash</b> cli</span></div>
  <h1>A standardized output theme for every command</h1>
  <p class="lede">One visual language across the whole CLI: a brand header, colored status badges,
  aligned tables, key/value panels, and syntax-highlighted JSON. The themed view renders on an
  interactive terminal; piped or scripted output stays byte-for-byte plain, so <code>jq</code>,
  scripts, and the agent contract are untouched.</p>
  <div class="legend">
    <span><i class="sw" style="background:#00afff"></i>accent / links</span>
    <span><i class="sw" style="background:#5fd787"></i>success / done</span>
    <span><i class="sw" style="background:#ffaf00"></i>cost / numbers</span>
    <span><i class="sw" style="background:#af87ff"></i>literals</span>
    <span><i class="sw" style="background:#ff5f5f"></i>failure</span>
    <span><i class="sw" style="background:#585858"></i>structure</span>
  </div>
</header>
"""

FOOT = """
<footer>Rendered from the real <code>flash</code> handlers via <code>FLASH_STYLE=1</code>. The
plain column is the exact machine output (<code>FLASH_STYLE=0</code> / non-TTY). Pure standard
library, no new dependencies. Regenerate with <code>uv run python docs/cli-style/generate.py</code>.</footer>
</div></body></html>
"""


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "index.html"
    with tempfile.TemporaryDirectory() as td:
        cases = build_cases(Path(td))
    blocks = [HEAD]
    for title, sub, plain, styled in cases:
        blocks.append(
            f'<section class="card"><div class="head"><code>{_html.escape(title)}</code>'
            f'<span class="sub">{_html.escape(sub)}</span></div><div class="cols">'
            f'<div class="col before"><div class="tag"><span class="pill"></span>before</div>'
            f"<pre>{ansi_to_html(plain) or '&nbsp;'}</pre></div>"
            f'<div class="col after"><div class="tag"><span class="pill"></span>after</div>'
            f"<pre>{ansi_to_html(styled) or '&nbsp;'}</pre></div></div></section>"
        )
    blocks.append(FOOT)
    out_path.write_text("".join(blocks))
    print(f"wrote {out_path} ({len(cases)} commands)")


if __name__ == "__main__":
    main()
