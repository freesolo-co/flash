# probe: does the bare-integer exemption at reviewed head fcfade1ee leak numeric credentials?
import importlib.util
import os
import re
import sys
import urllib.parse

sys.path.insert(0, "/home/azureuser/benchmark/flash-opd-diag")

# --- reviewed-head (fcfade1ee) redactor, transcribed verbatim from .rev_bridge_fcfade.py ---
_MIN_SECRET_COMPONENT = 8
_SECRET_DETAIL_OLD = re.compile(
    r"(?i)(authorization|api[-_ ]?key|access[-_ ]?token|token|secret|password)"
    r"(\s*[:=]\s*)(?:bearer\s+)?([^\s,;]+)"
)


def _secret_env_name(name):
    upper = str(name).upper()
    return upper in {"AUTHORIZATION", "HF_TOKEN"} or upper.endswith(
        ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
    )


def _secret_needles(secret):
    forms = {secret}
    if "\n" in secret:
        forms.update(
            line for raw in secret.splitlines() if len(line := raw.strip()) >= _MIN_SECRET_COMPONENT
        )
    needles = set()
    for form in forms:
        encoded = urllib.parse.quote(form, safe="")
        needles.update(
            {form, encoded, re.sub(r"%[0-9A-Fa-f]{2}", lambda m: m.group(0).lower(), encoded)}
        )
    return needles


def old_detail(message):
    secrets = {value for name, value in os.environ.items() if value and _secret_env_name(name)}
    for secret in sorted(secrets, key=len, reverse=True):
        for needle in sorted(_secret_needles(secret), key=len, reverse=True):
            if len(needle) >= 8:
                message = message.replace(needle, "<redacted>")
                continue
            left = r"(?<!\w)" if needle[:1].isalnum() or needle[:1] == "_" else ""
            right = r"(?!\w)" if needle[-1:].isalnum() or needle[-1:] == "_" else ""
            message = re.sub(f"{left}{re.escape(needle)}{right}", "<redacted>", message)
    message = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", message)
    return _SECRET_DETAIL_OLD.sub(
        lambda match: (
            match.group(0)
            if match.group(3).isdigit()
            else f"{match.group(1)}{match.group(2)}<redacted>"
        ),
        message,
    )


# --- reference redactor shipped in-tree ---
spec = importlib.util.spec_from_file_location(
    "bs", "/home/azureuser/benchmark/flash-opd-diag/flash/providers/_lifecycle/bootstrap_secrets.py"
)
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)

# --- current HEAD (66eebe9ff) redactor, imported live ---
spec2 = importlib.util.spec_from_file_location(
    "br", "/home/azureuser/benchmark/flash-opd-diag/flash/engine/worker/train/opd/child/bridge.py"
)
br = importlib.util.module_from_spec(spec2)
try:
    spec2.loader.exec_module(br)
    new_detail = lambda m: br._safe_child_failure_detail(RuntimeError(m))
except Exception as exc:  # noqa: BLE001
    print(f"!! could not import current bridge.py: {type(exc).__name__}: {exc}")
    new_detail = None

CASES = [
    "upstream rejected credentials: password: 12345678",
    "teacher auth failed, Authorization: 90218374651029384756",
    "provider error api_key: 8675309867530986753098",
    "broker refused capability, access_token: 4483920175",
    "bad eos, token: 151643",  # the intended save
]

# empty environment: a runtime-minted credential is in no env var, so the value pass is inert.
for key in [k for k in os.environ if _secret_env_name(k)]:
    del os.environ[key]

print("=== reviewed head fcfade1ee (candidate's target) ===")
for case in CASES:
    out = old_detail(case)
    ref = bs._safe_detail(case)
    leaked = out == case and "<redacted>" in ref
    print(f"  in   : {case}")
    print(f"  head : {out}")
    print(f"  ref  : {ref}")
    print(f"  LEAK : {leaked}\n")

if new_detail is not None:
    print("=== current working HEAD 66eebe9ff ===")
    for case in CASES:
        out = new_detail(case)
        print(f"  in   : {case}")
        print(f"  head : {out}")
        print(f"  LEAK : {out == case and '<redacted>' in bs._safe_detail(case)}\n")
