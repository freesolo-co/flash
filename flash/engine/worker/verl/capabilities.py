"""Ask the verl child what it can do, and turn the answers into launch decisions.

One subprocess pays the torch/verl import once and answers every independent capability question
at the same time; everything below either runs that probe or reads its result to decide how the
child gets launched -- which interpreter, which attention backend, whether the rollout can sleep,
whether GDN boundary resets are available.

Split out of `flash.engine.worker.train.entry.backend_common` to keep that module under the file-size limit.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
from collections.abc import Sequence

from flash.engine.support.verl_policy import FsdpGeneration
from flash.engine.worker.train.entry.backend_common import (
    CAUSAL_CONV1D_REQUIREMENT,
    FLA_REQUIREMENT,
    FLASH_QLA_REQUIREMENT,
    TRANSFORMERS_INSTALL_REQUIREMENT,
    VERL_REQUIREMENT_NAME,
    VERL_REQUIREMENT_URL,
    VERL_VENV_PYTHON,
    VERL_VENV_STAMP,
    _install_causal_conv1d,
    _install_flash_attn,
)

# ---------------------------------------------------------------------------
# batched child-interpreter capability probe
# answer independent questions in one child to pay one torch/verl import. child probes may import only
# packages installed in that venv; parent-only hub reads are interpolated. each answer fails closed
# independently and flushes before the live CUDA GDN smoke, which must run last.
# ---------------------------------------------------------------------------
_CAPABILITY_PROBE = """
import json


def emit(key, value):
    print("FLASH_VERL_CAPS=" + json.dumps({key: value}), flush=True)


try:
    from verl.workers.config.rollout import RolloutConfig

    emit("rollout_fields", sorted(RolloutConfig.__dataclass_fields__))
except Exception:
    emit("rollout_fields", None)

try:
    import torch

    emit(
        "capability",
        list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    )
except Exception:
    emit("capability", None)

try:
    import flashinfer  # noqa: F401

    emit("flashinfer", True)
except Exception:
    emit("flashinfer", False)

try:
    import wandb  # noqa: F401

    emit("wandb", True)
except Exception:
    emit("wandb", False)

# gdn is asked only when the parent resolved a model_type to ask about (a gdn hybrid). the module
# name is a parent-resolved literal: it comes from the checkpoint config, which needs a hub/cache
# read the child must not repeat.
if %(gdn_module)r:
    try:
        import importlib
        import inspect

        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )

        # the reason the LAST failing check produced, so a clean negative can name it. the two
        # package flags cannot express the source check: it fails with both installed, and an
        # operator reading "fla=True causal_conv1d=True" alongside "install fla + causal_conv1d"
        # is being sent to fix something that is not broken.
        why = ""
        ok = is_flash_linear_attention_available() and is_causal_conv1d_available()
        if not ok:
            why = (
                "fla="
                + str(is_flash_linear_attention_available())
                + " causal_conv1d="
                + str(is_causal_conv1d_available())
                + " (install the missing package into the verl interpreter)"
            )
        if ok:
            import causal_conv1d  # noqa: F401  -- fail a built-but-broken ABI here, not at model load

            mod = importlib.import_module(%(gdn_module)r)
            gdn = next(
                (
                    c
                    for n, c in vars(mod).items()
                    if isinstance(c, type) and n.endswith("GatedDeltaNet")
                ),
                None,
            )
            # read THROUGH decorators. transformers wraps this forward in
            # @force_accelerate_hooks("conv1d"), which returns a plain closure and sets no
            # __wrapped__, so both getsource() and inspect.unwrap() return the hook wrapper --
            # ~1.1k chars of accelerate plumbing that mentions neither name. That reads as "this
            # transformers build cannot reset gdn state", which is a FALSE NEGATIVE: every released
            # 5.x reaches the values as kwargs.get("cu_seq_lens_q") / kwargs.get("seq_idx") in the
            # real body, so no version bump can fix it and the raise sends operators to reinstall
            # packages that are already present. The undecorated function is reachable through the
            # wrapper's closure cells; walk them until the source answers the question.
            src = ""
            if gdn is not None:
                fn = gdn.forward
                seen = set()
                while True:
                    src = inspect.getsource(fn)
                    if ("cu_seq_lens_q" in src) and ("seq_idx" in src):
                        break
                    nxt = None
                    for cell in getattr(fn, "__closure__", None) or ():
                        try:
                            val = cell.cell_contents
                        except ValueError:
                            continue
                        if inspect.isfunction(val) and id(val) not in seen:
                            seen.add(id(val))
                            nxt = val
                            break
                    if nxt is None:
                        break
                    fn = nxt
            ok = ("cu_seq_lens_q" in src) and ("seq_idx" in src)
            if not ok:
                # both packages are present, so this is the transformers version, not the env.
                why = (
                    "fla and causal_conv1d are both installed, but "
                    + (
                        "no *GatedDeltaNet class in " + %(gdn_module)r
                        if gdn is None
                        else (
                            "its forward() takes neither cu_seq_lens_q nor seq_idx"
                            if ("cu_seq_lens_q" not in src) and ("seq_idx" not in src)
                            else "its forward() is missing "
                            + ("cu_seq_lens_q" if "cu_seq_lens_q" not in src else "seq_idx")
                        )
                    )
                    + " -- this transformers build cannot reset gdn state at packed boundaries; "
                    + "installing packages will not fix it"
                )
        if ok:
            import torch

            # a causal_conv1d compiled without this arch imports fine and raises at the first
            # forward, so smoke it now: the answer decides whether we pack, and being wrong means
            # training contaminated.
            if torch.cuda.is_available():
                from causal_conv1d import causal_conv1d_fn

                causal_conv1d_fn(
                    torch.zeros(1, 4, 8, device="cuda", dtype=torch.bfloat16),
                    torch.zeros(4, 3, device="cuda", dtype=torch.bfloat16),
                )
                torch.cuda.synchronize()
        if not ok:
            # say WHY on a clean negative too: which of the three checks failed is the difference
            # between "install fla" and "rebuild causal_conv1d for this arch".
            print(
                "[verl] gdn boundary resets unavailable in the child: " + why,
                flush=True,
            )
        emit("gdn_boundary_resets", bool(ok))
    except Exception as e:
        # a bare False here is what made this invisible. the caller raises on a hybrid that cannot
        # reset, and an operator reading only that has no way back to which check fell over --
        # a missing wheel, a broken ABI, or a conv kernel compiled without this arch all look
        # identical from the parent. printed to STDOUT, which probe_verl_capabilities forwards
        # on the '[verl] ' prefix -- stderr would be dropped, it is only reported when the
        # child answers nothing at all.
        import traceback

        print(
            "[verl] gdn boundary resets unavailable in the child: "
            + " | ".join(traceback.format_exc(limit=3).strip().splitlines()),
            flush=True,
        )
        emit("gdn_boundary_resets", False)
"""

# every answer's fail-closed default, used when the child dies, times out, or prints nothing
# readable. `None` means "could not answer" for the two callers that must distinguish that from a
# negative answer: rollout_fields (unknown verl -> do not claim a fork field is missing) and
# capability (unknown card -> leave the library default alone rather than guessing a workaround).
_CAPABILITIES_UNAVAILABLE: dict = {
    "rollout_fields": None,
    "capability": None,
    "flashinfer": False,
    "wandb": False,
    "gdn_boundary_resets": None,
}

# the batched timeout is the sum of the former 300 + 600 + 120 + 120 + 120 second budgets, so the
# final GDN smoke retains its cold-import allowance. reaching the sum indicates a wedged child.
_CAPABILITY_PROBE_TIMEOUT_S = 300 + 600 + 120 + 120 + 120


def probe_verl_capabilities(python_bin: str, gdn_module: str = "") -> dict:
    """ask the VERL CHILD every independent capability question in ONE subprocess.

    ``gdn_module`` is parent-resolved; ``""`` skips the live CUDA GDN smoke. every key has a fail-closed
    default, and incrementally flushed answers survive a later child timeout.
    """
    probe = _CAPABILITY_PROBE % {"gdn_module": gdn_module}
    caps = dict(_CAPABILITIES_UNAVAILABLE)
    timed_out = False
    try:
        done = subprocess.run(
            [python_bin, "-c", probe],
            capture_output=True,
            text=True,
            timeout=_CAPABILITY_PROBE_TIMEOUT_S,
        )
        stdout, stderr, returncode = done.stdout, done.stderr, done.returncode
    except subprocess.TimeoutExpired as e:
        # the killed child still flushed every answer it reached, and TimeoutExpired carries that
        # output. a slow gdn smoke must not retract the cheap answers that already landed -- an
        # empty rollout_fields aborts grpo, which is a worse outcome than training padded.
        timed_out = True
        stdout = (
            e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        )
        stderr = (
            e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        )
        returncode = None
        print(
            f"[verl] capability probe timed out after {_CAPABILITY_PROBE_TIMEOUT_S}s; "
            "keeping the answers it flushed, failing the rest closed"
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[verl] capability probe failed ({e}); treating every capability as unavailable")
        return caps
    answered_any = False
    for line in (stdout or "").splitlines():
        if not line.startswith("FLASH_VERL_CAPS="):
            continue
        with contextlib.suppress(Exception):
            answered = json.loads(line.split("=", 1)[1])
            if isinstance(answered, dict):
                # merge rather than replace: a child from a future/older flash that answers a key
                # this build does not know must not add it, and one it never reached must keep its
                # fail-closed default rather than vanishing from the dict.
                caps.update({k: v for k, v in answered.items() if k in caps})
                answered_any = True
    # the child's own diagnostics, forwarded BEFORE the early return. the gdn question prints why it
    # answered no, and it is answered LAST -- so every real run answers something earlier, and a
    # bare `return caps` here would discard the one line the gate's raise tells operators to read.
    # matched on the prefix rather than forwarded wholesale: the child's stdout also carries verl's
    # own import chatter, which is not worth a parent log line.
    for line in (stdout or "").splitlines():
        if line.startswith("[verl] "):
            print(line, flush=True)
    if answered_any:
        return caps
    if not timed_out:
        print(
            "[verl] capability probe returned no readable answer "
            f"(exit {returncode}); treating every capability as unavailable"
        )
    if stderr:
        print(f"[verl] capability probe stderr: {stderr.strip()[:800]}")
    return caps


def verl_declares_rollout_field(caps: dict, field: str) -> bool:
    """report whether the probed verl declares `field` on RolloutConfig.

    the fork adds rollout fields stock verl does not have. hydra composes an unknown key happily and
    only fails later in omega_conf_to_dataclass, so callers ask here before emitting a fork-only
    override rather than letting the run abort at dataclass conversion. an unanswerable probe reads
    as "not declared", which is what the subprocess version returned when the child could not be
    reached: the caller raises with the cause instead of composing a key that would abort in
    dataclass conversion on a paid gpu.
    """
    fields = caps.get("rollout_fields")
    return bool(fields) and field in fields


def gdn_probe_module(model_id: str, revision: str = "") -> str:
    """the ``transformers.models.*`` modeling module the gdn question must inspect.

    resolved by the PARENT and handed to the child as a literal, because it comes from the
    checkpoint config -- a hub/cache read the child should not repeat. pair every call with
    ``gdn_reset_arch_from_caps`` below so the shim is rendered for the SAME arch the child cleared.
    """
    from flash.engine.worker.model.packing import gdn_model_type

    model_type = gdn_model_type(model_id, revision=revision)
    return f"transformers.models.{model_type}.modeling_{model_type}"


def strict_gdn_probe_module(model_id: str, revision: str = "") -> str:
    """``gdn_probe_module`` that RAISES rather than guessing the arch, for a packed run.

    ``gdn_model_type`` answers ``qwen3_5`` both when the config says so and when the config could
    not be read at all. Those are the same string and a different meaning. For a packed run the
    difference is silent corruption: ``Qwen/Qwen3.6-35B-A3B`` is ``qwen3_5_moe``, so the fallback
    names the DENSE module -- the child clears it, the shim patches it, the MoE layers it was
    supposed to protect stay unpatched, and packed examples bleed state while the log reports
    ``gdn packed-boundary resets active``. A failed read must stop the run instead.
    """
    from transformers import AutoConfig

    from flash.engine.worker.io.hf import model_revision_kwargs

    try:
        cfg = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True, **model_revision_kwargs(revision)
        )
        model_type = getattr(cfg, "model_type", None)
    except Exception as e:
        raise RuntimeError(
            f"packed gdn run could not read the model config for {model_id!r}, so the modeling "
            "module to patch is unknown. refusing to guess: the fallback arch would patch a "
            "different module than the model uses and train across packed example boundaries "
            "while reporting resets as active."
        ) from e
    if not model_type:
        raise RuntimeError(
            f"packed gdn run read the config for {model_id!r} but it declares no model_type, so "
            "the modeling module to patch is unknown. refusing to guess."
        )
    return f"transformers.models.{model_type}.modeling_{model_type}"


def gdn_reset_arch_from_caps(caps: dict, gdn_module: str) -> str | None:
    """the architecture to patch when the VERL CHILD can honor packed GDN boundary resets, else None.

    transformers silently discards ``cu_seqlens`` without fla and ``seq_idx`` without causal_conv1d,
    contaminating packed boundaries. probe the child interpreter because it owns the model stack.

    return the verified ``model_type`` rather than a bool so the shim cannot patch a different module;
    ``qwen3_5`` and ``qwen3_5_moe`` are distinct. child probes must not import parent-only ``flash``.
    """
    if not caps.get("gdn_boundary_resets"):
        return None
    # "transformers.models.<arch>.modeling_<arch>" -> "<arch>", read from the string the CHILD was
    # handed rather than re-resolved here: a second resolve could disagree with what was cleared.
    return gdn_module.split(".")[2]


def require_gdn_boundary_resets(caps: dict, gdn_module: str) -> str | None:
    """The arch to patch for a gdn hybrid, raising if the child cannot honor boundary resets.

    On grpo/opd there is no safe fallback, so this raises rather than launching. verl's padded path
    (``use_remove_padding=False``) is boundary-correct but cannot complete a step: flash sets
    ``use_fused_kernels=True`` unconditionally and verl's fsdp engine has no guard for that pair, so
    the padded+fused branch returns a dense ``[bsz, response_len]`` where siblings re-nest via
    ``cu_seqlens``. That fails at EVERY batch size, measured dead at ``padding.py:144`` on four
    matched real-gpu arms, and the error names neither gdn nor this decision -- so letting it launch
    spends the whole rental on an untraceable shape error. Dropping ``use_fused_kernels`` is NOT the
    alternative: ``engine/vram.py`` sizes assuming no dense ``[b, s, vocab]`` logits tensor exists
    (~130 GB at 32k on a 248k vocab), trading this crash for an OOM. Give the child the kernels.

    sft reaches this too, and a packed sft run needs resets for the same reason: its control-plane
    gate is device-independent by construction (see ``gdn_packing_contract_available``), so it can
    prove the kernels are installed but never that
    the conv kernel runs on this card. an ``exact-unpacked`` sft run stays safe without resets --
    ``examples_per_update`` is 1, so there are no packed neighbours to contaminate -- which is why
    sft asks this only for a packed profile.
    """
    if not gdn_module:
        return None
    arch = gdn_reset_arch_from_caps(caps, gdn_module)
    if arch is None:
        raise RuntimeError(
            "gdn hybrid without child-side boundary resets: the padded fallback "
            "(use_remove_padding=False) is incompatible with the use_fused_kernels=True this "
            "recipe sets, and cannot complete a training step on verl's fsdp engine. see the "
            "'[verl] gdn boundary resets unavailable' line above for why the child could not "
            "honor resets -- installing fla + causal_conv1d in the verl interpreter is the fix, "
            "not disabling fused kernels."
        )
    return arch


def resolve_verl_python(workdir: str, *, install_wandb: bool = False) -> str:
    """return an interpreter that can import verl.

    prefer ``FLASH_VERL_PYTHON`` unchanged; otherwise provision an isolated venv with the pinned verl
    stack and optional wandb. rebuild owned venvs when their stamp differs. a missing or blank preset
    takes the provisioning path.
    """
    preset = os.environ.get("FLASH_VERL_PYTHON", "").strip()
    if preset:
        return preset
    venv = os.path.join(workdir, "verl-venv")
    py = os.path.join(venv, "bin", "python")
    stamp = os.path.join(venv, "flash-verl-requirement")
    installed = ""
    if os.path.exists(stamp):
        with open(stamp) as f:
            installed = f.read().strip()
    if installed != VERL_VENV_STAMP or not os.path.exists(py):
        # a retry reuses the pod workdir, so this venv can be from an earlier attempt, from an earlier
        # flash release pinning a different verl, or from an install that died partway. remove it and
        # start clean: reusing it would train on the wrong verl, and `uv venv` refuses to write into a
        # directory that already holds a pyvenv.cfg, so a half-built venv wedges every later retry.
        shutil.rmtree(venv, ignore_errors=True)
        # dev-only fallback (production uses FLASH_VERL_PYTHON on a prebuilt verl image): verl brings
        # its own torch/vllm, so use a full install rather than --no-deps to include runtime deps.
        subprocess.run(["uv", "venv", "--python", VERL_VENV_PYTHON, venv], check=True)
        # the SAME four overrides Dockerfile.worker writes, for the same reason: this pin
        # set deliberately violates declared ceilings, and a bare pin cannot break
        # any of them -- a pin is a constraint the resolver must satisfy alongside the
        # declaration, so the pair is simply unsatisfiable and the install fails outright.
        # only --override makes uv IGNORE the declaration. all four lines are required;
        # dropping any one leaves the set unsatisfiable or silently wrong: verl +
        # transferqueue declare numpy<2.0.0 -> vllm 0.19.1 needs numpy>=2 verl[vllm]
        # declares vllm>=0.8.5,<=0.12.0 -> flash needs 0.19.1 for the Qwen3.5 archs vllm
        # declares xgrammar>=0.1.32 -> structured opd gates on EXACTLY 0.1.25 verl and
        # vllm both depend on transformers with no upper bound -> an unpinned resolve
        # drifts the interpreter that TRAINS onto a transformers line nothing validated,
        # and transformers owns the gdn modelling code the boundary-reset shim patches.
        overrides = os.path.join(workdir, "verl-overrides.txt")
        with open(overrides, "w") as f:
            f.write(
                f"numpy==2.2.6\nxgrammar==0.1.25\nvllm==0.19.1\n"
                f"{TRANSFORMERS_INSTALL_REQUIREMENT}\n"
            )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                py,
                "--override",
                overrides,
                # the [vllm] extra, matching Dockerfile.worker's VERL_SPEC. verl's bare
                # install_requires omits vllm, and every entrypoint flash launches needs it:
                # main_ppo/main_ppo_sync set rollout.name=vllm.
                f"{VERL_REQUIREMENT_NAME}[vllm] @ {VERL_REQUIREMENT_URL}",
                "vllm==0.19.1",
                "numpy==2.2.6",
                TRANSFORMERS_INSTALL_REQUIREMENT,
                # not transitively guaranteed: verl imports these at module level on the launch path
                # (main_ppo -> ppo.ray_trainer -> rollout.llm_server imports cachetools, and
                # rollout.utils imports uvicorn + fastapi), yet declares none of them. vllm happens
                # to pull cachetools and fastapi today, but that is vllm's dependency choice, not a
                # contract verl states -- exact pins mirror the worker dockerfile's resolved venv.
                "cachetools==7.1.7",
                "uvicorn==0.52.4",
                "fastapi==0.141.1",
                # opd's entrypoint calls tq.init()/tq.close(); absent from verl's setup.py.
                "TransferQueue==0.1.7",
                # older raises AttributeError on PyArrow PyExtensionType.
                "datasets==5.0.1",
                "bitsandbytes==0.50.1",
                "qwen-vl-utils==0.0.14",
                "torchvision==0.25.0",
                "xgrammar==0.1.25",
                "tqdm==4.70.0",
                "pyarrow==25.0.1",
                # gated-deltanet kernels, in LOCKSTEP with Dockerfile.worker's
                # verl-venv layer (same fla sha, same tilelang pins). the model
                # trains in THIS interpreter, so fla being installed in the
                # worker's own env says nothing: without it here, transformers
                # binds chunk_gated_delta_rule to the pure-torch fallback that
                # takes **kwargs and discards cu_seqlens, and packed gdn runs
                # train across example boundaries while looking patched.
                # apache-tvm-ffi is pinned to 0.1.9 because flash-qla requires exactly that, and
                # 0.1.12 double-registers TVM-FFI and aborts `import tilelang`.
                FLA_REQUIREMENT,
                # the flashqla GDN backend fla 0.5.2 dispatches to. same lockstep requirement:
                # the shim binds it in THIS interpreter, so a wheel present only in the worker's
                # env would leave the child on the old kernel while the marker still printed.
                FLASH_QLA_REQUIREMENT,
                "tilelang==0.1.9",
                "apache-tvm-ffi==0.1.9",
            ],
            check=True,
        )
        # a SEPARATE install, exactly as Dockerfile.worker runs it: the wheel is prebuilt against
        # torch 2.10, so it needs --no-build-isolation, and that flag must not apply to the resolve
        # above. required, not best-effort -- remove-padding is the default on all three backends and
        # verl's cuda path imports flash_attn.bert_padding unguarded with no sdpa fallback, so a venv
        # without it dies at the first training batch on a paid gpu rather than degrading.
        _install_flash_attn(py)
        conv_installed = _install_causal_conv1d(py)
        # after every install that could have written tilelang into this venv, and before the child
        # ever imports vLLM. only on the rebuild path: a reused venv was repaired when it was built,
        # and re-running it there would turn venv reuse back into work.
        cudart_safe = _neutralize_child_tilelang_cudart_stub(py)
        # written only after ALL installs succeed, so a venv that died between them is
        # unstamped and the next attempt rebuilds it rather than reusing a
        # half-provisioned interpreter. the conv kernel is best-effort to INSTALL but not
        # optional to RECORD. grpo and opd pack, so require_gdn_boundary_resets raises for
        # a gdn model whose child lacks it; stamping a venv that missed the build would
        # hand every later attempt on this pod the same broken interpreter, with the stamp
        # asserting it is fully provisioned. leaving it unstamped costs one rebuild and
        # gives the next attempt a real chance at the kernel.
        # the stub repair is withheld from the stamp for the same reason and on the same terms: it
        # runs only on this rebuild path, so stamping a venv whose stub still shadows libcudart
        # freezes that in, and every later attempt on this pod reuses an interpreter that asserts
        # it is provisioned while its child aborts on vLLM import.
        if conv_installed and cudart_safe:
            with open(stamp, "w") as f:
                f.write(VERL_VENV_STAMP)
        elif not conv_installed:
            print(
                f"[verl] {CAUSAL_CONV1D_REQUIREMENT} did not install; leaving the venv unstamped so "
                "the next attempt rebuilds it rather than reusing an interpreter that cannot honor "
                "gdn boundary resets",
                flush=True,
            )
        else:
            print(
                "[verl] the child tilelang libcudart stub was not neutralized; leaving the venv "
                "unstamped so the next attempt rebuilds it rather than reusing an interpreter "
                "whose child aborts on vLLM import",
                flush=True,
            )
    if install_wandb:
        # keep wandb outside the stamped rebuild branch. wandb is absent from the stamp, so reused venvs
        # and transient failures must retry it on later runs; installing an already satisfied package
        # is a cheap no-op.
        # exact-pinned to the same version baked into Dockerfile.worker's verl venv.
        subprocess.run(["uv", "pip", "install", "--python", py, "wandb==0.28.2"], check=False)
    return py


# the parent neutralizes tilelang's libcudart stub in its OWN site-packages, but the child trains in
# a separate venv with its own tilelang, and vLLM imports THERE. run the same repair against the
# child interpreter, after every install that could have (re)written its tilelang.
#
# inlined rather than imported: flash is not installed in the child venv (the parent resolves
# flash-dependent facts and hands the child literals), so `from flash...` would raise ImportError
# and the repair would silently never happen -- the exact failure this exists to prevent.
_CHILD_CUDART_FIX = """
import ctypes, ctypes.util, glob, importlib.util, os

def find_real():
    def verify(cand):
        try:
            lib = ctypes.CDLL(cand)
        except OSError:
            return None
        if not hasattr(lib, "cudaDeviceReset"):
            return None
        if os.path.isabs(cand) and os.path.exists(cand):
            return os.path.realpath(cand)
        base = os.path.basename(cand)
        try:
            with open("/proc/self/maps") as f:
                for line in f:
                    if base in line and "/" in line:
                        p = line[line.index("/"):].rstrip()
                        if os.path.basename(p).startswith(base) and os.path.exists(p):
                            return os.path.realpath(p)
        except OSError:
            pass
        return None

    candidates = []
    try:
        import nvidia
        for base in sorted(map(str, getattr(nvidia, "__path__", []) or [])):
            candidates += sorted(glob.glob(os.path.join(base, "*", "lib", "libcudart.so.*")))
    except Exception:
        pass
    for pat in (
        "/usr/local/cuda*/lib64/libcudart.so.*",
        "/usr/local/cuda*/targets/*/lib/libcudart.so.*",
        "/usr/lib/x86_64-linux-gnu/libcudart.so.*",
    ):
        candidates += sorted(glob.glob(pat))
    found = ctypes.util.find_library("cudart")
    if found:
        candidates.append(found)
    for cand in candidates:
        real = verify(cand)
        if real is not None:
            return real
    return None

try:
    spec = importlib.util.find_spec("tilelang")
except Exception:
    spec = None
locs = list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
if not locs:
    print("no tilelang in child venv; nothing to neutralize")
else:
    stub = os.path.join(locs[0], "lib", "libcudart_stub.so")
    if not os.path.lexists(stub):
        print("no libcudart_stub.so in child tilelang")
    elif os.path.islink(stub) and os.path.exists(stub):
        print("child stub already repointed")
    else:
        # do NOT ctypes.CDLL the stub to probe it: that dlopens it into /proc/self/maps,
        # which is the crash being prevented.
        real = find_real()
        if real is None:
            # the one outcome that leaves the stub a plain file: it still shadows libcudart, so
            # vLLM aborts its import here. exit nonzero so the caller can tell this apart from the
            # benign no-op cases above, which also print and also exit 0.
            print("no real libcudart found in child; left as-is")
            raise SystemExit(1)
        else:
            backup = stub + ".orig"
            if not os.path.exists(backup):
                os.replace(stub, backup)
            else:
                try:
                    os.remove(stub)
                except FileNotFoundError:
                    pass
            os.symlink(real, stub)
            print("redirected child tilelang libcudart_stub.so -> " + real)
"""


def _neutralize_child_tilelang_cudart_stub(py: str) -> bool:
    """Repoint the CHILD venv's tilelang libcudart stub at a real libcudart.

    vLLM aborts its import when it finds tilelang's stub (no ``cudaDeviceReset``) in
    /proc/self/maps. The parent's fix only ever sees the parent's site-packages, so without this
    the child dies on vLLM import on a paid GPU, after the pod is already rented.

    Returns whether the child venv is safe to import vLLM in, which is NOT the same as "the repair
    ran". A child with no tilelang, or whose stub is already a symlink, has nothing shadowing
    libcudart and is safe untouched -- those exit 0. What is not safe is a stub left as a plain
    file, which the script reports by exiting nonzero. The caller withholds the venv stamp on
    False: the repair only runs on the rebuild path, so a stamped failure is frozen in and every
    later attempt on this pod reuses an interpreter whose child still aborts.

    Still fails open WITHIN a launch: this only decides whether to record the venv as provisioned,
    and never blocks a child that would have imported fine.
    """
    try:
        result = subprocess.run(
            [py, "-c", _CHILD_CUDART_FIX],
            check=False,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"[verl] child libcudart stub neutralize did not run: {error}", flush=True)
        return False
    for stream in (getattr(result, "stdout", None), getattr(result, "stderr", None)):
        if not stream:
            continue
        if isinstance(stream, bytes):
            stream = stream.decode("utf-8", "replace")
        for line in stream.splitlines():
            if line.strip():
                print(f"[verl] child cudart: {line}", flush=True)
    # default 0, matching `_install_causal_conv1d`: a fake subprocess.run in a test returns None,
    # and only an explicit nonzero means the stub was left shadowing libcudart.
    return getattr(result, "returncode", 0) == 0


def resolve_verl_loggers(caps: dict) -> list[str]:
    """verl's ``trainer.logger`` list, gated on the interpreter that actually logs.

    verl logs from its own interpreter, so wandb must be importable THERE, not in flash's env. The
    prebuilt worker image's /opt/verl-venv does not install wandb, and a preset FLASH_VERL_PYTHON is
    returned by resolve_verl_python untouched (flash does not own that interpreter), so a run with
    WANDB_API_KEY set would otherwise ask verl for a logger it cannot import and die at logger init
    on a paid GPU.
    """
    if not os.environ.get("WANDB_API_KEY"):
        return ["console"]
    if not caps.get("wandb"):
        print(
            "[verl] WANDB_API_KEY set but wandb is unavailable in the verl interpreter; "
            "using console logger only"
        )
        return ["console"]
    return ["console", "wandb"]


def verl_device_capability(caps: dict) -> tuple[int, int] | None:
    """device 0's ``(major, minor)`` cuda capability as the VERL interpreter sees it, or None.

    probed in verl's interpreter rather than flash's because verl owns the rollout engine and pins
    its own torch/vllm stack, and both callers below are deciding what that stack can do. None means
    the probe could not answer -- no cuda, no torch, a hung import -- and every caller must read it
    as "leave the default alone" rather than guessing a workaround onto an unknown card.
    """
    cc = caps.get("capability")
    if not cc:
        return None
    try:
        return (int(cc[0]), int(cc[1]))
    except (TypeError, ValueError, IndexError):
        return None


def resolve_rollout_enforce_eager(cc: tuple[int, int] | None) -> bool:
    """whether this GPU must run the rollout eagerly instead of capturing cuda graphs.

    One knob is enough and cannot fight verl's: vllm 0.19.1 resolves ``enforce_eager`` LAST, forcing
    ``compilation_config.mode=NONE`` and ``cudagraph_mode=NONE`` regardless of what was requested
    (``config/vllm.py:847-853`` and ``:1024-1029``). An unanswerable probe (``cc is None``) leaves
    verl's default in place rather than guessing eager onto an unknown card.
    """
    if cc is None:
        return False
    # sm86 is the only measured graph-capture failure. everything else -- sm80/sm89/sm90 and
    # blackwell -- captures. an arch nobody has measured keeps verl's default rather than being
    # forced eager on the strength of one other Ampere minor version's defect.
    if cc == (8, 6):
        print(
            "[verl] sm86: enforce_eager=True for the rollout (vllm 0.19.1 graph capture degenerates "
            "on this arch: completions repeat to the token cap without emitting EOS)"
        )
        return True
    return False


def rollout_sleep_unsupported(model_id: str) -> bool:
    """whether vLLM's sleep/wake cycle is non-functional for this model.

    a few models HANG on wake/reload rather than erroring (live-confirmed, every attempt), so the
    catalog flags them and the rollout engine must stay RESIDENT instead. shared by grpo and opd:
    verl runs the same ``vllm_async_server`` sleep path for both, so the flag has to mean the same
    thing on both drivers or the one that ignores it wedges on the flagged model.
    """
    from flash.core.catalog import MODELS

    info = MODELS.get(model_id)
    return bool(info is not None and getattr(info, "sleep_unsupported", False))


def rollout_fp8_kv(cc_ok: bool, gdn_hybrid: bool, model_id: str) -> bool:
    """whether the rollout engine may cache KV in fp8. shared by grpo and opd.

    ``cc_ok`` (cc>=8.9) is the hardware floor. the gdn exclusion on top of it is narrower than it
    looks: the crash it avoids is vllm's ``init_fp8_kv_scales``, which zeroes each cache tensor and
    hits ``'list' object has no attribute 'zero_'`` on a hybrid cache. that function is reachable
    from exactly ONE place -- ``gpu_worker.wake_up`` (``v1/worker/gpu_worker.py:198-203``) -- and
    verl only ever wakes an engine it put to sleep: both training-path callers are gated on
    ``rollout.free_cache_engine`` (``engine_workers.py:706,741`` -> ``vllm_rollout.py:161``). a model
    the catalog flags ``sleep_unsupported`` runs with that flag pinned false by
    ``rollout_resident_overrides``, so it never sleeps, never wakes, and never reaches the crash.

    the resident engine still gets fp8 for its ordinary attention layers: the recurrent state is a
    separate ``MambaSpec`` carrying its own dtypes, so fp8 never applies to it. vram sizing already
    prices this exact combination -- ``mamba_block_size=1072`` is the fp8-derived attention-token
    page (the 1,097,728-byte gdn state over a 1024-byte fp8 attention token, rounded to vllm's
    16-token alignment) and ``kv_sizing`` applies it only when ``fp8_kv`` is set.

    this also closes a planner/worker disagreement rather than merely freeing memory: the grpo
    resident wall in ``vram.py`` sizes a ``sleep_unsupported`` model with ``fp8_kv=True`` already,
    so a worker that sent bf16 needed ~6-7 GB MORE than the planner had reserved for it (measured
    at 4096x8 and 8192x4). the two halves now assume the same cache dtype.
    """
    if not cc_ok:
        return False
    if not gdn_hybrid:
        return True
    # gdn + sleeping engine is the combination that crashes on wake; gdn + resident is not.
    return rollout_sleep_unsupported(model_id)


_MIN_ROLLOUT_MAX_NUM_SEQS = 16


def rollout_max_num_seqs(rollout_batch: int) -> int:
    """concurrent sequence slots to provision the rollout engine for.

    verl leaves ``rollout.max_num_seqs`` at its yaml default of 1024 (rollout.yaml:79) and passes
    it straight to the vllm engine (vllm_async_server.py:261). vllm sizes two up-front, fixed
    allocations from that number, neither of which shrinks to the batch actually submitted:

    * cuda-graph capture. ``_set_cudagraph_sizes`` (vllm ``config/vllm.py``, 0.19.1 -- the version
      ``Dockerfile.worker`` pins) derives a capture ladder from
      ``max_cudagraph_capture_size = min(max_num_seqs * 2, 512, max_num_batched_tokens)``, taking
      ``[1,2,4]``, then ``range(8, min(size+1, 256), 8)``, then ``range(256, size+1, 16)`` -- each
      rung clamped by that ceiling. verl's ``max_num_batched_tokens`` default is 8192
      (``rollout.yaml:73``), so it does not bind. at 1024 the ceiling is 512 and the ladder is 51
      sizes; a lora run captures each size twice, since the dispatcher takes
      ``product(cudagraph_capture_sizes, lora_cases)`` with ``lora_cases == [0, 1]`` when
      ``cudagraph_specialize_lora`` is on (``v1/cudagraph_dispatcher.py:110-193``, default True) --
      exactly the ``0/102`` seen in the failing console. the ladder saturates at 512, so 1024, 512
      and 256 are indistinguishable; only dropping below 256 shortens it, and 32 yields a ceiling
      of 64, 11 sizes, and 22 graphs. note the prose comment above that function omits the per-rung
      clamp and predicts 34 sizes for any small value; the implementation is authoritative.
    * recurrent state. a gdn/mamba hybrid reserves one state block per decode slot, a dense
      allocation that -- unlike paged kv -- cannot be shared, paged out, or grown on demand.

    both are paid before the first rollout token, inside the ``gpu_memory_utilization`` budget. a
    grpo/opd step submits exactly ``prompts_per_step * group_size`` sequences, so a run with 32
    concurrent generations was provisioning for 1024 and dying during graph capture with ~80% of
    the card free -- on 234, 358, and 460 GB alike, since the cost is fixed rather than
    proportional to the card.

    the floor keeps small runs on a usable capture ladder rather than collapsing it to one size.
    """
    if rollout_batch <= 0:
        raise ValueError("rollout_batch must be positive")
    return max(_MIN_ROLLOUT_MAX_NUM_SEQS, int(rollout_batch))


def rollout_resident_overrides(sleep_unsupported: bool) -> list[str]:
    """verl overrides that pin the rollout engine resident, or ``[]`` to keep verl's default.

    disable both ``free_cache_engine`` and ``enable_sleep_mode`` or flagged models hang on wake
    (``vllm_async_server.py:626,648`` gates sleep on both). VRAM sizing already uses the resident
    peak. prefixes differ: the yaml key takes a bare override, while ``enable_sleep_mode`` is
    dataclass-only (``workers/config/rollout.py:277``) and so requires ``+``.
    """
    if not sleep_unsupported:
        return []
    return [
        "actor_rollout_ref.rollout.free_cache_engine=false",
        "+actor_rollout_ref.rollout.enable_sleep_mode=false",
    ]


def rollout_layered_summon_overrides(target_parameters: Sequence[str] | None) -> list[str]:
    """gather the fsdp weight sync one submodule at a time for fused-expert lora models.

    the rollout weight sync collects lora params through ``collect_lora_params``
    (verl ``utils/fsdp_utils.py:673``). its default branch opens
    ``FSDP.summon_full_params(module)`` over the WHOLE model, so every layer's expert stack is
    unsharded and resident simultaneously. ``layered_summon`` takes
    ``layered_summon_lora_params`` instead, which summons one submodule at a time and calls
    ``empty_cache()`` between layers, so the peak is a single layer's slab.

    that difference only matters when a target tensor is large, and a non-empty
    ``target_parameters`` is exactly the fused-moe case. there ``mlp.experts.gate_up_proj`` is a
    fused bf16 slab of ``lora_expert_count x 2048 x 1024 x 2`` bytes -- on Qwen3.6-35B-A3B
    (``lora_expert_count=256``) that is 1,073,741,824 bytes, i.e. EXACTLY 1024 MiB, which is
    precisely the contiguous request that OOM'd at step 0 on a 180 GB B200 across every card count
    and context tested. dense models keep verl's default: their per-tensor gathers are small and
    the layered walk is slower.

    verl gates the layered path on the base weights already living in vllm
    (``fsdp_utils.py:684``: it raises unless ``base_sync_done``, which
    ``engine_workers.py:614`` derives as ``"dummy" not in load_format``). the rl and opd drivers
    both already pin ``load_format=safetensors``, so that precondition holds wherever this
    applies. ``layered_summon`` is a rollout dataclass field
    (``workers/config/rollout.py:261``) that is also declared in ``rollout.yaml:98``, so it takes
    a bare override.
    """
    if not target_parameters:
        return []
    return ["actor_rollout_ref.rollout.layered_summon=true"]


def actor_fsdp_strategy_overrides(fsdp_generation: FsdpGeneration) -> list[str]:
    """render one explicit bare actor strategy override from the resolved private policy."""
    strategy = "fsdp2" if fsdp_generation == 2 else "fsdp"
    return [f"actor_rollout_ref.actor.strategy={strategy}"]


def rollout_mm_processor_cache_overrides() -> list[str]:
    """disable vllm's split multimodal processor cache for every verl rollout.

    verl sleep resets the engine-side receiver without resynchronizing the frontend sender, so a
    repeated image can become a hash-only request for an item the receiver no longer holds. zero is
    vllm's disable value for both halves and is inert when a rollout has no multimodal items.
    """
    return ["+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0"]


def trainer_dtype_overrides() -> list[str]:
    """verl overrides that store the frozen base in bf16 instead of verl's fp32 default.

    fsdp.yaml passes its fp32 default to ``from_pretrained`` (``transformer_impl.py:234``). flash
    freezes the base and computes under bf16 mixed precision, while PEFT keeps trainable adapter
    weights fp32 (``peft/mapping_func.py:35``, not overridden by verl), so fp32 base residency has
    no master-weight benefit. this is specific to LoRA, not full tuning.

    set only ``actor`` because ``ref_in_actor`` aliases the reference worker to it
    (``ray_trainer.py:897``), rather than creating a second resident copy.
    """
    return ["actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"]


def resolve_blackwell_attention_backends(
    caps: dict, cc: tuple[int, int] | None
) -> tuple[str | None, str | None]:
    """the rollout ``(attention_backend, mm_encoder_attn_backend)`` this GPU needs, or ``(None, None)``.

    on Blackwell, use FLASHINFER or TRITON_ATTN for decoder attention; TORCH_SDPA is not registered
    there in vllm 0.19.1 (``vllm/platforms/cuda.py:99-111``). pin ViT attention to TORCH_SDPA
    because no published cutlass version satisfies both CUTE APIs; ``cuda.py:344-373`` picks the
    default ViT backend and honors an explicit override. non-Blackwell or unknown caps keep defaults.
    """
    if cc is None:
        return (None, None)
    major = cc[0]
    if major not in (10, 12):
        return (None, None)
    has_flashinfer = bool(caps.get("flashinfer"))
    decoder = "FLASHINFER" if has_flashinfer else "TRITON_ATTN"
    if not has_flashinfer:
        print(
            f"[verl] sm{major}0 (Blackwell): flashinfer is not importable in the verl interpreter "
            "-> attention_backend=TRITON_ATTN (PTX-independent registered decoder backend)"
        )
    else:
        print(
            f"[verl] sm{major}0 (Blackwell): attention_backend=FLASHINFER "
            "(flash-attn PTX is unreliable on consumer Blackwell -> empty-rollout failures)"
        )
    print(
        f"[verl] sm{major}0 (Blackwell): mm_encoder_attn_backend=TORCH_SDPA "
        "(vllm 0.19.1 ViT CUTE flash-attn is unimportable vs every nvidia-cutlass-dsl)"
    )
    return (decoder, "TORCH_SDPA")
