"""Source for sitecustomize shims rendered into the verl child interpreter.

Renderers return source text because the pinned child cannot import flash. Active shims print a
marker so the parent can prove each patch executed.

NOTHING HERE MAY IMPORT torch, verl OR vllm AT FRAGMENT SCOPE -- see `_deferred_patch` below.
"""

from __future__ import annotations

from flash.engine.worker.verl.child_io import SHIM_FRAGMENT_FAILED_EXIT_CODE

# Why every fragment below defers its imports into a function body.
#
# sitecustomize runs at INTERPRETER STARTUP, and ray starts each actor's interpreter BEFORE it
# narrows that actor's CUDA_VISIBLE_DEVICES to its own card. `verl/utils/device.py` evaluates
# `torch.cuda.is_available()` at MODULE scope, and torch's own `is_available` calls
# `cudaGetDeviceCount`, which initializes the CUDA driver via `cuInit`. So a fragment that imports
# any verl module at startup builds the CUDA device map against the FULL device list, and a later
# env-only narrowing cannot rebuild it -- every rank keeps device 0 and nccl aborts init with
#
#   Duplicate GPU detected : rank 1 and rank 0 both on CUDA device <pci>
#
# from `WorkerDict.actor_rollout_init_model()`, on hardware whose cards are all present and idle.
#
# `render_gdn_varlen_shim` (verl/child_io.py) already defers for exactly this reason. It fixed the
# one fragment it owned; these were left eager, and the reentrant-checkpointing fragment renders on
# EVERY grpo run -- `grpo_use_reentrant` is true for every catalog model, since all of them are
# GatedDeltaNet hybrids -- so the collapse survived that fix on every multi-card run.
#
# A deferred patch runs at the first import of the module it patches instead, which happens inside
# the actor after ray has pinned it. `render_deferred_patch_runtime` emits the ONE registry that
# arranges this and `_deferred_patch` registers each fragment with it; `test_rl_train.py` asserts no
# fragment reintroduces a module-scope import of torch, verl or vllm.
#
# ONE finder holding many callbacks, never one finder per fragment. Fragments genuinely share
# targets -- the stop-sequence and image-pad patches both hook the agent loop -- and a finder that
# has to delegate around its siblings either recurses forever (each skips only itself, so they call
# each other) or silently drops a patch (the first match wins and the sibling never fires). Both
# were reproduced against the per-fragment design this replaced. A registry keyed by module name has
# neither failure mode: one interception, every callback for that module, in registration order.
_DEFER_RUNTIME_MARKER = "_flash_defer_registry"


def render_deferred_patch_runtime() -> str:
    """emit the shared deferral registry every ``_deferred_patch`` fragment registers with.

    Rendered ONCE per sitecustomize, above the fragments. Idempotent so a caller that emits it
    twice cannot discard callbacks already registered by the first copy.
    """
    return f'''
# --- flash: run each verl patch at ITS module's import, not at interpreter startup ---
# (see train/rl/shims.render_deferred_patch_runtime)
import sys as _flash_defer_sys


def _flash_defer_fail(target, name):
    """a required patch could not apply: kill the child rather than let it train unpatched.

    ``wrap_shim_fragment``'s try/except cannot cover this. It spans the REGISTRATION, which has
    already returned by the time the body runs, so a raise here would surface as an ordinary
    ImportError out of the target's import -- and an importer that catches it and retries gets a
    clean load: python drops the failed module from sys.modules, this registry has already popped
    the callback, and the target comes back with NO patch and NO marker. The interpreter then keeps
    running unpatched until the parent happens to check markers, which for a correctness-critical
    patch (kl anchoring, save gating, rank/device pinning) means a paid run producing wrong output.

    os._exit is the same escape hatch the wrapper uses: it cannot be swallowed by an except in the
    importer, by execsitecustomize, or by ray's own import error handling.
    """
    import os as _flash_defer_os
    import traceback as _flash_defer_traceback

    _flash_defer_traceback.print_exc()
    print(
        "[flash-verl] required shim fragment " + repr(name) + " failed to apply at the import of "
        + repr(target) + "; exiting {SHIM_FRAGMENT_FAILED_EXIT_CODE} rather than training unpatched",
        file=_flash_defer_sys.stderr,
        flush=True,
    )
    _flash_defer_sys.stderr.flush()
    _flash_defer_os._exit({SHIM_FRAGMENT_FAILED_EXIT_CODE})


if not hasattr(_flash_defer_sys, "_flash_defer_registry"):

    class _FlashDeferRegistry:
        """maps a module name to the patches waiting on it, and installs one meta_path finder."""

        def __init__(self):
            self._pending = {{}}
            self._armed = False

        def register(self, target, body):
            # already imported: nothing left to intercept, so apply now. this keeps the contract
            # identical whether or not a parent process happened to import the module first.
            if target in _flash_defer_sys.modules:
                self._apply(target, body)
                return
            self._pending.setdefault(target, []).append(body)
            if not self._armed:
                _flash_defer_sys.meta_path.insert(0, self)
                self._armed = True

        def _apply(self, target, body):
            # every path that runs a body goes through here, so there is one place a failure can
            # be handled and no way to add a caller that quietly skips the hard exit.
            try:
                body()
            except BaseException:
                _flash_defer_fail(target, getattr(body, "_flash_shim_name", "<unknown>"))

        def _run(self, target):
            # pop first: a body that imports its own target must not re-enter this. popping is safe
            # only because _apply cannot return after a failure -- see _flash_defer_fail.
            for body in self._pending.pop(target, []):
                self._apply(target, body)
            if not self._pending:
                self.uninstall()

        def uninstall(self):
            _flash_defer_sys.meta_path[:] = [
                f for f in _flash_defer_sys.meta_path if f is not self
            ]
            self._armed = False

        def find_spec(self, fullname, path=None, target=None):
            if fullname not in self._pending:
                return None
            # delegate to the finders AFTER this one. that resolves the real spec without importing
            # anything, so arming touches no cuda; only the loader is wrapped.
            for finder in [f for f in _flash_defer_sys.meta_path if f is not self]:
                find = getattr(finder, "find_spec", None)
                if find is None:
                    continue
                spec = find(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = _FlashDeferLoader(spec.loader, self, fullname)
                    return spec
            return None

    class _FlashDeferLoader:
        """wrap the real loader so patches land once the module is fully executed.

        ``exec_module`` returning means the module the importer will hand the caller is finished,
        so this is the first safe moment. Patching from ``find_spec`` instead would run against a
        half-built module that the real import then replaces.
        """

        def __init__(self, inner, registry, fullname):
            self._inner = inner
            self._registry = registry
            self._fullname = fullname

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            self._inner.exec_module(module)
            self._registry._run(self._fullname)

        def __getattr__(self, name):  # keep the rest of the loader protocol intact
            return getattr(self._inner, name)

    _flash_defer_sys._flash_defer_registry = _FlashDeferRegistry()
'''


def render_rank_device_assert_shim(n_gpus: int) -> str:
    """assert every rank opened a DISTINCT physical gpu, before any model is loaded.

    The failure this catches is expensive by construction. A collapsed rank->device map is
    classified non-retriable and fires only after allocation, image pull, model prefetch and ray
    startup -- roughly a minute AFTER ``stage=rl_step step=0`` prints, so a status check in that
    window shows a healthy running run with no error. The abort that eventually arrives names a pci
    id and no rank mapping, and the surrounding wrapper messages name none of it.

    Checking here converts that into an immediate, self-describing failure with the mapping
    attached. It is deliberately cheap: torch already knows the ordinal, and the uuid comes from the
    device properties torch has loaded anyway, so there is no extra probe and nothing to time out.

    Single-card runs are skipped -- one rank cannot collide with itself, and the collective path
    that fails is never entered.

    Fails CLOSED. If two ranks share a device, every later symptom (nccl abort, an OOM at half the
    expected capacity, or a run that silently trains on one card) is worse than refusing here.
    """
    if int(n_gpus) < 2:
        return ""
    return _deferred_patch(
        "rankdevice",
        "verl.single_controller.base.worker",
        f'''
import os as _flash_rank_os

import torch as _flash_rank_torch
from verl.single_controller.base import worker as _flash_rank_worker

_FLASH_EXPECTED_RANKS = {int(n_gpus)}
_flash_rank_original_init = _flash_rank_worker.Worker.__init__


def _flash_rank_device_identity():
    """(ordinal, uuid) for the device THIS rank would train on, or None if it cannot be read."""
    if not _flash_rank_torch.cuda.is_available():
        return None
    ordinal = _flash_rank_torch.cuda.current_device()
    try:
        # the uuid is what makes this a real check: two ranks that both report ordinal 0 may still
        # be two different physical cards (each with its own CUDA_VISIBLE_DEVICES), while two ranks
        # on ONE card share a uuid no matter what ordinals they report.
        uuid = str(getattr(_flash_rank_torch.cuda.get_device_properties(ordinal), "uuid", "") or "")
    except Exception:
        uuid = ""
    return ordinal, uuid


def _flash_rank_check(self):
    identity = _flash_rank_device_identity()
    if identity is None:
        return
    ordinal, uuid = identity
    rank = int(_flash_rank_os.environ.get("RANK", "0"))
    local_rank = int(_flash_rank_os.environ.get("LOCAL_RANK", "0"))
    visible = _flash_rank_os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(
        "[rl-verl] rank-device binding rank=" + repr(rank)
        + " local_rank=" + repr(local_rank)
        + " cuda_ordinal=" + repr(ordinal)
        + " visible_devices=" + repr(visible)
        + " uuid=" + repr(uuid or "<unavailable>"),
        flush=True,
    )
    if not uuid:
        # without a uuid there is nothing to compare, and inventing a pass would defeat the check.
        # the line above still records the binding for a post-mortem.
        return
    # rendezvous through a file rather than a collective: this runs BEFORE the process group
    # exists, which is the whole point -- nccl's own duplicate-device check is what we are trying
    # to beat to the failure. the directory is per-run and the writes are O_APPEND single lines.
    claims_path = _flash_rank_os.environ.get("FLASH_RANK_DEVICE_CLAIMS", "")
    if not claims_path:
        return
    with open(claims_path, "a") as handle:
        handle.write(repr(rank) + " " + uuid + "\\n")
    claims = {{}}
    with open(claims_path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 2:
                claims.setdefault(parts[1], set()).add(int(parts[0]))
    collided = {{u: sorted(r) for u, r in claims.items() if len(r) > 1}}
    if collided:
        raise RuntimeError(
            "flash: ranks were bound to the SAME physical gpu before training started -- "
            + repr(collided)
            + " (uuid -> ranks). this rank: rank=" + repr(rank)
            + " local_rank=" + repr(local_rank)
            + " cuda_ordinal=" + repr(ordinal)
            + " CUDA_VISIBLE_DEVICES=" + repr(visible)
            + ". expected " + repr(_FLASH_EXPECTED_RANKS) + " ranks on distinct cards; nccl would "
            "abort a minute later with 'Duplicate GPU detected' and no rank mapping."
        )


def _flash_rank_init(self, *args, **kwargs):
    result = _flash_rank_original_init(self, *args, **kwargs)
    # AFTER verl's own __init__: that is what applies ray's CUDA_VISIBLE_DEVICES narrowing and the
    # set_device call, so checking before it would measure the unpinned state and always pass.
    _flash_rank_check(self)
    return result


if not getattr(_flash_rank_worker.Worker.__init__, "_flash_rank_checked", False):
    _flash_rank_init._flash_rank_checked = True
    _flash_rank_worker.Worker.__init__ = _flash_rank_init
''',
        "rank-device-assert",
    )


def _deferred_patch(tag: str, target: str, body: str, marker: str) -> str:
    """Run ``body`` when ``target`` finishes importing, rather than at sitecustomize time.

    ``target`` is the module the fragment actually patches, so the body runs at the first moment
    that module exists and still before verl uses it. Hooking the exact module rather than the
    ``verl`` package matters: the body imports heavy submodules, and doing that while the parent
    package is itself mid-execution would run against a half-initialized ``verl``.

    ``body`` may import torch/verl/vllm freely -- by the time it runs, ray has narrowed this actor
    to its own card, so the CUDA context it builds is the right one.

    ``tag`` uniquifies the closure this emits. Fragments are concatenated into ONE sitecustomize at
    module scope, so a shared function name would let the last fragment silently replace an earlier
    one's body.

    ``marker`` is the fragment's name, recorded from INSIDE the body once the patch has landed.
    Deferral moves the work off sitecustomize time, so ``wrap_shim_fragment``'s own
    ``_flash_record_applied_shim`` call no longer sits after the patch -- it sits after the
    REGISTRATION, and would prove only that a callback was queued. The parent's
    ``verify_applied_shim_markers`` would then accept a child whose patch never ran, which is
    exactly the train-unpatched hole the wrapper exists to close. Recording here keeps one marker
    meaning one thing: this patch is installed.

    Fail-closed in both directions: an exception in the body propagates out of the child's own
    import of ``target`` (killing the run), and a body that never runs records no marker (failing
    the parent's check).

    Requires ``render_deferred_patch_runtime()`` earlier in the same sitecustomize.
    """
    import textwrap

    applied = f"{textwrap.dedent(body).strip()}\n_flash_record_applied_shim({marker!r})"
    return f"""
def _flash_deferred_body_{tag}():
{textwrap.indent(applied, "    ")}


# the registry reads this to name the fragment in its hard-exit message; without it a failed patch
# reports only the module it was hooking, which is shared by several fragments.
_flash_deferred_body_{tag}._flash_shim_name = {marker!r}
_flash_defer_sys._flash_defer_registry.register({target!r}, _flash_deferred_body_{tag})
"""


_ENTROPY_QUANTILE_MARKER = "[flash-verl] top-entropy token masking active"
_STOP_SEQUENCES_MARKER = "[flash-verl] rollout stop strings active"
_STRUCTURED_OUTPUTS_MARKER = "[flash-verl] rollout structured outputs active"
_EXACT_SAVE_STEPS_MARKER = "[flash-verl] exact save steps active"
_IMAGE_PAD_BAN_MARKER = "[flash-verl] image-pad token banned from rollouts"
_KL_REF_ADAPTER_MARKER = "[flash-verl] kl reference anchored to the warm-start adapter"


def render_kl_ref_adapter_shim(warmstart: bool) -> str:
    """return source that anchors verl's kl reference to the warm-start adapter.

    ``no_lora_adapter=True`` otherwise evaluates the bare base and pulls a warm start backward.
    Store the frozen reference as non-persistent buffers so FSDP, optimizers, state dicts, and
    ``base_model_merger.save_lora_adapter`` never treat it as a second trainable adapter. Rebuild it
    from ``lora_adapter_path`` on resume. Swap ``BaseTunerLayer._active_adapter`` directly because
    peft's ``set_adapter`` changes ``requires_grad`` and breaks FSDP flat-parameter uniformity.
    """
    if not warmstart:
        return ""
    return _deferred_patch(
        "klref",
        "verl.workers.engine.fsdp.transformer_impl",
        f'''
from contextlib import contextmanager as _flash_ref_contextmanager

import torch.nn as _flash_ref_nn
from peft.tuners.tuners_utils import BaseTunerLayer as _flash_ref_tuner_layer
from verl.workers.engine.fsdp import transformer_impl as _flash_ref_impl

_FLASH_REF_ADAPTER = "flash_kl_ref"


def _flash_ref_snapshot(module):
    """freeze a copy of the warm-start adapter under a second, non-trainable adapter name."""
    if _FLASH_REF_ADAPTER in getattr(module, "peft_config", {{}}):
        return module
    module.add_adapter(_FLASH_REF_ADAPTER, module.peft_config[module.active_adapter])
    for name, param in module.named_parameters():
        if ".default." in name:
            twin = module.get_parameter(name.replace(".default.", f".{{_FLASH_REF_ADAPTER}}."))
            twin.data.copy_(param.data)
    # demote the snapshot from parameters to non-persistent buffers. lora_A/lora_B are ModuleDicts
    # keyed by adapter name, so the snapshot's leaves are exactly the ones under our key.
    demoted = 0
    for container in module.modules():
        if isinstance(container, _flash_ref_nn.ModuleDict) and _FLASH_REF_ADAPTER in container:
            leaf = container[_FLASH_REF_ADAPTER]
            for attr, value in list(leaf.named_parameters(recurse=False)):
                frozen = value.detach().clone()
                delattr(leaf, attr)
                leaf.register_buffer(attr, frozen, persistent=False)
                demoted += 1
    if not demoted:
        raise RuntimeError("flash kl reference snapshot found no adapter weights to freeze")
    print({_KL_REF_ADAPTER_MARKER!r} + " " + repr(demoted), flush=True)
    return module


_flash_ref_original_build_lora = _flash_ref_impl.FSDPEngine._build_lora_module


def _flash_ref_build_lora_module(self, module):
    # after the warm-start adapter is loaded, before _build_fsdp_module wraps it.
    return _flash_ref_snapshot(_flash_ref_original_build_lora(self, module))


@_flash_ref_contextmanager
def _flash_ref_use_ref_adapter(module):
    """activate the frozen snapshot without touching any requires_grad flag."""
    layers = [m for m in module.modules() if isinstance(m, _flash_ref_tuner_layer)]
    if not layers:
        raise RuntimeError("flash kl reference: no lora layers on the actor module")
    saved = [layer._active_adapter for layer in layers]
    for layer in layers:
        layer._active_adapter = [_FLASH_REF_ADAPTER]
    try:
        yield
    finally:
        for layer, previous in zip(layers, saved, strict=True):
            layer._active_adapter = previous


def _flash_ref_disable_adapter(self):
    # this shim is only written for a warm start, so the snapshot must exist. falling back to the
    # stock disable_adapter() here would silently anchor the kl term to the base -- the exact
    # defect this patch removes -- and the run would look healthy while training the wrong thing.
    module = self.module
    inner = getattr(module, "_fsdp_wrapped_module", module)
    if _FLASH_REF_ADAPTER not in getattr(inner, "peft_config", {{}}):
        raise RuntimeError(
            "flash kl reference adapter missing: expected " + _FLASH_REF_ADAPTER + " on the actor"
        )
    return _flash_ref_use_ref_adapter(module)


_flash_ref_impl.FSDPEngine._build_lora_module = _flash_ref_build_lora_module
_flash_ref_impl.FSDPEngine.disable_adapter = _flash_ref_disable_adapter
''',
        "kl-ref-adapter",
    )


def render_structured_outputs_shim(structured_outputs: dict | None) -> str:
    """return source that constrains verl rollout sampling to a guided grammar.

    Wrap the value in ``StructuredOutputsParams``: vllm silently accepts a raw dict but applies no
    constraint. The object survives ``server.generate.remote(...)`` through Ray cloudpickle. The
    reasoning-parser engine override remains in ``_build_verl_overrides``.
    """
    if not structured_outputs:
        return ""
    return _deferred_patch(
        "structuredoutputs",
        "verl.experimental.agent_loop",
        f'''
from verl.experimental.agent_loop import agent_loop as _flash_so_agent_loop
from vllm.sampling_params import StructuredOutputsParams as _FlashStructuredOutputsParams

_flash_structured_outputs = {structured_outputs!r}


def _flash_patch_structured_outputs():
    """add ``structured_outputs`` to the per-sample sampling params on their way into the loop.

    patched on ``_run_agent_loop`` for the same reason as the stop strings: it receives the
    per-sample dict after verl's validate/greedy overrides, so the constraint also applies to
    validation rollouts -- as on the retired trl path, where it lived in generation_kwargs and was not swapped out
    for eval.
    """
    original = _flash_so_agent_loop.AgentLoopWorker._run_agent_loop

    async def _run_agent_loop(self, sampling_params, *args, **kwargs):
        params = dict(sampling_params)
        # build a fresh params object per request: vllm mutates sampling params in place (it
        # resolves the structured-outputs backend on first use and caches it on the instance), so a
        # shared one would leak that resolution across requests.
        params["structured_outputs"] = _FlashStructuredOutputsParams(**_flash_structured_outputs)
        return await original(self, params, *args, **kwargs)

    _flash_so_agent_loop.AgentLoopWorker._run_agent_loop = _run_agent_loop


if not getattr(
    _flash_so_agent_loop.AgentLoopWorker._run_agent_loop, "_flash_so_patched", False
):
    _flash_patch_structured_outputs()
    _flash_so_agent_loop.AgentLoopWorker._run_agent_loop._flash_so_patched = True
    print({_STRUCTURED_OUTPUTS_MARKER!r} + " " + repr(_flash_structured_outputs), flush=True)
''',
        "structured-outputs",
    )


def render_exact_save_steps_shim(save_at_steps: tuple[int, ...], total_steps: int) -> str:
    """return source that suppresses verl's extra checkpoint writes.

    verl's gcd ``save_freq`` writes a superset of arbitrary ``save_at_steps``. Keep only requested
    and final steps. Read ``self.global_steps`` because ``RayPPOTrainer._save_checkpoint`` has no
    step argument; skipped steps must not advance ``latest_checkpointed_iteration.txt``.
    """
    if not save_at_steps:
        return ""
    return _deferred_patch(
        "exactsave",
        "verl.trainer.ppo.ray_trainer",
        f'''
from verl.trainer.ppo import ray_trainer as _flash_save_ray_trainer

_flash_required_save_steps = frozenset({tuple(sorted(save_at_steps))!r})
_flash_total_steps = {int(total_steps)}

def _flash_patch_exact_save_steps():
    """save only the steps flash asked for, plus the final step.

    the gcd interval makes verl save a superset; every extra write is a full-state dump that is
    never published and is pruned again a few steps later. the last step stays because the run's
    final publish reads the checkpoint verl writes there.
    """
    original = _flash_save_ray_trainer.RayPPOTrainer._save_checkpoint

    def _save_checkpoint(self):
        step = int(self.global_steps)
        if step not in _flash_required_save_steps and step != _flash_total_steps:
            return None
        return original(self)

    _flash_save_ray_trainer.RayPPOTrainer._save_checkpoint = _save_checkpoint

if not getattr(
    _flash_save_ray_trainer.RayPPOTrainer._save_checkpoint, "_flash_save_patched", False
):
    _flash_patch_exact_save_steps()
    _flash_save_ray_trainer.RayPPOTrainer._save_checkpoint._flash_save_patched = True
    print(
        {_EXACT_SAVE_STEPS_MARKER!r} + " " + repr(sorted(_flash_required_save_steps))
        + " final=" + repr(_flash_total_steps),
        flush=True,
    )
''',
        "exact-save-steps",
    )


def render_stop_sequences_shim(stop_sequences: tuple[str, ...]) -> str:
    """return source that adds flash stop strings to verl rollouts.

    Patch ``_run_agent_loop`` where the per-sample dict is owned and passed into vllm
    ``SamplingParams``. vllm truncates text but preserves ``token_ids``, so trained tokens retain
    trailing delimiter tokens.
    """
    if not stop_sequences:
        return ""
    return _deferred_patch(
        "stopsequences",
        "verl.experimental.agent_loop",
        f'''
from verl.experimental.agent_loop import agent_loop as _flash_agent_loop

_flash_stop_sequences = {list(stop_sequences)!r}


def _flash_patch_run_agent_loop():
    """add ``stop`` to the per-sample sampling params on their way into the agent loop.

    ``_run_agent_loop`` receives the fully-built dict for one sample, after verl has applied its
    validation/greedy overrides. patching here rather than at dict construction means the stop
    strings survive those overrides and apply to validation rollouts too, as on the retired trl path, where the
    stop list lives in generation_kwargs and is not swapped out for eval.
    """
    original = _flash_agent_loop.AgentLoopWorker._run_agent_loop

    async def _run_agent_loop(self, sampling_params, *args, **kwargs):
        params = dict(sampling_params)
        params["stop"] = list(_flash_stop_sequences)
        return await original(self, params, *args, **kwargs)

    _flash_agent_loop.AgentLoopWorker._run_agent_loop = _run_agent_loop


# patch once. wrapping twice would be harmless here (the key is overwritten, not appended) but the
# guard keeps the behavior obvious and matches the entropy shim.
if not getattr(_flash_agent_loop.AgentLoopWorker._run_agent_loop, "_flash_stop_patched", False):
    _flash_patch_run_agent_loop()
    _flash_agent_loop.AgentLoopWorker._run_agent_loop._flash_stop_patched = True
    print({_STOP_SEQUENCES_MARKER!r} + " " + repr(_flash_stop_sequences), flush=True)
''',
        "stop-sequences",
    )


def render_image_pad_ban_shim(image_pad_token_id: int | None) -> str:
    """return source that prevents multimodal rollouts from emitting the image-pad token.

    A sampled placeholder has no image and breaks processor alignment on the next forward pass.
    Apply ``-100.0`` logit bias to every row in a multimodal job, including text-only rows.
    """
    if image_pad_token_id is None:
        return ""
    return _deferred_patch(
        "imagepadban",
        "verl.experimental.agent_loop",
        f"""
from verl.experimental.agent_loop import agent_loop as _flash_image_agent_loop

_flash_image_pad_token_id = {int(image_pad_token_id)!r}


def _flash_patch_image_pad_ban():
    original = _flash_image_agent_loop.AgentLoopWorker._run_agent_loop

    async def _run_agent_loop(self, sampling_params, *args, **kwargs):
        params = dict(sampling_params)
        logit_bias = dict(params.get("logit_bias") or {{}})
        logit_bias[_flash_image_pad_token_id] = -100.0
        params["logit_bias"] = logit_bias
        return await original(self, params, *args, **kwargs)

    _flash_image_agent_loop.AgentLoopWorker._run_agent_loop = _run_agent_loop


# patch once, and independently of the stop-strings patch: both wrap the same method, so each
# needs its own marker attribute or the second would be skipped by the first one's flag.
if not getattr(
    _flash_image_agent_loop.AgentLoopWorker._run_agent_loop, "_flash_image_pad_patched", False
):
    _flash_patch_image_pad_ban()
    _flash_image_agent_loop.AgentLoopWorker._run_agent_loop._flash_image_pad_patched = True
    print({_IMAGE_PAD_BAN_MARKER!r} + " " + repr(_flash_image_pad_token_id), flush=True)
""",
        "image-pad-ban",
    )


def render_per_turn_credit_shim(per_turn_credit: bool) -> str:
    """return source that adds per-turn group-relative credit to verl.

    Wrap ``compute_advantage`` because custom estimators do not receive ``non_tensor_batch`` spans.
    Keep stock GRPO as the baseline and overwrite token advantages. Fallback must cover the whole
    group so episode and per-turn scales are never mixed.
    """
    if not per_turn_credit:
        return ""
    return _deferred_patch(
        "perturncredit",
        "verl.trainer.ppo.ray_trainer",
        '''
import torch as _flash_pt_torch
from verl.trainer.ppo import ray_trainer as _flash_pt_ray_trainer

_flash_pt_original_compute_advantage = _flash_pt_ray_trainer.compute_advantage
# a one-slot list rather than a bare flag with `global`: this fragment is rendered INSIDE a deferred
# body (see _deferred_patch), where `global` would bind a module-level name that nothing defines and
# raise NameError on the first advantage call. mutating a closed-over container works identically at
# module scope and inside the body.
_flash_pt_logged = [False]


def _flash_pt_rows(non_tensor_batch, batch_size):
    """per-row (spans, turns) or None when this batch carries no usable per-turn metadata."""
    spans_column = non_tensor_batch.get("flash_turn_spans")
    rewards_column = non_tensor_batch.get("flash_turn_rewards")
    if spans_column is None or rewards_column is None:
        return None
    if len(spans_column) != batch_size or len(rewards_column) != batch_size:
        return None
    rows = []
    for spans, turns in zip(spans_column, rewards_column):
        if spans is None or turns is None or len(spans) != len(turns):
            # a row the loop could not align. keep the row so its group can be identified and
            # dropped whole, rather than silently centring the rest against a smaller sample.
            rows.append(None)
            continue
        rows.append(
            (
                tuple((int(start), int(end)) for start, end in spans),
                tuple(float(value) for value in turns),
            )
        )
    return rows


def _flash_pt_per_turn_advantages(rows, index, episode_advantages):
    """centre each turn against the same turn of its group; returns [B, width].

    starts from stock grpo's own output and overwrites only the rows of groups that earned
    per-turn credit. a group that falls back therefore keeps the exact tensor grpo produced rather
    than a reconstruction of it -- there is no scalar to recover, so nothing can drift.
    """
    advantages = episode_advantages.clone()
    groups = {}
    for row_index, uid in enumerate(index):
        groups.setdefault(uid, []).append(row_index)
    for member_indexes in groups.values():
        if any(rows[row_index] is None for row_index in member_indexes):
            continue
        for row_index in member_indexes:
            advantages[row_index] = 0.0
        turn_total = max(len(rows[row_index][1]) for row_index in member_indexes)
        for turn_index in range(turn_total):
            scoring = [
                row_index
                for row_index in member_indexes
                if turn_index < len(rows[row_index][1])
                and rows[row_index][0][turn_index][1] > rows[row_index][0][turn_index][0]
            ]
            if not scoring:
                # every member emitted nothing for this turn; an empty turn carries no signal and
                # must not skew the baseline for the members that did emit one.
                continue
            baseline = sum(rows[row_index][1][turn_index] for row_index in scoring) / len(scoring)
            for row_index in scoring:
                start, end = rows[row_index][0][turn_index]
                advantages[row_index, start:end] = rows[row_index][1][turn_index] - baseline
    return advantages


def _flash_pt_compute_advantage(data, *args, **kwargs):
    data = _flash_pt_original_compute_advantage(data, *args, **kwargs)
    episode = data.batch.get("advantages")
    if episode is None or episode.dim() != 2:
        return data
    batch_size, width = episode.shape
    rows = _flash_pt_rows(data.non_tensor_batch, batch_size)
    if rows is None or all(row is None for row in rows):
        return data
    index = data.non_tensor_batch.get("uid")
    if index is None or len(index) != batch_size:
        return data
    for row in rows:
        if row is None:
            continue
        for start, end in row[0]:
            if not 0 <= start <= end <= width:
                raise ValueError(
                    f"turn span [{start}, {end}) exceeds the response width {width}"
                )
    advantages = _flash_pt_per_turn_advantages(rows, index, episode)
    if not bool(_flash_pt_torch.isfinite(advantages).all()):
        raise ValueError("per-turn advantages must be finite")
    # keep the response mask authoritative: glue tokens sit inside a turn span only when the
    # environment reply was appended mid-turn, and they must never carry gradient.
    response_mask = data.batch.get("response_mask")
    if response_mask is not None:
        advantages = advantages * response_mask.to(dtype=advantages.dtype)
    data.batch["advantages"] = advantages
    # returns feeds the critic, which grpo does not use; stock grpo sets it to the same tensor.
    data.batch["returns"] = advantages
    if not _flash_pt_logged[0]:
        print("[rl-verl] multi-turn per-turn group-relative credit is active", flush=True)
        _flash_pt_logged[0] = True
    return data


_flash_pt_ray_trainer.compute_advantage = _flash_pt_compute_advantage
''',
        "per-turn-credit",
    )


def render_reentrant_checkpointing_shim(reentrant: bool, *, multimodal: bool = False) -> str:
    """return source that makes verl gradient checkpointing reentrant.

    verl hardcodes ``use_reentrant=False`` at ``workers/engine/fsdp/transformer_impl.py:304``;
    MoE and GDN metadata changes make that fail on the first backward. Patch only the kwarg because
    ``transformer_impl.py:433-434`` also uses the enable flag for activation offloading.

    Reentrant LoRA needs ``enable_input_require_grads()`` or frozen embeddings drop all adapter
    gradients (GRAD-001). Multimodal runs also need the vision patch-embed hook in
    ``tests/test_multimodal_input_grads.py``.
    """
    if not reentrant:
        return ""
    # 8 spaces, not 4: this is interpolated INSIDE the `if enable_gradient_checkpointing` body
    # below. at 4 it dedents out of the block and the rendered sitecustomize is a SyntaxError, so
    # every multimodal reentrant run dies before the shim can do anything.
    vision_hook = (
        """
        _flash_install_vision_input_grads(module)"""
        if multimodal
        else ""
    )
    vision_helper = (
        '''

def _flash_install_vision_input_grads(module):
    """mark the vision patch-embed output as requiring grad; see the docstring above."""
    import torch as _flash_vision_torch

    _get_base = getattr(module, "get_base_model", None)
    if callable(_get_base):
        _base = _get_base()
    else:
        _peft_base = getattr(module, "base_model", None)
        _base = getattr(_peft_base, "model", module)

    def _flash_require_output_grad(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) and output else output
        if isinstance(tensor, _flash_vision_torch.Tensor) and tensor.is_floating_point():
            tensor.requires_grad_(True)

    for _path, _submodule in _base.named_modules():
        if _path.endswith("visual.patch_embed"):
            _submodule.register_forward_hook(_flash_require_output_grad)
            print(f"[rl-verl] vision input gradients enabled at {_path}", flush=True)
            return
    print("[rl-verl] no visual.patch_embed found; vision input gradients not installed", flush=True)
'''
        if multimodal
        else ""
    )
    return _deferred_patch(
        "reentrant",
        "verl.workers.engine.fsdp.transformer_impl",
        f"""
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine as _FlashReentrantEngine

_flash_reentrant_original_build_module = _FlashReentrantEngine._build_module
{vision_helper}

def _flash_reentrant_build_module(self):
    module = _flash_reentrant_original_build_module(self)
    # only when verl actually enabled checkpointing: calling gradient_checkpointing_enable on a
    # model verl deliberately left uncheckpointed would turn it ON and change the memory profile.
    if getattr(self.model_config, "enable_gradient_checkpointing", False):
        # the LANGUAGE-side counterpart of the vision hook below, and required for the same
        # reason (GRAD-001): lora freezes the embeddings, so nothing entering the first
        # checkpointed decoder layer requires grad and reentrant recompute drops the backward
        # for the whole segment -- where every lora parameter lives. the vision hook only ever
        # covered the patch embeddings on multimodal runs; text-only runs had no hook at all and
        # trained nothing while reporting success.
        module.enable_input_require_grads()
        module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={{"use_reentrant": True}}){vision_hook}
        print("[rl-verl] reentrant gradient checkpointing is active", flush=True)
    return module


_FlashReentrantEngine._build_module = _flash_reentrant_build_module
""",
        "reentrant-checkpointing",
    )


def render_entropy_quantile_shim(entropy_quantile: float | None) -> str:
    """return source that adds top-entropy token masking to verl.

    Patch ``ppo_loss`` because registered policy losses cannot see per-token entropy. Mask only the
    policy-gradient term; KL and entropy bonus keep the full response mask. Flash pins
    ``seq-mean-token-sum-norm`` so removing numerator tokens does not rescale the rest.
    """
    if entropy_quantile is None or float(entropy_quantile) >= 1.0:
        return ""
    threshold = 1.0 - float(entropy_quantile)
    return _deferred_patch(
        "entropyquantile",
        "verl.workers.utils.losses",
        f'''
import threading as _flash_threading

import torch as _flash_torch
import torch.distributed as _flash_dist
from verl.workers.utils import losses as _flash_losses
from verl.workers.utils.padding import no_padding_2_padding as _flash_no_padding_2_padding

_flash_entropy_threshold_q = {threshold!r}
_flash_original_ppo_loss = _flash_losses.ppo_loss
_flash_original_get_policy_loss_fn = _flash_losses.get_policy_loss_fn
# carries this micro-batch's padded entropy from ppo_loss down to the policy loss, which verl
# never passes it. thread-local rather than a module global so the handoff cannot leak across
# concurrent loss calls, and always cleared in a finally.
_flash_entropy_state = _flash_threading.local()


def _flash_high_entropy_mask(entropy, response_mask):
    """keep tokens at or above the global entropy quantile."""
    local = entropy[response_mask.bool()].float().reshape(-1)
    # the quantile is over the whole global batch, not the local shard, so every rank thresholds
    # identically. verl's actor shards over the default process group, so the quantile must be global.
    if _flash_dist.is_available() and _flash_dist.is_initialized() and _flash_dist.get_world_size() > 1:
        sizes = [_flash_torch.zeros(1, dtype=_flash_torch.long, device=local.device)
                 for _ in range(_flash_dist.get_world_size())]
        _flash_dist.all_gather(sizes, _flash_torch.tensor([local.numel()], dtype=_flash_torch.long, device=local.device))
        largest = int(max(int(s.item()) for s in sizes))
        if largest == 0:
            return _flash_torch.zeros_like(entropy, dtype=_flash_torch.bool)
        # pad with a negative sentinel: entropy is non-negative, so it can never collide with a
        # real value and is dropped after the gather.
        padded = _flash_torch.full((largest,), -1e9, dtype=_flash_torch.float32, device=local.device)
        padded[: local.numel()] = local
        buckets = [_flash_torch.empty_like(padded) for _ in range(_flash_dist.get_world_size())]
        _flash_dist.all_gather(buckets, padded)
        gathered = _flash_torch.cat(buckets)
        gathered = gathered[gathered != -1e9]
    else:
        gathered = local
    if gathered.numel() == 0:
        return _flash_torch.zeros_like(entropy, dtype=_flash_torch.bool)
    cutoff = _flash_torch.quantile(gathered, _flash_entropy_threshold_q)
    return ((entropy * response_mask.float()) >= cutoff) & response_mask.bool()


def _flash_masked_policy_loss_fn(loss_mode):
    """wrap the resolved policy loss so it sees the entropy-masked response mask.

    masking here rather than in ppo_loss keeps the mask on the policy-gradient term only: ppo_loss
    aggregates the kl and entropy-bonus terms against its own unmasked response_mask, exactly as
    the kl term is added after multiplying per_token_loss by the entropy mask.
    """
    inner = _flash_original_get_policy_loss_fn(loss_mode)

    def _masked(*args, **kwargs):
        entropy = getattr(_flash_entropy_state, "entropy", None)
        response_mask = kwargs.get("response_mask", None)
        if entropy is not None and response_mask is not None:
            kwargs["response_mask"] = _flash_high_entropy_mask(entropy, response_mask)
        return inner(*args, **kwargs)

    return _masked


def _flash_entropy_masked_ppo_loss(config, model_output, data, dp_group=None):
    entropy = model_output.get("entropy", None)
    if entropy is None:
        raise RuntimeError(
            "train.entropy_quantile needs per-token entropy, but verl produced none; "
            "actor.calculate_entropy must be true."
        )
    # ppo_loss re-derives this internally; convert here too so the mask is padded to the same
    # (bsz, response_len) shape the policy loss receives its response_mask in.
    _flash_entropy_state.entropy = _flash_no_padding_2_padding(entropy, data)
    try:
        return _flash_original_ppo_loss(config, model_output, data, dp_group)
    finally:
        _flash_entropy_state.entropy = None


# patch once. python imports sitecustomize a single time per interpreter, so this should not come
# up -- but wrapping an already-wrapped loss would mask the top quantile OF the top quantile and
# train on a fraction of the requested tokens, with nothing in the logs to show for it.
if not getattr(_flash_original_ppo_loss, "_flash_entropy_masked", False):
    _flash_entropy_masked_ppo_loss._flash_entropy_masked = True
    _flash_losses.get_policy_loss_fn = _flash_masked_policy_loss_fn
    _flash_losses.ppo_loss = _flash_entropy_masked_ppo_loss
    print({_ENTROPY_QUANTILE_MARKER!r} + " quantile={entropy_quantile:g}", flush=True)
''',
        "entropy-quantile",
    )


def render_reward_module(url_env: str = "FLASH_VERL_REWARD_URL") -> str:
    """source for the verl custom reward module.

    runs INSIDE the verl interpreter, so it must be self-contained (stdlib only, no flash import).
    it forwards (index, solution_str) to the flash reward bridge and returns the float score.
    """
    return (
        '"""flash reward bridge shim (generated). posts each completion to the flash worker."""\n'
        "import json\n"
        "import os\n"
        "import urllib.error\n"
        "import urllib.request\n"
        "\n"
        f"_URL = os.environ.get({url_env!r}, '')\n"
        "\n"
        "\n"
        "def compute_score(data_source, solution_str, ground_truth, extra_info=None):\n"
        "    idx = (extra_info or {}).get('index')\n"
        "    if idx is None:\n"
        "        raise RuntimeError('flash reward bridge received no example index')\n"
        "    if not _URL:\n"
        "        raise RuntimeError('flash reward bridge url is not configured')\n"
        "    if isinstance(idx, bool) or getattr(getattr(idx, 'dtype', None), 'kind', None) == 'b':\n"
        "        raise RuntimeError('flash reward bridge received an invalid example index: %r' % idx)\n"
        "    try:\n"
        "        exact_idx = int(idx)\n"
        "    except (TypeError, ValueError, OverflowError) as exc:\n"
        "        raise RuntimeError('flash reward bridge received an invalid example index: %r' % idx) from exc\n"
        "    if exact_idx != idx:\n"
        "        raise RuntimeError('flash reward bridge received an invalid example index: %r' % idx)\n"
        "    idx = exact_idx\n"
        "    body = json.dumps({'index': idx, 'solution_str': solution_str or ''}).encode()\n"
        "    req = urllib.request.Request(\n"
        "        _URL.rstrip('/') + '/score', data=body, headers={'Content-Type': 'application/json'}\n"
        "    )\n"
        "    try:\n"
        # NO client deadline. verl fans this call out hard: RewardLoopManager builds
        # reward.num_workers (8) ray workers unconditionally on the grpo path
        # (ray_trainer.py:901-910), and each one asyncio.gathers every row in its chunk
        # (reward_loop.py:138-143). start_reward_server coalesces those requests behind one scoring
        # thread, so a per-request timeout would still bound QUEUE WAIT as well as the env call -- a
        # caller can fail for arriving behind a slow-but-healthy judge batch.
        # a wedged env is caught by the training stall watchdog instead (STALL_AFTER_S=1500s in
        # providers/_poll.py), which measures training progress rather than one request.
        "        with urllib.request.urlopen(req) as r:\n"
        "            payload = json.loads(r.read().decode())\n"
        "            return float(payload['score'])\n"
        "    except urllib.error.URLError as exc:\n"
        "        raise RuntimeError('flash reward bridge request failed: %s' % exc) from exc\n"
        "    except Exception as exc:\n"
        "        raise RuntimeError('flash reward bridge returned an invalid response: %s' % exc) from exc\n"
    )
