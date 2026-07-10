"""Resolved OPD knobs from the JobSpec's [train] table (falling back to RECIPE.opd).

Extracted from ``opd`` so the training loop imports a typed config object instead of carrying the
resolution logic inline. ``opd`` re-imports ``OpdKnobs`` and ``_resolve_opd_knobs`` so they stay
importable from it.
"""

from __future__ import annotations

from dataclasses import dataclass

from flash.engine.recipe import RECIPE, resolve_teacher
from flash.engine.vram import opd_completion_len
from flash.engine.worker._pkg import W as _w


@dataclass(frozen=True)
class OpdKnobs:
    """Resolved opd knobs from the JobSpec's [train] table (falling back to RECIPE.opd), returned by
    ``_resolve_opd_knobs``. A typed container replacing the old stringly-typed dict; field names match
    the former dict keys one-for-one. The defaults are placeholders only — ``_resolve_opd_knobs``
    always sets every field explicitly — kept so partial construction stays ergonomic for tests."""

    teacher_model: str = ""
    teacher_base_url: str = ""
    epochs: int = RECIPE.opd.num_epochs
    learning_rate: float = 0.0
    temperature: float = 0.0
    top_p: float = 1.0
    max_completion: int = 0
    prompts_per_step: int = 0
    group_size: int = 0
    # gkd reverse-KL scale; reuses the existing [train] kl_penalty_coef knob (default 1.0).
    kl_coef: float = 1.0
    # Weight of the terminal-EOS behaviour-cloning term (see _eos_reinforce_term). The reverse-KL
    # over shared decoded-text spans cannot supervise the zero-width stop token, so this restores the
    # stop signal; 0 disables it. [train].opd_eos_loss_coef, else RECIPE.opd.eos_loss_coef.
    eos_loss_coef: float = RECIPE.opd.eos_loss_coef
    save_every: int = 0
    max_length: int = 0
    # Student on-policy sampling stops at these delimiters (parity with GRPO), so the teacher never
    # scores/trains on text past the intended answer boundary.
    stop_sequences: tuple = ()
    # Canonical StructuredOutputsParams kwargs JSON from [train] structured_outputs ("" = off);
    # constrains every student rollout (parity with GRPO). Teacher echo-scoring needs no schema —
    # it scores the student's already-constrained tokens.
    structured_outputs: str = ""


def _resolve_opd_knobs() -> OpdKnobs:
    """Resolve every opd knob from the JobSpec's [train] table, falling back to RECIPE.opd."""
    d = RECIPE.opd
    t = _w.JOB_SPEC.train if _w.JOB_SPEC else None

    def opt(name, default):
        v = getattr(t, name, None) if t else None
        return v if v is not None else default

    # kl_penalty_coef IS the gkd distillation objective's scale: the OPD loss multiplies every span
    # coefficient by it, so kl_coef=0 makes EVERY backward a zero gradient while the loop still counts
    # opt_steps and would publish/charge a fully-untrained adapter. The shared schema allows 0 for GRPO
    # (a valid "no KL penalty"), but for OPD it must be positive, so reject an explicit 0 here (a plain
    # `or` would also wrongly treat 0.0 as unset). Omitting the field uses the positive recipe default.
    _kl = opt("kl_penalty_coef", None)
    kl_coef = float(_kl if _kl is not None else d.kl_coef)
    if kl_coef == 0.0:
        raise RuntimeError(
            "opd: [train] kl_penalty_coef must be > 0 — it scales the gkd distillation objective, so "
            "0 makes every optimizer step a no-op (zero gradient) yet still counts toward `steps` and "
            "publishes an untrained adapter. Omit the field to use the default, or set a positive value."
        )
    # Resolve the managed teacher from [train].teacher_model (the resolved Fireworks model id, "" =>
    # the GLM 5.2 default). Parse already validated + canonicalized it, but JobSpec.from_dict is a
    # tolerant deserializer, so re-validate at this boundary (like the kl_coef guard above): resolve
    # is idempotent for a canonical model id, and a spec that reaches the worker with an unsupported
    # teacher fails loudly here rather than as an opaque Fireworks 404 mid-run. base_url is shared by
    # every allow-listed teacher (one Fireworks endpoint + one managed key), so it stays d.teacher_base_url.
    try:
        teacher = resolve_teacher(opt("teacher_model", ""))
    except ValueError as e:
        raise RuntimeError(f"opd: {e}") from e
    return OpdKnobs(
        teacher_model=teacher.model_id,
        teacher_base_url=d.teacher_base_url,
        epochs=int(t.epochs) if t and t.epochs is not None else d.num_epochs,
        learning_rate=float(opt("learning_rate", 0) or d.learning_rate),
        temperature=float(
            opt("temperature", None)
            if (t and t.temperature is not None)
            else d.sampling_temperature
        ),
        top_p=d.sampling_top_p,
        max_completion=opd_completion_len(opt("max_completion_tokens", 0), _w.THINKING),
        prompts_per_step=int(opt("batch_size", 0) or d.prompts_per_step),
        group_size=int(opt("group_size", 0) or d.group_size),
        kl_coef=kl_coef,
        # 0 is a VALID explicit value here (disable EOS reinforcement), unlike kl_coef, so only fall
        # back to the recipe default when the field is UNSET (None) — a plain `or` would swallow 0.0.
        eos_loss_coef=max(
            0.0,
            float(
                _eos if (_eos := opt("opd_eos_loss_coef", None)) is not None else d.eos_loss_coef
            ),
        ),
        save_every=int(opt("save_every", 0) or 20),
        max_length=int(opt("max_context_tokens", 0) or 0),
        stop_sequences=tuple(getattr(t, "stop_sequences", ()) or ()),
        structured_outputs=str(getattr(t, "structured_outputs", "") or ""),
    )
