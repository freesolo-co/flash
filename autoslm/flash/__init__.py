"""Compatibility shim: RunPod Flash code moved to ``autoslm.providers.runpod``.

The RunPod substrate now lives under the symmetric provider layout
(``autoslm.providers.runpod.{api,auth,pricing,gpus,durable,train,preflight}``). New code
should import from there (or the provider registry / ``autoslm.providers.base``).

This package keeps ``autoslm.flash.*`` working by ALIASING the old submodule names to
the SAME module objects the providers package uses (via ``sys.modules``). Aliasing the
object — rather than re-exporting names into a fresh module — is what keeps existing
test ``monkeypatch.setattr(autoslm.flash.train, ...)`` patches effective: they mutate
the very module the orchestrator imports from. ``gpus``/``preflight``/``auth`` keep
dedicated shim files (their surface is split/aggregated across the new layout).
"""

from __future__ import annotations

import sys

from autoslm.providers.runpod import api as _api
from autoslm.providers.runpod import auth as _auth
from autoslm.providers.runpod import durable as _durable
from autoslm.providers.runpod import pricing as _pricing
from autoslm.providers.runpod import train as _train

# Bind the historical submodule names to the real provider module OBJECTS so both
# ``from autoslm.flash import runpod_api`` and a monkeypatch against the alias land on
# the module the new code actually uses.
#
# Two registrations are needed for BOTH import spellings to work:
#   * ``from autoslm.flash import train`` resolves via getattr on THIS package, so the
#     submodule must be an ATTRIBUTE of the package object.
#   * ``import autoslm.flash.train`` resolves via ``sys.modules["autoslm.flash.train"]``;
#     since there is no real ``train.py`` file for the finder to load, the entry must be
#     pre-seeded in ``sys.modules`` or the import machinery raises ModuleNotFoundError.
# Setting only one of the two leaves the other spelling broken, so we set both.
_ALIASES = {
    "runpod_api": _api,
    "auth": _auth,
    "train": _train,
    "durable": _durable,
    "pricing": _pricing,
}
for _name, _mod in _ALIASES.items():
    globals()[_name] = _mod  # attribute of the package -> `from autoslm.flash import X`
    sys.modules.setdefault(f"autoslm.flash.{_name}", _mod)  # `import autoslm.flash.X`
