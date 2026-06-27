"""Eligibility gate: is a paper replicable on Flash, and on which model?"""

from __future__ import annotations

from autoenv.gate.eligibility import gate_case
from autoenv.gate.model_match import ModelMatch, resolve_flash_model

__all__ = ["ModelMatch", "gate_case", "resolve_flash_model"]
