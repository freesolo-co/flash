"""GSM8K verifiable-math example environment (repo-root demo, path-loaded).

Loaded by ``autoslm.envs.registry`` as the ``gsm8k`` built-in via importlib, so
all GSM8K-specific code (env, grader, data wiring) lives here rather than in the
core ``autoslm`` package.
"""

from .env import GSM8KEnvironment, load_environment

__all__ = ["GSM8KEnvironment", "load_environment"]
