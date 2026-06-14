"""MATH (\\boxed{} LaTeX/numeric) example environment (repo-root demo, path-loaded).

Loaded by ``autoslm.envs.registry`` as the ``math`` built-in via importlib, so all
MATH-specific code (env + grader) lives here rather than in the core ``autoslm``
package.
"""

from .env import MathEnvironment, load_environment

__all__ = ["MathEnvironment", "load_environment"]
