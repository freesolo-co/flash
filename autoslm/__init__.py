"""AutoSLM — managed LoRA post-training: claim a key, train.

A focused developer experience (TOML run specs, pluggable environments,
CLI/API/MCP entry points, adapter deployment). Users authenticate with a single
AutoSLM key (`slm login`); the control plane runs each job on a dedicated managed
consumer GPU (RTX 4090 / RTX 5090) behind the scenes.
"""

__all__ = ["__version__"]

__version__ = "0.2.0"
