"""Flash — managed LoRA post-training: log in with your freesolo key, train.

A focused developer experience (TOML run specs, pluggable environments,
CLI/API/MCP entry points, adapter deployment). Users authenticate with their
freesolo API key (`flash login`); the control plane runs each job on a managed
GPU (RunPod or Vast.ai) behind the scenes.
"""

__all__ = ["__version__"]

__version__ = "0.2.17"
