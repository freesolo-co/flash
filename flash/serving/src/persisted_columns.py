"""The adapter columns the serving runtime reads, kept importable without the runtime's deps.

`persistence.py` imports httpx at module scope, and the modal deploy job installs only modal and
python-dotenv -- so the production schema gate cannot import that module to learn which columns to
probe. This module has no imports at all, so both the runtime and the gate read one definition
instead of restating it and drifting apart.
"""

from __future__ import annotations

PERSISTED_COLUMNS = (
    "adapter_id,repo_id,org_id,url,base_model,subfolder,repo_type,checkpoint,private,"
    "status,metadata,created_at,updated_at,deployment_generation"
)
