"""hosted checkpoint columns shared by runtime persistence and schema gates."""

from __future__ import annotations

PERSISTED_COLUMNS = (
    "id,org_id,run_id,checkpoint,checkpoint_id,source_repo_type,source_repository,"
    "source_revision,source_subfolder,artifact_digest,artifact_fingerprint,base_model,"
    "lora_config,serving_defaults,url,status,deployment_generation,created_at,updated_at"
)
