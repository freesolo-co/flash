"""Authoritative environment index in Azure PostgreSQL.

Maps a managed environment slug (``namespace/name``) to its package blob (container + key) and
content hash. The control plane writes a row on publish and reads it to resolve a run's env to a
blob (then mints a SAS URL — see ``flash/server/azure_blob``). This is the source of truth for
resolution; the Supabase ``flash_environments`` table is only a best-effort UI mirror.

Config (control-plane env):
  * ``FLASH_ENV_PG_URL`` — PostgreSQL connection URL/DSN for the Azure Flexible Server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

PG_URL_ENV = "FLASH_ENV_PG_URL"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS flash_environments (
    slug            text PRIMARY KEY,
    namespace       text NOT NULL,
    name            text NOT NULL,
    blob_container  text NOT NULL,
    blob_key        text NOT NULL,
    package_sha256  text NOT NULL,
    size_bytes      bigint NOT NULL,
    version         integer NOT NULL DEFAULT 1,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);
"""


class EnvironmentStoreNotConfigured(RuntimeError):
    """The control plane has no Azure Postgres URL configured."""


@dataclass(frozen=True)
class EnvironmentRecord:
    slug: str
    namespace: str
    name: str
    blob_container: str
    blob_key: str
    package_sha256: str
    size_bytes: int
    version: int


def _dsn() -> str:
    dsn = (os.environ.get(PG_URL_ENV) or "").strip()
    if not dsn:
        raise EnvironmentStoreNotConfigured(
            f"{PG_URL_ENV} is not set; the control plane cannot index environment packages"
        )
    return dsn


def _connect():  # type: ignore[no-untyped-def]
    # Lazy import: only the control plane installs psycopg.
    import psycopg

    return psycopg.connect(_dsn())


def ensure_schema() -> None:
    """Create the index table if it does not exist (idempotent)."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(_SCHEMA)


def upsert(
    *,
    slug: str,
    namespace: str,
    name: str,
    blob_container: str,
    blob_key: str,
    package_sha256: str,
    size_bytes: int,
) -> None:
    """Insert or update the pointer for ``slug`` (bumps ``version`` on re-publish)."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO flash_environments
                (slug, namespace, name, blob_container, blob_key, package_sha256, size_bytes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (slug) DO UPDATE SET
                namespace      = EXCLUDED.namespace,
                name           = EXCLUDED.name,
                blob_container = EXCLUDED.blob_container,
                blob_key       = EXCLUDED.blob_key,
                package_sha256 = EXCLUDED.package_sha256,
                size_bytes     = EXCLUDED.size_bytes,
                version        = flash_environments.version + 1,
                updated_at     = now()
            """,
            (slug, namespace, name, blob_container, blob_key, package_sha256, size_bytes),
        )


def lookup(slug: str) -> EnvironmentRecord | None:
    """Return the pointer record for ``slug``, or ``None`` if it isn't indexed."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT slug, namespace, name, blob_container, blob_key,
                   package_sha256, size_bytes, version
            FROM flash_environments
            WHERE slug = %s
            """,
            (slug,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return EnvironmentRecord(
        slug=row[0],
        namespace=row[1],
        name=row[2],
        blob_container=row[3],
        blob_key=row[4],
        package_sha256=row[5],
        size_bytes=int(row[6]),
        version=int(row[7]),
    )
