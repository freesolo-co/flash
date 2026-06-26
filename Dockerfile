# Flash control plane (operator-side).
#
#   docker build -t flash-control-plane .
#   docker run -p 8080:8080 \
#     -e RUNPOD_API_KEY=... -e HF_TOKEN=... \
#     -v flash-state:/root/.flash flash-control-plane
#
# All persistent state (key DB, run records, results) lives under ~/.flash (fixed paths,
# = /root/.flash for the default root user) — mount a volume there. Run exactly ONE
# container instance per state volume (state is local files + SQLite; no horizontal scaling).

FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git curl \
    && curl -1sLf 'https://artifacts-cli.infisical.com/setup.deb.sh' | bash \
    && apt-get update && apt-get install -y --no-install-recommends infisical \
    && rm -rf /var/lib/apt/lists/* \
    && chmod +x /app/infisical-entrypoint.sh
RUN pip install --no-cache-dir ".[server]"

VOLUME /root/.flash
EXPOSE 8080

# Use the per-arch baked worker images (ghcr.io/.../flash-worker:cu128-<sm>) so cold workers skip the
# ~10-15 min first-use JIT; the allocator maps each GPU class to its matching -smXX tag. All validated
# SMs (sm80/86/89/90/120) are published. NOTE: rebakes are MANUAL -- after a Dockerfile.worker/deps
# change rebuilds :cu128, re-run bake-kernel-cache.yml so the -smXX tags don't ship stale deps (the
# worker-image build posts a reminder). Override at runtime with `-e FLASH_WORKER_IMAGE_PER_SM=0`.
ENV FLASH_WORKER_IMAGE_PER_SM=1

# secret injection wrapper: no-op passthrough unless INFISICAL_CLIENT_ID is set, else
# `infisical login` (universal-auth) then `infisical run --path /flash` before the server.
ENTRYPOINT ["/app/infisical-entrypoint.sh"]
CMD ["python", "-m", "flash.server", "--host", "0.0.0.0", "--port", "8080"]
