# Flash control plane (operator-side).
#
#   docker build -t flash-control-plane .
#   docker run -p 8080:8080 \
#     -e RUNPOD_API_KEY=... -e HF_TOKEN=... \
#     -v flash-state:/root/.flash flash-control-plane
#
# All persistent state (key DB, run records, results) lives under ~/.flash (= /root/.flash for
# the default root user) — mount a volume there. Run exactly ONE container instance per state
# volume (state is local files + SQLite; no horizontal scaling).
#
# FLASH_DATA_DIR moves that root. The VOLUME below names the DEFAULT path, and Docker cannot
# declare a volume at a path chosen at runtime, so a container that sets FLASH_DATA_DIR must
# also mount its own volume at the new location — otherwise state lands on the container's
# writable layer and disappears when the container is replaced.

FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir ".[server]"

VOLUME /root/.flash
EXPOSE 8080

# The allocator uses the per-arch baked worker images (ghcr.io/.../flash-worker:cu128-<sm>) so cold
# workers skip the ~10-15 min first-use JIT; each GPU class maps to its matching -smXX tag. All
# validated SMs (sm80/86/89/90/120) are published; unbaked arches fall back to the base :cu128 image.
# Rebakes are MANUAL -- after a Dockerfile.worker/deps change rebuilds :cu128, re-run
# bake-kernel-cache.yml so the -smXX tags don't ship stale deps (the worker-image build posts a
# reminder).

# Secrets come from the container environment (-e, --env-file, or your orchestrator's secret
# store), so this image needs no secret-manager tooling baked in. Pulling secrets from a manager
# at container start is an opt-in overlay that wraps this image's entrypoint and inherits the CMD
# below -- see deploy/infisical/ for a working example.
CMD ["python", "-m", "flash.server", "--host", "0.0.0.0", "--port", "8080"]
