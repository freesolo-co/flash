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

# Optional Infisical CLI, for deployments that pull secrets from a vault at container start
# instead of receiving them as environment variables. OFF by default: a plain `docker build .`
# produces a vendor-free image, which is what an open-source consumer gets. Freesolo's own
# images turn it on (.github/workflows/publish-image.yml passes INSTALL_INFISICAL=true).
#
# Installing the CLI does not by itself change how the container behaves -- entrypoint.sh is a
# passthrough until INFISICAL_CLIENT_ID is set, so an image built with the CLI still runs on a
# plain `--env-file` exactly as one built without it. Presence of the binary and use of the
# binary are deliberately two separate switches.
#
# Pinned by version AND sha256 against the vendor's signed release artifact rather than piped
# from a setup script: `curl ... | bash` executes whatever the vendor serves at request time,
# which is neither reviewable nor reproducible, and a repository that ships it is teaching the
# pattern to everyone who reads its Dockerfile. Bump INFISICAL_VERSION and both digests
# together -- a stale digest fails the build loudly instead of installing an unverified binary.
#
# Digests are the vendor's published checksums.txt entries for this version:
#   https://github.com/Infisical/cli/releases/download/v${INFISICAL_VERSION}/checksums.txt
ARG INSTALL_INFISICAL=false
ARG INFISICAL_VERSION=0.43.121
ARG INFISICAL_SHA256_AMD64=31716b899abe6fed8dfa77daa9af088f7f5a52b6ee45cbaf0ef9c326df31c110
ARG INFISICAL_SHA256_ARM64=fb82ef4838a4a743ff7a311056e772dcffc3a8a37f20ef01e92822152edfa844
RUN if [ "${INSTALL_INFISICAL}" = "true" ]; then \
        set -eu; \
        # dpkg's architecture, not uname's: the .deb asset names use Debian arch strings.
        arch="$(dpkg --print-architecture)"; \
        case "${arch}" in \
            amd64) sha="${INFISICAL_SHA256_AMD64}" ;; \
            arm64) sha="${INFISICAL_SHA256_ARM64}" ;; \
            *) echo "no pinned infisical build for dpkg arch: ${arch}" >&2; exit 1 ;; \
        esac; \
        deb="/tmp/infisical.deb"; \
        curl -fsSL -o "${deb}" \
            "https://github.com/Infisical/cli/releases/download/v${INFISICAL_VERSION}/infisical_${INFISICAL_VERSION}_linux_${arch}.deb"; \
        # Verify BEFORE dpkg touches it: a mismatch must fail the build, never install.
        echo "${sha}  ${deb}" | sha256sum -c -; \
        apt-get update; \
        apt-get install -y --no-install-recommends "${deb}"; \
        rm -f "${deb}"; \
        rm -rf /var/lib/apt/lists/*; \
        infisical --version; \
    fi

VOLUME /root/.flash
EXPOSE 8080

# The allocator uses the per-arch baked worker images (ghcr.io/.../flash-worker:cu128-<sm>) so cold
# workers skip the ~10-15 min first-use JIT; each GPU class maps to its matching -smXX tag. All
# validated SMs (sm80/86/89/90/120) are published; unbaked arches fall back to the base :cu128 image.
# Rebakes are MANUAL -- after a Dockerfile.worker/deps change rebuilds :cu128, re-run
# bake-kernel-cache.yml so the -smXX tags don't ship stale deps (the worker-image build posts a
# reminder).

# Secrets reach Flash as ordinary environment variables. Both supported ways of producing that
# environment run through this one entrypoint:
#
#   docker run --env-file .env ...        -> INFISICAL_CLIENT_ID unset, wrapper execs CMD directly
#   docker run -e INFISICAL_CLIENT_ID=... -> wrapper logs in and injects, then execs CMD
#
# The switch is the environment, not the image: the same image does both. With the variable
# unset the wrapper is indistinguishable from having no entrypoint at all, so an image built
# with INSTALL_INFISICAL=true still runs on a plain --env-file with no vault involved.
COPY deploy/infisical/entrypoint.sh /usr/local/bin/flash-infisical-entrypoint
RUN chmod +x /usr/local/bin/flash-infisical-entrypoint
ENTRYPOINT ["/usr/local/bin/flash-infisical-entrypoint"]

# Docker resets a derived image's inherited CMD to null when that image declares its own
# ENTRYPOINT, so any overlay built FROM this one must restate this line verbatim.
CMD ["python", "-m", "flash.server", "--host", "0.0.0.0", "--port", "8080"]
