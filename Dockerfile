# Flash control plane (operator-side).
#
#   docker build -t flash-control-plane .
#   docker run -p 8080:8080 \
#     -e MODAL_TOKEN_ID=... -e MODAL_TOKEN_SECRET=... -e HF_TOKEN=... \
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

# digest-pinned so control-plane rebuilds reuse the exact reviewed base image bytes.
FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git curl \
    && rm -rf /var/lib/apt/lists/*

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

# Optional Nebius CLI, required only by the Nebius Spot provider (`[gpu] spot = true`): Flash
# drives Nebius through `nebius ... --no-browser` and treats the provider as unconfigured when the
# binary is absent (flash/providers/nebius/config.py). OFF by default, like Infisical, so a plain
# `docker build .` stays vendor-free; Freesolo's published images turn it on
# (.github/workflows/publish-image.yml passes INSTALL_NEBIUS=true). The binary alone grants
# nothing: the operator still mounts an authenticated service-account profile at /root/.nebius
# and sets FLASH_NEBIUS_* (see SELF_HOSTING.md).
#
# Pinned by version AND sha256 rather than piping the vendor's install.sh into a shell. Nebius
# publishes the binaries at artifacts.nebius.cloud/cli/release/<version>/linux/<arch>/nebius
# without a checksums file, so the digests below were taken from those exact release binaries.
# Bump NEBIUS_CLI_VERSION and both digests together; a mismatch fails the build.
ARG INSTALL_NEBIUS=false
ARG NEBIUS_CLI_VERSION=0.12.302
ARG NEBIUS_CLI_SHA256_AMD64=44315ff3fec500bcf3e7836f738ab456dcdf2b796c79b531ded6ca8f40622f56
ARG NEBIUS_CLI_SHA256_ARM64=0d4781905d777b3a4fc295158d0f382691e453cc7828c02dca2212cf331712b9
RUN if [ "${INSTALL_NEBIUS}" = "true" ]; then \
        set -eu; \
        case "$(dpkg --print-architecture)" in \
            amd64) arch="x86_64"; sha="${NEBIUS_CLI_SHA256_AMD64}" ;; \
            arm64) arch="arm64"; sha="${NEBIUS_CLI_SHA256_ARM64}" ;; \
            *) echo "no pinned nebius CLI build for this architecture" >&2; exit 1 ;; \
        esac; \
        bin="/tmp/nebius"; \
        curl -fsSL -o "${bin}" \
            "https://artifacts.nebius.cloud/cli/release/${NEBIUS_CLI_VERSION}/linux/${arch}/nebius"; \
        # Verify BEFORE installing: a mismatch must fail the build, never install.
        echo "${sha}  ${bin}" | sha256sum -c -; \
        install -m 0755 "${bin}" /usr/local/bin/nebius; \
        rm -f "${bin}"; \
        test "$(nebius version)" = "${NEBIUS_CLI_VERSION}"; \
    fi

# Source changes reuse the operating-system and optional Infisical and Nebius layers.
COPY . .
RUN pip install --no-cache-dir ".[server]"

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
# ENTRYPOINT, so any image built FROM this one must restate this line verbatim. Without it the
# entrypoint receives no arguments and stops with "no command given".
CMD ["python", "-m", "flash.server", "--host", "0.0.0.0", "--port", "8080"]
