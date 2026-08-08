# Infisical CLI overlay for an ALREADY-BUILT Flash control-plane image.
#
# You probably do not need this file. The base image supports Infisical directly:
#
#   docker build --build-arg INSTALL_INFISICAL=true -t flash-control-plane .
#
# ...and that is how Freesolo's published images are built. Either way the switch at runtime is
# the environment (INFISICAL_CLIENT_ID), not the image.
#
# This overlay exists for the one case the build argument cannot cover: adding the CLI to an
# image you have already been given and cannot rebuild -- a published tag, or an artifact from
# a pipeline you do not control. It re-runs only the install step, on top of that image.
#
#   docker build -f deploy/infisical/Dockerfile \
#     --build-arg FLASH_IMAGE=ghcr.io/freesolo-co/freesolo-flash:main \
#     -t flash-control-plane-infisical .
#
# The build context is the repository root.

ARG FLASH_IMAGE=ghcr.io/freesolo-co/freesolo-flash:main
FROM ${FLASH_IMAGE}

# Same pinned artifact and digests as the base Dockerfile -- keep them in step when bumping.
# Verified before install: a mismatch fails the build rather than installing an unverified
# binary. Deliberately NOT `curl ... | bash`: a setup script executes whatever the vendor
# serves at request time, which is neither reviewable nor reproducible.
ARG INFISICAL_VERSION=0.43.121
ARG INFISICAL_SHA256_AMD64=31716b899abe6fed8dfa77daa9af088f7f5a52b6ee45cbaf0ef9c326df31c110
ARG INFISICAL_SHA256_ARM64=fb82ef4838a4a743ff7a311056e772dcffc3a8a37f20ef01e92822152edfa844
RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl; \
    arch="$(dpkg --print-architecture)"; \
    case "${arch}" in \
        amd64) sha="${INFISICAL_SHA256_AMD64}" ;; \
        arm64) sha="${INFISICAL_SHA256_ARM64}" ;; \
        *) echo "no pinned infisical build for dpkg arch: ${arch}" >&2; exit 1 ;; \
    esac; \
    deb="/tmp/infisical.deb"; \
    curl -fsSL -o "${deb}" \
        "https://github.com/Infisical/cli/releases/download/v${INFISICAL_VERSION}/infisical_${INFISICAL_VERSION}_linux_${arch}.deb"; \
    echo "${sha}  ${deb}" | sha256sum -c -; \
    apt-get install -y --no-install-recommends "${deb}"; \
    rm -f "${deb}"; \
    rm -rf /var/lib/apt/lists/*; \
    infisical --version

# The base image already installs this same script at this same path and declares it as its
# ENTRYPOINT. Both are restated because FLASH_IMAGE may point at an older image built before
# the base carried either, and re-copying an identical file is cheap.
COPY deploy/infisical/entrypoint.sh /usr/local/bin/flash-infisical-entrypoint
RUN chmod +x /usr/local/bin/flash-infisical-entrypoint
ENTRYPOINT ["/usr/local/bin/flash-infisical-entrypoint"]

# CMD must be restated, and must stay in step with the base Dockerfile's CMD. Setting
# ENTRYPOINT in a DERIVED image resets the inherited CMD to null, so without this line the
# wrapper is handed no command at all and the container never starts the server. Do not
# "deduplicate" this by deleting it.
CMD ["python", "-m", "flash.server", "--host", "0.0.0.0", "--port", "8080"]
