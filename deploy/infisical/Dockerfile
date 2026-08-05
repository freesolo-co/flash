# OPT-IN Infisical overlay for the Flash control plane.
#
# Flash itself needs nothing from Infisical: it reads secrets from its process environment,
# so the base image already works with `docker run --env-file`, a kubernetes Secret, or any
# orchestrator's secret store. This overlay is for deployments that pull secrets from
# Infisical at container start instead. See README.md in this directory.
#
# Build it against a published tag:
#
#   docker build -f deploy/infisical/Dockerfile \
#     --build-arg FLASH_IMAGE=ghcr.io/freesolo-co/freesolo-flash:main \
#     -t flash-control-plane-infisical .
#
# ...or against an image you just built yourself:
#
#   docker build -t flash-control-plane .
#   docker build -f deploy/infisical/Dockerfile \
#     --build-arg FLASH_IMAGE=flash-control-plane -t flash-control-plane-infisical .
#
# The build context is the repository root (that is where entrypoint.sh is copied from).

ARG FLASH_IMAGE=ghcr.io/freesolo-co/freesolo-flash:main
FROM ${FLASH_IMAGE}

# The Infisical CLI comes from the vendor's apt repo, and is deliberately NOT installed by the
# base image: the published artifact ships no vendor secret tooling and no vendor apt source.
# (The base image's `COPY . .` does carry this directory's inert shell script, but nothing in
# the base image ever executes it.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl -1sLf 'https://artifacts-cli.infisical.com/setup.deb.sh' | bash \
    && apt-get update && apt-get install -y --no-install-recommends infisical \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/infisical/entrypoint.sh /usr/local/bin/flash-infisical-entrypoint
RUN chmod +x /usr/local/bin/flash-infisical-entrypoint

# No-op passthrough unless INFISICAL_CLIENT_ID is set, otherwise `infisical login`
# (universal-auth) then `infisical run` before the server.
ENTRYPOINT ["/usr/local/bin/flash-infisical-entrypoint"]

# CMD must be restated, and must stay in step with the base Dockerfile's CMD. Setting
# ENTRYPOINT in a DERIVED image resets the inherited CMD to null, so without this line the
# wrapper is handed no command at all and the container never starts the server. Do not
# "deduplicate" this by deleting it.
CMD ["python", "-m", "flash.server", "--host", "0.0.0.0", "--port", "8080"]
