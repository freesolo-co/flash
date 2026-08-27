#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_OS:?RUNNER_OS is required}"
: "${RUNNER_ARCH:?RUNNER_ARCH is required}"
: "${TOOLCHAIN_PATH:?TOOLCHAIN_PATH is required}"
: "${GITHUB_PATH:?GITHUB_PATH is required}"
: "${GITHUB_ENV:?GITHUB_ENV is required}"
: "${GITHUB_OUTPUT:?GITHUB_OUTPUT is required}"

if [[ "$RUNNER_OS" != "Linux" || "$RUNNER_ARCH" != "X64" ]]; then
  printf 'the uv toolchain checksum is defined only for Linux x86_64 runners\n' >&2
  exit 1
fi

mapfile -t toolchain < <(
  python3 - "$TOOLCHAIN_PATH" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
toolchain = json.loads(path.read_text(encoding="utf-8"))
expected_keys = {
    "version",
    "github_release_linux_x86_64_sha256",
}
if set(toolchain) != expected_keys:
    raise SystemExit("uv-toolchain.json has unexpected or missing fields")

version = toolchain["version"]
checksum = toolchain["github_release_linux_x86_64_sha256"]
if re.fullmatch(r"[0-9]+(?:\.[0-9]+){2}", version) is None:
    raise SystemExit("uv-toolchain.json version must be an exact semantic version")
if re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
    raise SystemExit("uv-toolchain.json checksum must be a lowercase sha256 digest")
if checksum == "0" * 64:
    raise SystemExit("uv-toolchain.json checksum must not be all zeroes")

print(version)
print(checksum)
PY
)

if [[ ${#toolchain[@]} -ne 2 ]]; then
  printf '%s\n' "failed to read the uv toolchain" >&2
  exit 1
fi

version=${toolchain[0]}
checksum=${toolchain[1]}
state_root=$(mktemp -d /dev/shm/flash-uv-download.XXXXXX)
install_root=$(mktemp -d "/dev/shm/flash-uv-${version}.XXXXXX")
archive="$state_root/uv.tar.gz"
published=0
cleanup() {
  rm -rf "$state_root"
  if [[ $published -ne 1 ]]; then
    rm -rf "$install_root"
  fi
}
trap cleanup EXIT

mkdir -p /dev/shm/flash-uv-cache
curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location \
  --output "$archive" \
  "https://github.com/astral-sh/uv/releases/download/${version}/uv-x86_64-unknown-linux-gnu.tar.gz"
printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check --strict -
tar --extract --gzip --file "$archive" --directory "$install_root" --strip-components=1

installed_version=$("$install_root/uv" --version)
if [[ "$installed_version" != "uv $version" && "$installed_version" != "uv $version "* ]]; then
  printf 'verified archive reported unexpected version: %s\n' "$installed_version" >&2
  exit 1
fi

printf '%s\n' "$install_root" >> "$GITHUB_PATH"
printf '%s\n' "UV_CACHE_DIR=/dev/shm/flash-uv-cache" >> "$GITHUB_ENV"
printf 'version=%s\n' "$version" >> "$GITHUB_OUTPUT"
published=1
