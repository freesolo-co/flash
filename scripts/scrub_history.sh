#!/usr/bin/env bash
#
# One-time history scrub, to run BEFORE flipping this repository to public.
#
# What it removes from every commit on every ref:
#   - "Co-Authored-By: Claude ..." trailers            (333 commits)
#   - "Claude-Session: https://claude.ai/..." trailers  (84 commits)
#   - "Generated with ... Claude Code" lines            (1 commit)
#   - the build-box identity (a *.internal.cloudapp.net address) leaked into the
#     author/committer fields and into "Co-authored-by:" trailers
#
# This rewrites every commit sha. Run it once, on a fresh mirror clone, while the
# repository is still private. Rewriting after publication is pointless: the old
# shas stay reachable through forks, caches, and the GitHub API.
#
# Counts above were measured on origin/main + origin/dev at the time of writing.
# The script re-reports them, so re-check the printed before/after numbers rather
# than trusting the comment.
#
# Usage:
#   ./scripts/scrub_history.sh /tmp/flash-scrub
#
# It clones into that directory, rewrites, and stops. It never pushes. Review the
# result, then push manually (see the instructions it prints at the end).

set -euo pipefail

REMOTE="${FLASH_REMOTE:-https://github.com/freesolo-co/flash.git}"
WORKDIR="${1:-}"

if [ -z "$WORKDIR" ]; then
  echo "usage: $0 <empty-working-directory>" >&2
  exit 2
fi
if [ -e "$WORKDIR" ]; then
  echo "error: $WORKDIR already exists; pass a path that does not exist yet" >&2
  exit 2
fi

# git-filter-repo is a single python script; uv fetches it without a global install.
FILTER_REPO=(uv run --with git-filter-repo git-filter-repo)

echo "==> cloning a fresh mirror of $REMOTE"
# --mirror: filter-repo must see every ref (main, dev, tags, and any other branch).
# Rewriting a single-branch clone would leave the other branches on the old shas.
git clone --mirror "$REMOTE" "$WORKDIR"
cd "$WORKDIR"

count_pattern() {
  # commits (not lines) whose message matches $1. One git pass: messages are
  # NUL-delimited so a multi-line body counts once.
  git log --all --format='%B%x00' | awk -v pat="$1" '
    BEGIN { RS = "\0" }
    tolower($0) ~ tolower(pat) { n++ }
    END { print n + 0 }
  '
}

count_identities() {
  # commits whose author OR committer identity matches $1
  git log --all --format='%an <%ae>%n%cn <%ce>' | grep -ci -- "$1" || true
}

echo "==> before:"
echo "    total commits:         $(git rev-list --count --all)"
echo "    branches:              $(git for-each-ref --format='%(refname)' refs/heads | wc -l)"
echo "    Co-Authored-By Claude: $(count_pattern 'Co-Authored-By: Claude')"
echo "    Claude-Session:        $(count_pattern 'Claude-Session')"
echo "    Generated with Claude: $(count_pattern 'Generated with')"
echo "    leaked hostname:       $(count_pattern 'internal[.]cloudapp[.]net')"
echo "    Claude identities:     $(count_identities '^Claude')"
echo "    hostname identities:   $(count_identities 'internal[.]cloudapp[.]net')"

echo "==> rewriting commit messages and identities"

# --- identity rewrite ------------------------------------------------------
# Two problems in the author/committer fields:
#
#  1. A build box committed as "Ubuntu <...@...internal.cloudapp.net>", leaking an
#     internal hostname into the identity.
#  2. Some commits are authored by an AI assistant identity outright.
#
# Both are the maintainer's own work, so both map onto CANONICAL_IDENTITY below.
#
# The mailmap is GENERATED from the repository rather than hardcoded: writing the
# leaked hostname into this file would reintroduce, in the published tree, the exact
# string the scrub exists to remove.
CANONICAL_IDENTITY="${FLASH_CANONICAL_IDENTITY:-David Shan <78061174+DavidBShan@users.noreply.github.com>}"

MAILMAP="$WORKDIR/flash-scrub-mailmap"
git log --all --format='%an <%ae>%n%cn <%ce>' \
  | sort -u \
  | grep -iE 'internal\.cloudapp\.net|^Claude[[:space:]]*<|^Claude[[:space:]]' \
  | sed "s|^|$CANONICAL_IDENTITY |" > "$MAILMAP"

if [ ! -s "$MAILMAP" ]; then
  echo "    (no identities needed remapping)"
else
  echo "    remapping $(wc -l < "$MAILMAP") identit(ies) onto $CANONICAL_IDENTITY"
fi

# Drop whole trailer lines. Matching is line-anchored, so a commit body that merely
# mentions Claude in prose is left alone; only trailer-shaped lines are removed.
"${FILTER_REPO[@]}" \
  --mailmap "$MAILMAP" \
  --commit-callback '
import re

patterns = [
    rb"(?im)^[ \t]*Co-Authored-By:[ \t]*Claude[^\n]*\n?",
    rb"(?im)^[ \t]*Co-authored-by:[^\n]*internal\.cloudapp\.net[^\n]*\n?",
    rb"(?im)^[ \t]*Claude-Session:[^\n]*\n?",
    rb"(?im)^[^\n]*Generated with[^\n]*Claude[^\n]*\n?",
]
message = commit.message
for pattern in patterns:
    message = re.sub(pattern, b"", message)
# collapse the blank-line run the removed trailers leave behind
message = re.sub(rb"\n{3,}", b"\n\n", message)
commit.message = message.rstrip(b"\n") + b"\n"
' \
  --force

echo "==> after:"
echo "    total commits:         $(git rev-list --count --all)"
echo "    Co-Authored-By Claude: $(count_pattern 'Co-Authored-By: Claude')"
echo "    Claude-Session:        $(count_pattern 'Claude-Session')"
echo "    Generated with Claude: $(count_pattern 'Generated with')"
echo "    leaked hostname:       $(count_pattern 'internal[.]cloudapp[.]net')"
echo "    Claude identities:     $(count_identities '^Claude')"
echo "    hostname identities:   $(count_identities 'internal[.]cloudapp[.]net')"
echo
echo "    remaining author/committer identities:"
git log --all --format='%an <%ae>%n%cn <%ce>' | sort -u | sed 's/^/      /'

rm -f "$MAILMAP"

cat <<EOF

==> rewrite complete, nothing pushed.

Every count above should be 0, and no remaining identity should contain an
internal hostname or an AI-assistant name. Spot-check a few commits
(git log -n 20 --format=full) before continuing.

BEFORE PUBLISHING, prune stale branches. This mirror carries every branch the
remote has (the count is printed above; it was ~407 at the time of writing, of
which only ~28 are merged into dev). Going public publishes all of them, including
abandoned experiment branches nobody has reviewed for internal detail. Delete the
ones you do not want public from the mirror before pushing:

  git branch -D <branch>          # repeat, or script it against a keep-list

Keeping just main, dev, and any live release branches is the safe default.

To publish the rewritten history:

  1. Tell every collaborator to stop pushing and to re-clone afterwards. Their
     existing clones point at shas that will no longer exist.
  2. Temporarily disable branch protection on main and dev (a rewrite is a
     force-push; protection will otherwise reject it).
  3. From $WORKDIR:

       git remote add origin $REMOTE
       git push --force --mirror origin

  4. Re-enable branch protection, then flip the repository to public.
  5. Re-clone locally. Do not reuse an old clone.

Open pull requests will show as rewritten and are best closed and reopened from
fresh branches.
EOF
