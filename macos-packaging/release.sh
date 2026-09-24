#!/usr/bin/env bash
# Bump the app version, tag it, and push so GitHub Actions publishes the DMGs.
# Changelog section for that version must already exist.
# Usage: macos-packaging/release.sh X.Y.Z
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -ne 1 ]]; then
  echo "usage: macos-packaging/release.sh X.Y.Z" >&2
  exit 1
fi

VERSION="$1"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "version must be MAJOR.MINOR.PATCH, got: $VERSION" >&2
  exit 1
fi

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$branch" != "main" ]]; then
  echo "release must run on main (current: $branch)" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "working tree is not clean" >&2
  exit 1
fi

notes="$(python3 macos-packaging/release_notes.py "v${VERSION}")"
if [[ "$notes" == "TrackLab v${VERSION}" ]]; then
  echo "CHANGELOG.md has no notes for ${VERSION}" >&2
  exit 1
fi

python3 - "$VERSION" <<'PY'
import re
import sys
from pathlib import Path

version = sys.argv[1]
path = Path("app/__init__.py")
text = path.read_text(encoding="utf-8")
updated, count = re.subn(
    r'__version__\s*=\s*"[^"]+"',
    f'__version__ = "{version}"',
    text,
    count=1,
)
if count != 1:
    raise SystemExit("could not update __version__ in app/__init__.py")
path.write_text(updated, encoding="utf-8")
PY

if ! git diff --quiet -- app/__init__.py; then
  git add app/__init__.py
  git commit -m "Release ${VERSION}"
fi

tag="v${VERSION}"
if git rev-parse "$tag" >/dev/null 2>&1; then
  echo "tag ${tag} already exists" >&2
  exit 1
fi

git tag "$tag"
git push origin HEAD
git push origin "$tag"
echo "https://github.com/haogesong2011-collab/tracklab/actions"
