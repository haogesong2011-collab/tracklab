#!/usr/bin/env bash
# Build TrackLab.app and TrackLab-<arch>.dmg for the current Mac architecture.
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "macOS packaging must run on Darwin" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

ARCH="$(uname -m)"
APP="$ROOT/dist/TrackLab.app"
DMG="$ROOT/dist/TrackLab-${ARCH}.dmg"

"$PYTHON" macos-packaging/make_icon.py
"$PYTHON" -m PyInstaller \
  --noconfirm \
  --clean \
  --distpath "$ROOT/dist" \
  --workpath "$ROOT/build/pyinstaller" \
  "$ROOT/macos-packaging/TrackLab.spec"

if [[ ! -d "$APP" ]]; then
  echo "PyInstaller did not produce $APP" >&2
  exit 1
fi

"$PYTHON" - "$APP" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
leaks = []
for path in root.rglob("*"):
    name = path.name.lower()
    rel = str(path.relative_to(root))
    if name in {"torch", "torchvision", "torchaudio", "sam2", "moge", "cv2"} and path.is_dir():
        leaks.append(rel)
    if "libtorch" in name:
        leaks.append(rel)
if leaks:
    print("standard bundle leaked heavy optional deps:", *leaks, sep="\n  ")
    sys.exit(1)
print("bundle leak check ok")
PY

codesign --force --deep --sign - --timestamp=none "$APP"

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
cp -R "$APP" "$STAGING/TrackLab.app"
ln -s /Applications "$STAGING/Applications"
rm -f "$DMG"
hdiutil create \
  -volname "TrackLab" \
  -srcfolder "$STAGING" \
  -ov \
  -format UDZO \
  "$DMG"

echo "built $APP"
echo "built $DMG"
