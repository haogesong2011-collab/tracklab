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

chmod -R u+w "$APP" "$ROOT/dist/TrackLab" "$ROOT/build/pyinstaller" 2>/dev/null || true
rm -rf "$APP" "$ROOT/dist/TrackLab" "$ROOT/build/pyinstaller"

"$PYTHON" macos-packaging/make_icon.py
"$PYTHON" - <<'PY'
from ai.model_manager import DEFAULT_SPEC, ensure_checkpoint

ensure_checkpoint(DEFAULT_SPEC, download=True)
print("sam checkpoint ready")
PY
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
names = {path.name.lower() for path in root.rglob("*") if path.is_dir() or path.is_file()}
if not any(name == "torch" or name.startswith("libtorch") for name in names):
    print("precise bundle missing torch")
    sys.exit(1)
if "sam2" not in names and not any("sam2" in name for name in names):
    print("precise bundle missing sam2")
    sys.exit(1)
ckpt = list(root.rglob("sam2.1_hiera_tiny.pt"))
if not ckpt:
    print("precise bundle missing SAM checkpoint")
    sys.exit(1)
if not list(root.rglob("libavcodec*.dylib")):
    print("bundle missing PyAV ffmpeg: video decoding would fail")
    sys.exit(1)
leaks = []
for path in root.rglob("*"):
    name = path.name.lower()
    rel = str(path.relative_to(root))
    if name in {"moge", "cv2"} and path.is_dir():
        leaks.append(rel)
for hint in ("QtWebEngineCore", "QtQuick3D", "QtDesigner", "torch/include"):
    hits = [
        str(p.relative_to(root))
        for p in root.rglob("*")
        if hint.replace("/", "") in str(p.relative_to(root)).replace("/", "")
    ]
    if hits:
        leaks.append(f"{hint}: {hits[0]}")
if leaks:
    print("bundle leaked optional deps:", *leaks, sep="\n  ")
    sys.exit(1)
print("bundle precise-mode check ok")
PY

# File-level checks cannot tell a dead tree from a live one: dropping
# torch/distributed once passed every check above and still failed to import.
if ! QT_QPA_PLATFORM=offscreen TRACKLAB_SKIP_UPDATE_CHECK=1 \
    "$APP/Contents/MacOS/TrackLab" --smoke; then
  echo "bundle failed the --smoke import/track check" >&2
  exit 1
fi

codesign_app() {
  if [[ -n "${MACOS_CERT:-}" ]]; then
    # Optional Developer ID path. Full notarization stays a hook until the
    # Apple credentials are actually configured in CI.
    codesign --force --deep --options runtime --timestamp --sign "$MACOS_CERT" "$APP"
    if [[ -n "${MACOS_NOTARY_KEYCHAIN_PROFILE:-}" ]]; then
      echo "notarization hook: xcrun notarytool submit \"$DMG\" --keychain-profile \"$MACOS_NOTARY_KEYCHAIN_PROFILE\" --wait"
      echo "notarization hook: xcrun stapler staple \"$APP\""
      echo "notarization requested but submit/staple is not enabled in this build"
    fi
  else
    codesign --force --deep --sign - --timestamp=none "$APP"
  fi
}

codesign_app

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

# The COLLECT onedir is a byte-for-byte duplicate of the bundle (~1.2 GB each).
chmod -R u+w "$ROOT/dist/TrackLab" "$ROOT/build/pyinstaller" 2>/dev/null || true
rm -rf "$ROOT/dist/TrackLab" "$ROOT/build/pyinstaller"

echo "built $APP ($(du -sh "$APP" | cut -f1))"
echo "built $DMG ($(du -sh "$DMG" | cut -f1))"
