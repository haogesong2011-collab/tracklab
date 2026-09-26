"""Fingerprint the parts of a TrackLab.app that a code patch must not touch."""

from __future__ import annotations

import hashlib
from pathlib import Path

# Replaced by a code patch. Everything else is the runtime fingerprint.
PATCH_FILES = (
    "Contents/MacOS/TrackLab",
    "Contents/Info.plist",
)
PATCH_PREFIXES = ("Contents/Resources/app/",)
FINGERPRINT_SKIP_PREFIXES = ("Contents/_CodeSignature/",)


def is_patch_path(relative: str) -> bool:
    text = relative.replace("\\", "/").lstrip("/")
    if text in PATCH_FILES or text == "Contents/Resources/app":
        return True
    return any(text.startswith(prefix) for prefix in PATCH_PREFIXES)


def _ignored_for_fingerprint(relative: str) -> bool:
    text = relative.replace("\\", "/").lstrip("/")
    if is_patch_path(text):
        return True
    return any(text.startswith(prefix) for prefix in FINGERPRINT_SKIP_PREFIXES)


def iter_patch_files(bundle: Path) -> list[str]:
    """Paths inside the bundle that a patch zip should contain, sorted."""
    root = Path(bundle)
    found: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if is_patch_path(relative):
            found.append(relative)
    return sorted(found)


def runtime_fingerprint(bundle: Path) -> str:
    """SHA-256 of sorted ``relative-path<TAB>size`` lines for non-patch files.

    Symlinks are recorded by their own size and are not followed, so a
    Frameworks link to Resources is not hashed twice.
    """
    root = Path(bundle)
    rows: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if _ignored_for_fingerprint(relative):
            continue
        if path.is_symlink() or path.is_file():
            rows.append(f"{relative}\t{path.lstat().st_size}")
    rows.sort()
    payload = "\n".join(rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
