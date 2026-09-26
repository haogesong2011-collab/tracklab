#!/usr/bin/env python3
"""Write the release update.json from app version constants. Do not hand-edit it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import __minimum_version__, __update_critical__, __version__  # noqa: E402


def build_manifest(
    version: str = __version__,
    minimum_version: str = __minimum_version__,
    critical: bool = __update_critical__,
    notes: str = "",
    assets: list[str] | None = None,
    patch: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": version,
        "minimum_version": minimum_version,
        "critical": bool(critical),
    }
    text = str(notes or "").strip()
    if text:
        payload["notes"] = text
    if assets:
        payload["assets"] = list(assets)
    if patch:
        payload["patch"] = patch
    return payload


def _changelog_notes() -> str:
    changelog = ROOT / "CHANGELOG.md"
    if not changelog.is_file():
        return ""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tracklab_release_notes",
        ROOT / "macos-packaging" / "release_notes.py",
    )
    if spec is None or spec.loader is None:
        return ""
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.extract_notes(changelog.read_text(encoding="utf-8"), f"v{__version__}"))


def collect_patches(paths: list[Path]) -> dict[str, object]:
    """Read ``patch`` objects written by ``make_patch.py`` for each architecture."""
    patch: dict[str, object] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        extra = payload.get("patch") if isinstance(payload, dict) else None
        if not isinstance(extra, dict) or not extra:
            raise SystemExit(f"{path} has no patch object")
        for arch, spec in extra.items():
            if not isinstance(spec, dict) or not spec.get("asset") or not spec.get("sha256"):
                raise SystemExit(f"{path} patch entry {arch} is incomplete")
            patch[str(arch)] = spec
    return patch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dest", nargs="?", default="update.json")
    parser.add_argument(
        "--patch-manifest",
        action="append",
        default=[],
        help="update.json produced by one architecture build",
    )
    args = parser.parse_args(argv)
    patch = collect_patches([Path(item) for item in args.patch_manifest])
    if args.patch_manifest and set(patch) != {"arm64", "x86_64"}:
        raise SystemExit(f"expected arm64 and x86_64 patches, got {sorted(patch)}")
    assets = ["TrackLab-arm64.dmg", "TrackLab-x86_64.dmg", "SHA256SUMS.txt"]
    for spec in patch.values():
        if isinstance(spec, dict):
            name = str(spec.get("asset") or "")
            if name and name not in assets:
                assets.append(name)
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = build_manifest(
        notes=_changelog_notes(),
        assets=assets,
        patch=patch or None,
    )
    dest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
