#!/usr/bin/env python3
"""Zip the code-only files of a built TrackLab.app and record its runtime fingerprint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.patch_runtime import iter_patch_files, runtime_fingerprint  # noqa: E402


def build_patch_zip(bundle: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    files = iter_patch_files(bundle)
    if "Contents/MacOS/TrackLab" not in files:
        raise SystemExit(f"{bundle} has no Contents/MacOS/TrackLab")
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            path = bundle / relative
            info = zipfile.ZipInfo(relative)
            mode = path.stat().st_mode & 0xFFFF
            info.external_attr = mode << 16
            archive.writestr(info, path.read_bytes())
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    return digest


def merge_manifest(
    manifest_path: Path,
    arch: str,
    asset: str,
    runtime: str,
    digest: str,
    size: int,
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{manifest_path} is not a JSON object")
    names = [str(item) for item in (payload.get("assets") or [])]
    for extra in (f"TrackLab-{arch}.dmg", "SHA256SUMS.txt", asset):
        if extra not in names:
            names.append(extra)
    payload["assets"] = names
    patch = dict(payload.get("patch") or {})
    patch[arch] = {
        "asset": asset,
        "runtime": runtime,
        "sha256": digest,
        "size": size,
    }
    payload["patch"] = patch
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--arch", required=True, choices=("arm64", "x86_64"))
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    bundle = args.app.resolve()
    digest = build_patch_zip(bundle, args.zip)
    runtime = runtime_fingerprint(bundle)
    size = args.zip.stat().st_size
    print(f"patch {args.zip.name} sha256={digest} bytes={size}")
    print(f"runtime {runtime}")
    if args.manifest is not None:
        merge_manifest(
            args.manifest,
            args.arch,
            args.zip.name,
            runtime,
            digest,
            size,
        )
        print(f"updated {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
