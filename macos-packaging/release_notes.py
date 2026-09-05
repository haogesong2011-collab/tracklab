#!/usr/bin/env python3
"""Extract the CHANGELOG section for a vMAJOR.MINOR.PATCH tag."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract_notes(changelog: str, tag: str) -> str:
    version = tag.strip()
    if version.startswith(("v", "V")):
        version = version[1:]
    pattern = rf"^## {re.escape(version)}\b[^\n]*\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, changelog, flags=re.M | re.S)
    if match is None:
        return f"TrackLab v{version}"
    body = match.group(1).strip()
    return body or f"TrackLab v{version}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument(
        "--changelog",
        default=str(ROOT / "CHANGELOG.md"),
    )
    args = parser.parse_args()
    text = Path(args.changelog).read_text(encoding="utf-8")
    print(extract_notes(text, args.tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
