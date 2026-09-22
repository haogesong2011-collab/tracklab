#!/usr/bin/env python3
"""Write the release update.json from app version constants. Do not hand-edit it."""

from __future__ import annotations

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
) -> dict[str, object]:
    return {
        "version": version,
        "minimum_version": minimum_version,
        "critical": bool(critical),
    }


def main() -> int:
    dest = Path(sys.argv[1] if len(sys.argv) > 1 else "update.json")
    dest.write_text(
        json.dumps(build_manifest(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
