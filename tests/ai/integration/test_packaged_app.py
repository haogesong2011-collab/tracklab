"""Frozen .app smoke test. Skips unless TRACKLAB_APP or dist/TrackLab.app exists."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import __version__  # noqa: E402


def _frozen_binary() -> Path | None:
    override = os.environ.get("TRACKLAB_APP", "").strip()
    if override:
        path = Path(override)
    else:
        path = ROOT / "dist" / "TrackLab.app"
    if path.is_dir() and path.name.endswith(".app"):
        path = path / "Contents" / "MacOS" / "TrackLab"
    if path.is_file():
        return path
    return None


class PackagedAppSmokeTests(unittest.TestCase):
    def test_frozen_binary_version_and_smoke(self) -> None:
        binary = _frozen_binary()
        if binary is None:
            self.skipTest("frozen app not built")
        env = os.environ.copy()
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        env.setdefault("TRACKLAB_SKIP_UPDATE_CHECK", "1")
        version = subprocess.run(
            [str(binary), "--version"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        self.assertEqual(version.returncode, 0, version.stderr)
        self.assertIn(__version__, version.stdout)
        smoke = subprocess.run(
            [str(binary), "--smoke"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertEqual(smoke.returncode, 0, smoke.stderr + smoke.stdout)
        self.assertIn("SMOKE OK", smoke.stdout)
        self.assertIn("frozen=True", smoke.stdout)


if __name__ == "__main__":
    unittest.main()
