"""Fast CI entry: unit tests + decoder integration + frozen 10-clip eval."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _unittest_cmd(*names: str, discover: str | None = None) -> list[str]:
    """Run unittest, then exit without tearing Qt down.

    PySide's macOS shutdown segfaults after a green suite (signal 11, which
    Actions reports as 245). The result is already known, so skip that teardown.
    """
    script = """
import os
import sys
import unittest

loader = unittest.TestLoader()
if sys.argv[1] == "--discover":
    suite = loader.discover(sys.argv[2])
else:
    suite = loader.loadTestsFromNames(sys.argv[1:])
result = unittest.TextTestRunner(verbosity=2).run(suite)
os._exit(0 if result.wasSuccessful() else 1)
"""
    if discover is not None:
        return [sys.executable, "-c", script, "--discover", discover]
    return [sys.executable, "-c", script, *names]


def main() -> int:
    cmds = [
        _unittest_cmd(discover="tests/ai/unit"),
        _unittest_cmd(
            "tests.ai.integration.test_pipeline",
            "tests.ai.integration.test_throughput",
        ),
        _unittest_cmd(
            "tests.ai.integration.test_desktop_acceptance",
            "tests.ai.integration.test_deepseek_client",
            "tests.ai.integration.test_packaged_app",
        ),
        [sys.executable, "-m", "tests.ai.evaluate", "--split", "ci", "--model", "oracle"],
        [sys.executable, "-m", "tests.ai.evaluate", "--split", "ci", "--model", "baseline"],
    ]
    if not (ROOT / "datasets" / "manifest.json").exists():
        cmds.insert(0, [sys.executable, "-m", "tests.ai.generate_fixtures"])
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("TRACKLAB_SKIP_UPDATE_CHECK", "1")
    env.setdefault("TRACKLAB_SKIP_TUTORIAL", "1")
    for cmd in cmds:
        print("+", " ".join(cmd), flush=True)
        completed = subprocess.run(cmd, cwd=ROOT, env=env)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
