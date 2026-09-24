"""Fast CI entry: unit tests + decoder integration + frozen 10-clip eval."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    cmds = [
        [sys.executable, "-m", "unittest", "discover", "-s", "tests/ai/unit", "-v"],
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.ai.integration.test_pipeline",
            "tests.ai.integration.test_throughput",
            "-v",
        ],
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.ai.integration.test_desktop_acceptance",
            "tests.ai.integration.test_deepseek_client",
            "tests.ai.integration.test_packaged_app",
            "-v",
        ],
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
