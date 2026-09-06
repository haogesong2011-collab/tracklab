"""Packaging metadata and resource-path smoke tests (no PyInstaller required)."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import BUNDLE_IDENTIFIER, GITHUB_REPO, __version__  # noqa: E402
from app.paths import is_frozen, style_path  # noqa: E402
from app.update_checker import parse_version  # noqa: E402


def _release_notes_module():
    path = ROOT / "macos-packaging" / "release_notes.py"
    spec = importlib.util.spec_from_file_location("tracklab_release_notes", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PackagingMetadataTests(unittest.TestCase):
    def test_version_is_semver(self) -> None:
        self.assertIsNotNone(parse_version(__version__))
        self.assertEqual(GITHUB_REPO, "haogesong2011-collab/tracklab")
        self.assertEqual(BUNDLE_IDENTIFIER, "com.tracklab.app")

    def test_stylesheet_exists_from_source(self) -> None:
        self.assertFalse(is_frozen())
        path = style_path()
        self.assertTrue(path.is_file(), path)
        self.assertIn("QMainWindow", path.read_text(encoding="utf-8"))

    def test_spec_includes_precise_runtime(self) -> None:
        spec = (ROOT / "macos-packaging" / "TrackLab.spec").read_text(encoding="utf-8")
        excludes = spec.split("EXCLUDES = [", 1)[1].split("]", 1)[0]
        self.assertNotIn('"torch"', excludes)
        self.assertNotIn('"sam2"', excludes)
        self.assertIn('"moge"', excludes)
        self.assertIn('"cv2"', excludes)
        self.assertIn('"torch"', spec)
        self.assertIn('"sam2"', spec)
        self.assertIn('"models"', spec)
        self.assertIn("style.qss", spec)
        self.assertIn("BUNDLE_IDENTIFIER", spec)
        self.assertIn("app.download_toast", spec)
        self.assertIn("app.chart_ticks", spec)
        self.assertIn("app.update_checker", spec)
        self.assertIn("app.self_update", spec)
        self.assertIn("ai.sam_runtime", spec)
        self.assertIn("ai.plane", spec)
        self.assertIn("ai.charuco", spec)
        self.assertIn("ai.depth_audit", spec)
        self.assertIn("ai.autotracker", spec)

    def test_build_script_and_workflow(self) -> None:
        script = (ROOT / "macos-packaging" / "build_macos.sh").read_text(encoding="utf-8")
        self.assertIn("codesign --force --deep --sign -", script)
        self.assertIn("TrackLab-${ARCH}.dmg", script)
        workflow = (ROOT / ".github" / "workflows" / "release-macos.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("macos-15-intel", workflow)
        self.assertIn("workflow_dispatch", workflow)
        self.assertIn('tags:', workflow)
        self.assertIn("startsWith(github.ref, 'refs/tags/v')", workflow)
        self.assertIn("TrackLab-arm64.dmg", workflow)
        self.assertIn("TrackLab-x86_64.dmg", workflow)

    def test_changelog_notes_for_current_version(self) -> None:
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        notes = _release_notes_module().extract_notes(changelog, f"v{__version__}")
        self.assertTrue(notes)
        self.assertNotEqual(notes, f"TrackLab v{__version__}")
        missing = _release_notes_module().extract_notes(changelog, "v9.9.9")
        self.assertEqual(missing, "TrackLab v9.9.9")

    def test_module_version_flag(self) -> None:
        import subprocess

        completed = subprocess.run(
            [sys.executable, "-m", "app", "--version"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), __version__)


if __name__ == "__main__":
    unittest.main()
