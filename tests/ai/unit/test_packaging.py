"""Packaging metadata and resource-path smoke tests (no PyInstaller required)."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import (  # noqa: E402
    BUNDLE_IDENTIFIER,
    GITHUB_REPO,
    __minimum_version__,
    __update_critical__,
    __version__,
)
from app.paths import is_frozen, style_path  # noqa: E402
from app.update_checker import parse_version  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _release_notes_module():
    return _load_module(
        "tracklab_release_notes",
        ROOT / "macos-packaging" / "release_notes.py",
    )


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
        self.assertIn("app.tutorial", spec)
        self.assertIn("app.update_checker", spec)
        self.assertIn("app.self_update", spec)
        self.assertIn("ai.sam_runtime", spec)
        self.assertIn("ai.plane", spec)
        self.assertIn("ai.charuco", spec)
        self.assertIn("ai.depth_audit", spec)
        self.assertIn("ai.autotracker", spec)

    def test_spec_filters_analysis_output_not_just_collect_all(self) -> None:
        spec = (ROOT / "macos-packaging" / "TrackLab.spec").read_text(encoding="utf-8")
        self.assertIn(
            "a.binaries = [entry for entry in a.binaries if _keep_binary(entry)]", spec
        )
        self.assertIn("a.datas = [entry for entry in a.datas if _keep_data(entry)]", spec)
        hints = spec.split("DROP_HINTS = (", 1)[1].split(")", 1)[0]
        for unused in ("WebEngine", "QtQuick", "QtDesigner", "qmlls", "QtShaderTools"):
            self.assertIn(f'"{unused}"', hints)
        # PyAV ships ffmpeg under the same soname family as Qt; filtering those
        # by name would break every video import.
        for needed in ("libavcodec", "libavformat", "libavutil", "libswscale"):
            self.assertNotIn(f'"{needed}"', hints)
        self.assertNotIn('"QtOpenGL"', hints)  # QtCharts links against it
        prefixes = spec.split("DROP_DATA_PREFIXES = (", 1)[1].split(")", 1)[0]
        self.assertIn('"torch/include"', prefixes)
        # These are imported at runtime; dropping them broke the frozen app.
        for needed in ("torch/distributed", "torch/onnx", "torch/testing"):
            self.assertNotIn(f'"{needed}"', prefixes)
        qt_ffmpeg = spec.split("QT_FFMPEG_SOURCES = (", 1)[1].split(")", 1)[0]
        self.assertIn('"PySide6/Qt/lib/libav"', qt_ffmpeg)

    def test_build_script_verifies_slimming_and_cleans_up(self) -> None:
        script = (ROOT / "macos-packaging" / "build_macos.sh").read_text(encoding="utf-8")
        self.assertIn("QtWebEngineCore", script)
        self.assertIn("libavcodec*.dylib", script)
        self.assertIn("--smoke", script)
        self.assertIn('rm -rf "$ROOT/dist/TrackLab" "$ROOT/build/pyinstaller"', script)

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
        self.assertIn("dist-upload/update.json", workflow)
        self.assertIn("write_update_json.py", workflow)
        self.assertIn("MACOS_CERT", script)
        self.assertIn("notarytool submit", script)

    def test_release_script_and_update_manifest(self) -> None:
        import json
        import os

        script = ROOT / "macos-packaging" / "release.sh"
        self.assertTrue(script.is_file())
        self.assertTrue(os.access(script, os.X_OK))
        text = script.read_text(encoding="utf-8")
        self.assertIn("git status --porcelain", text)
        self.assertIn("release_notes.py", text)
        self.assertIn('git tag "$tag"', text)
        self.assertIn("git push origin", text)
        writer = _load_module(
            "tracklab_write_update_json",
            ROOT / "macos-packaging" / "write_update_json.py",
        )
        manifest = writer.build_manifest()
        self.assertEqual(manifest["version"], __version__)
        self.assertEqual(manifest["minimum_version"], __minimum_version__)
        self.assertIs(__update_critical__, False)
        self.assertEqual(manifest["critical"], False)
        self.assertEqual(
            set(json.loads(json.dumps(manifest))),
            {"version", "minimum_version", "critical"},
        )

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
