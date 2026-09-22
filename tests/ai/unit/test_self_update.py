"""In-app macOS installer replacement (no network, no hdiutil)."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.self_update import (  # noqa: E402
    REPLACE_SCRIPT,
    SelfUpdateError,
    find_bundled_app,
    prepare_update,
    sha256_file,
    write_replace_script,
)
from app.update_checker import ReleaseAsset, UpdateInfo, UpdateStatus  # noqa: E402


class SelfUpdateTests(unittest.TestCase):
    def test_find_bundled_app_prefers_tracklab(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            mount = Path(raw)
            (mount / "Other.app" / "Contents").mkdir(parents=True)
            target = mount / "TrackLab.app" / "Contents"
            target.mkdir(parents=True)
            self.assertEqual(find_bundled_app(mount), mount / "TrackLab.app")

    def test_find_bundled_app_single_app(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            mount = Path(raw)
            app = mount / "Whatever.app"
            (app / "Contents").mkdir(parents=True)
            self.assertEqual(find_bundled_app(mount), app)

    def test_find_bundled_app_missing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(SelfUpdateError):
                find_bundled_app(Path(raw))

    def test_replace_script_uses_ditto_and_argv(self) -> None:
        self.assertIn("/usr/bin/ditto", REPLACE_SCRIPT)
        self.assertIn('"$OPENER" "$DEST"', REPLACE_SCRIPT)
        self.assertIn("TRACKLAB_OPEN:-open", REPLACE_SCRIPT)
        self.assertIn("${DEST}.new", REPLACE_SCRIPT)
        self.assertIn("updates/backup", REPLACE_SCRIPT)
        self.assertIn("更新失败，已恢复原版本。", REPLACE_SCRIPT)
        self.assertIn("kill -0", REPLACE_SCRIPT)
        with tempfile.TemporaryDirectory() as raw:
            script = write_replace_script(Path(raw))
            self.assertTrue(os.access(script, os.X_OK))
            self.assertIn("ditto", script.read_text(encoding="utf-8"))

    def test_prepare_update_downloads_verifies_and_extracts(self) -> None:
        payload = b"tracklab-dmg-bytes"
        digest = hashlib.sha256(payload).hexdigest()
        sums = f"{digest}  TrackLab-arm64.dmg\n"
        info = UpdateInfo(
            status=UpdateStatus.AVAILABLE,
            current="0.3.0",
            latest="v0.3.1",
            assets=(
                ReleaseAsset(
                    name="TrackLab-arm64.dmg",
                    url="https://github.com/haogesong2011-collab/tracklab/releases/download/v0.3.1/TrackLab-arm64.dmg",
                    size=len(payload),
                ),
                ReleaseAsset(
                    name="SHA256SUMS.txt",
                    url="https://github.com/haogesong2011-collab/tracklab/releases/download/v0.3.1/SHA256SUMS.txt",
                    size=len(sums),
                ),
            ),
        )
        extracted: list[Path] = []

        def fake_download(url, dest, **_kwargs):  # noqa: ANN001
            dest.parent.mkdir(parents=True, exist_ok=True)
            if url.endswith("SHA256SUMS.txt"):
                dest.write_text(sums, encoding="utf-8")
            else:
                dest.write_bytes(payload)

        def fake_extract(dmg: Path, dest_app: Path, **_kwargs) -> Path:  # noqa: ANN001
            self.assertEqual(sha256_file(dmg), digest)
            (dest_app / "Contents" / "MacOS").mkdir(parents=True)
            (dest_app / "Contents" / "MacOS" / "TrackLab").write_text("ok", encoding="utf-8")
            extracted.append(dest_app)
            return dest_app

        import app.self_update as module

        original = module.extract_app_from_dmg
        module.extract_app_from_dmg = fake_extract  # type: ignore[method-assign]
        old = os.environ.get("TRACKLAB_UPDATE_DIR")
        try:
            with tempfile.TemporaryDirectory() as raw:
                os.environ["TRACKLAB_UPDATE_DIR"] = raw
                bundle = Path(raw) / "current" / "TrackLab.app"
                bundle.mkdir(parents=True)
                new_app = prepare_update(
                    info,
                    bundle=bundle,
                    current="0.3.0",
                    machine="arm64",
                    download=fake_download,
                )
                self.assertTrue((new_app / "Contents" / "MacOS").is_dir())
                self.assertEqual(extracted[0], new_app)
                self.assertFalse((new_app.parent / "TrackLab-arm64.dmg").exists())
        finally:
            module.extract_app_from_dmg = original  # type: ignore[method-assign]
            if old is None:
                os.environ.pop("TRACKLAB_UPDATE_DIR", None)
            else:
                os.environ["TRACKLAB_UPDATE_DIR"] = old

    def test_prepare_update_rejects_bad_checksum(self) -> None:
        info = UpdateInfo(
            status=UpdateStatus.AVAILABLE,
            current="0.3.0",
            latest="v0.3.1",
            assets=(
                ReleaseAsset(
                    name="TrackLab-arm64.dmg",
                    url="https://github.com/haogesong2011-collab/tracklab/releases/download/v0.3.1/TrackLab-arm64.dmg",
                    digest="sha256:" + ("ab" * 32),
                ),
            ),
        )

        def fake_download(url, dest, **_kwargs):  # noqa: ANN001
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"nope")

        old = os.environ.get("TRACKLAB_UPDATE_DIR")
        try:
            with tempfile.TemporaryDirectory() as raw:
                os.environ["TRACKLAB_UPDATE_DIR"] = raw
                bundle = Path(raw) / "TrackLab.app"
                bundle.mkdir()
                with self.assertRaises(SelfUpdateError) as ctx:
                    prepare_update(
                        info,
                        bundle=bundle,
                        current="0.3.0",
                        machine="arm64",
                        download=fake_download,
                    )
                self.assertIn("校验", str(ctx.exception))
        finally:
            if old is None:
                os.environ.pop("TRACKLAB_UPDATE_DIR", None)
            else:
                os.environ["TRACKLAB_UPDATE_DIR"] = old

    def _app(self, root: Path, name: str, marker: str) -> Path:
        app = root / name
        contents = app / "Contents"
        contents.mkdir(parents=True)
        (contents / "marker").write_text(marker, encoding="utf-8")
        return app

    def _run_replace(self, home: Path, dest: Path, src: Path, *, open_code: int) -> subprocess.CompletedProcess[str]:
        opener = home / "open-stub.sh"
        opener.write_text("#!/bin/sh\nexit " + str(open_code) + "\n", encoding="utf-8")
        opener.chmod(0o755)
        alert = home / "alert.txt"
        env = os.environ.copy()
        env["HOME"] = str(home)
        env["TRACKLAB_OPEN"] = str(opener)
        env["TRACKLAB_ALERT_LOG"] = str(alert)
        script = write_replace_script(home)
        return subprocess.run(
            ["/bin/bash", str(script), "99999999", str(dest), str(src)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=20,
        )

    def test_replace_script_clears_backup_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            home = root / "home"
            home.mkdir()
            dest = self._app(root, "TrackLab.app", "old")
            src = self._app(root / "stage", "TrackLab.app", "new")
            result = self._run_replace(home, dest, src, open_code=0)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual((dest / "Contents" / "marker").read_text(encoding="utf-8"), "new")
            backup = home / "Library" / "Caches" / "tracklab" / "updates" / "backup"
            self.assertEqual(list(backup.glob("TrackLab-*.app")), [])
            log = (home / "Library" / "Logs" / "TrackLab-update.log").read_text(encoding="utf-8")
            self.assertIn("replaced", log)

    def test_replace_script_restores_backup_when_open_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            home = root / "home"
            home.mkdir()
            dest = self._app(root, "TrackLab.app", "old")
            src = self._app(root / "stage", "TrackLab.app", "new")
            result = self._run_replace(home, dest, src, open_code=1)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((dest / "Contents" / "marker").read_text(encoding="utf-8"), "old")
            alert = (home / "alert.txt").read_text(encoding="utf-8")
            self.assertIn("已恢复原版本", alert)
            self.assertFalse((root / "TrackLab.app.new").exists())


if __name__ == "__main__":
    unittest.main()
