"""In-app macOS installer replacement (no network, no hdiutil)."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.patch_runtime import runtime_fingerprint  # noqa: E402
from app.self_update import (  # noqa: E402
    REPLACE_SCRIPT,
    SelfUpdateError,
    find_bundled_app,
    prepare_update,
    sha256_file,
    write_replace_script,
)
from app.update_checker import PatchSpec, ReleaseAsset, UpdateInfo, UpdateStatus  # noqa: E402


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

    def _layout(self, app: Path, marker: str, runtime: str = "torch") -> None:
        mac = app / "Contents" / "MacOS"
        mac.mkdir(parents=True)
        exe = mac / "TrackLab"
        exe.write_text(marker, encoding="utf-8")
        exe.chmod(0o755)
        (app / "Contents" / "Info.plist").write_text("plist", encoding="utf-8")
        styles = app / "Contents" / "Resources" / "app"
        styles.mkdir(parents=True)
        (styles / "style.qss").write_text("qss", encoding="utf-8")
        fw = app / "Contents" / "Frameworks"
        fw.mkdir()
        (fw / "torch").write_text(runtime, encoding="utf-8")

    def test_runtime_fingerprint_ignores_the_executable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            app = Path(raw) / "TrackLab.app"
            self._layout(app, "old")
            first = runtime_fingerprint(app)
            (app / "Contents" / "MacOS" / "TrackLab").write_text("new", encoding="utf-8")
            self.assertEqual(runtime_fingerprint(app), first)
            (app / "Contents" / "Frameworks" / "torch").write_text("other-build", encoding="utf-8")
            self.assertNotEqual(runtime_fingerprint(app), first)

    def _patch_zip(self, app: Path, marker: str) -> tuple[Path, str]:
        dest = app.parent / "TrackLab-arm64-patch.zip"
        exe = app / "Contents" / "MacOS" / "TrackLab"
        exe.write_text(marker, encoding="utf-8")
        with zipfile.ZipFile(dest, "w") as archive:
            for relative in (
                "Contents/MacOS/TrackLab",
                "Contents/Info.plist",
                "Contents/Resources/app/style.qss",
            ):
                info = zipfile.ZipInfo(relative)
                info.external_attr = 0o755 << 16
                archive.writestr(info, (app / relative).read_bytes())
        return dest, hashlib.sha256(dest.read_bytes()).hexdigest()

    def test_prepare_update_applies_patch_when_runtime_matches(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            bundle = root / "installed" / "TrackLab.app"
            self._layout(bundle, "old")
            fingerprint = runtime_fingerprint(bundle)
            built = root / "built" / "TrackLab.app"
            self._layout(built, "patched")
            archive, digest = self._patch_zip(built, "patched")
            info = UpdateInfo(
                status=UpdateStatus.AVAILABLE,
                current="0.3.4",
                latest="v0.3.5",
                assets=(
                    ReleaseAsset(
                        name="TrackLab-arm64.dmg",
                        url="https://github.com/haogesong2011-collab/tracklab/releases/download/v0.3.5/TrackLab-arm64.dmg",
                    ),
                ),
                patches=(
                    PatchSpec(
                        arch="arm64",
                        asset=archive.name,
                        runtime=fingerprint,
                        sha256=digest,
                        size=archive.stat().st_size,
                    ),
                ),
            )

            def fake_download(url, dest, **_kwargs):  # noqa: ANN001
                self.assertTrue(url.endswith(archive.name))
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(archive, dest)

            def copy_bundle(source: Path, dest: Path) -> None:
                shutil.copytree(source, dest)

            old = os.environ.get("TRACKLAB_UPDATE_DIR")
            os.environ["TRACKLAB_UPDATE_DIR"] = str(root / "cache")
            try:
                new_app = prepare_update(
                    info,
                    bundle=bundle,
                    current="0.3.4",
                    machine="arm64",
                    download=fake_download,
                    sign=lambda _path: None,
                    copy_bundle=copy_bundle,
                )
            finally:
                if old is None:
                    os.environ.pop("TRACKLAB_UPDATE_DIR", None)
                else:
                    os.environ["TRACKLAB_UPDATE_DIR"] = old
            text = (new_app / "Contents" / "MacOS" / "TrackLab").read_text(encoding="utf-8")
            self.assertEqual(text, "patched")
            self.assertEqual(
                (new_app / "Contents" / "Frameworks" / "torch").read_text(encoding="utf-8"),
                "torch",
            )

    def test_prepare_update_falls_back_when_runtime_differs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            bundle = root / "installed" / "TrackLab.app"
            self._layout(bundle, "old", runtime="torch-a")
            other = root / "other" / "TrackLab.app"
            self._layout(other, "old", runtime="torch-b")
            built = root / "built" / "TrackLab.app"
            self._layout(built, "patched")
            archive, digest = self._patch_zip(built, "patched")
            info = UpdateInfo(
                status=UpdateStatus.AVAILABLE,
                current="0.3.4",
                latest="v0.3.5",
                assets=(
                    ReleaseAsset(
                        name="TrackLab-arm64.dmg",
                        url="https://github.com/haogesong2011-collab/tracklab/releases/download/v0.3.5/TrackLab-arm64.dmg",
                        digest="sha256:" + ("ab" * 32),
                    ),
                ),
                patches=(
                    PatchSpec(
                        arch="arm64",
                        asset=archive.name,
                        runtime=runtime_fingerprint(other),
                        sha256=digest,
                    ),
                ),
            )
            payload = b"full-dmg"

            def fake_download(url, dest, **_kwargs):  # noqa: ANN001
                dest.parent.mkdir(parents=True, exist_ok=True)
                if url.endswith(".zip"):
                    raise AssertionError("patch should not be downloaded")
                dest.write_bytes(payload)

            import app.self_update as module

            def fake_extract(dmg: Path, dest_app: Path, **_kwargs) -> Path:
                self.assertEqual(dmg.read_bytes(), payload)
                self._layout(dest_app, "from-dmg")
                return dest_app

            original = module.extract_app_from_dmg
            module.extract_app_from_dmg = fake_extract  # type: ignore[method-assign]
            # Checksum path: digest on the asset must match payload. Rebuild info.
            digest_dmg = hashlib.sha256(payload).hexdigest()
            info = UpdateInfo(
                status=info.status,
                current=info.current,
                latest=info.latest,
                assets=(
                    ReleaseAsset(
                        name="TrackLab-arm64.dmg",
                        url=info.assets[0].url,
                        digest="sha256:" + digest_dmg,
                    ),
                ),
                patches=info.patches,
            )
            old = os.environ.get("TRACKLAB_UPDATE_DIR")
            os.environ["TRACKLAB_UPDATE_DIR"] = str(root / "cache")
            try:
                new_app = prepare_update(
                    info,
                    bundle=bundle,
                    current="0.3.4",
                    machine="arm64",
                    download=fake_download,
                    sign=lambda _path: None,
                )
            finally:
                module.extract_app_from_dmg = original  # type: ignore[method-assign]
                if old is None:
                    os.environ.pop("TRACKLAB_UPDATE_DIR", None)
                else:
                    os.environ["TRACKLAB_UPDATE_DIR"] = old
            self.assertEqual(
                (new_app / "Contents" / "MacOS" / "TrackLab").read_text(encoding="utf-8"),
                "from-dmg",
            )

    def test_prepare_update_rejects_a_bad_patch_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            bundle = root / "installed" / "TrackLab.app"
            self._layout(bundle, "old")
            built = root / "built" / "TrackLab.app"
            self._layout(built, "patched")
            archive, _digest = self._patch_zip(built, "patched")
            info = UpdateInfo(
                status=UpdateStatus.AVAILABLE,
                current="0.3.4",
                latest="v0.3.5",
                assets=(),
                patches=(
                    PatchSpec(
                        arch="arm64",
                        asset=archive.name,
                        runtime=runtime_fingerprint(bundle),
                        sha256="ab" * 32,
                    ),
                ),
            )

            def fake_download(url, dest, **_kwargs):  # noqa: ANN001
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(archive, dest)

            old = os.environ.get("TRACKLAB_UPDATE_DIR")
            os.environ["TRACKLAB_UPDATE_DIR"] = str(root / "cache")
            try:
                with self.assertRaises(SelfUpdateError):
                    prepare_update(
                        info,
                        bundle=bundle,
                        current="0.3.4",
                        machine="arm64",
                        download=fake_download,
                        sign=lambda _path: None,
                        copy_bundle=lambda source, dest: shutil.copytree(source, dest),
                    )
            finally:
                if old is None:
                    os.environ.pop("TRACKLAB_UPDATE_DIR", None)
                else:
                    os.environ["TRACKLAB_UPDATE_DIR"] = old
            self.assertEqual(list((root / "cache").rglob("TrackLab")), [])

    def test_launch_replacer_is_not_a_child_of_this_process(self) -> None:
        import app.self_update as module

        calls: list[list[str]] = []

        def fake_popen(argv, **_kwargs):  # noqa: ANN001
            calls.append(list(argv))
            return None

        original = module.subprocess.Popen
        module.subprocess.Popen = fake_popen  # type: ignore[method-assign]
        old = os.environ.get("TRACKLAB_UPDATE_DIR")
        try:
            with tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                os.environ["TRACKLAB_UPDATE_DIR"] = str(root)
                new_app = root / "0.3.7" / "TrackLab.app"
                new_app.mkdir(parents=True)
                module.launch_replacer(pid=123, bundle=root / "TrackLab.app", new_app=new_app)
        finally:
            module.subprocess.Popen = original  # type: ignore[method-assign]
            if old is None:
                os.environ.pop("TRACKLAB_UPDATE_DIR", None)
            else:
                os.environ["TRACKLAB_UPDATE_DIR"] = old
        self.assertTrue(calls)
        self.assertEqual(calls[0][0], "/usr/bin/osascript")
        self.assertIn("nohup", calls[0][-1])
        self.assertTrue(calls[0][-1].startswith('do shell script "'))
        self.assertNotIn("QProcess", calls[0][-1])

    def test_replace_script_waits_while_the_old_process_is_alive(self) -> None:
        self.assertIn("old process still running", REPLACE_SCRIPT)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            home = root / "home"
            home.mkdir()
            dest = self._app(root, "TrackLab.app", "old")
            src = self._app(root / "stage", "TrackLab.app", "new")
            sleeper = subprocess.Popen(["/bin/sleep", "30"])
            opener = home / "open-stub.sh"
            opener.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            opener.chmod(0o755)
            env = os.environ.copy()
            env["HOME"] = str(home)
            env["TRACKLAB_OPEN"] = str(opener)
            env["TRACKLAB_ALERT_LOG"] = str(home / "alert.txt")
            env["TRACKLAB_WAIT_TICKS"] = "1"
            script = write_replace_script(home)
            try:
                result = subprocess.run(
                    ["/bin/bash", str(script), str(sleeper.pid), str(dest), str(src)],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=10,
                )
            finally:
                sleeper.kill()
                sleeper.wait(timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((dest / "Contents" / "marker").read_text(encoding="utf-8"), "old")
            self.assertIn("仍在运行", (home / "alert.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
