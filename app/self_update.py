"""Download a GitHub DMG and replace the running macOS .app after quit."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from ai.contracts import CancelToken
from app.update_checker import (
    UpdateInfo,
    UpdateStatus,
    _ssl_context,
    _user_agent,
    asset_sha256,
    checksums_asset,
    dmg_filename,
    parse_sha256sums,
)

ProgressCb = Callable[[int, int], None]
StageCb = Callable[[str], None]


class SelfUpdateError(RuntimeError):
    """Download, checksum, or install failed."""


class SelfUpdateCancelled(SelfUpdateError):
    """User cancelled an in-app update."""


REPLACE_SCRIPT = r"""#!/bin/bash
set -euo pipefail
PID="$1"
DEST="$2"
SRC="$3"
LOG="${HOME}/Library/Logs/TrackLab-update.log"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) wait pid=$PID dest=$DEST"
for _ in $(seq 1 120); do
  if ! kill -0 "$PID" 2>/dev/null; then
    break
  fi
  sleep 0.25
done
sleep 0.4
if [[ ! -d "$SRC" ]]; then
  echo "missing new app $SRC"
  exit 1
fi
chmod -R u+w "$DEST" 2>/dev/null || true
rm -rf "$DEST"
/usr/bin/ditto "$SRC" "$DEST"
xattr -dr com.apple.quarantine "$DEST" 2>/dev/null || true
open "$DEST"
rm -rf "$(dirname "$SRC")"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) replaced"
"""


def cache_dir() -> Path:
    override = os.environ.get("TRACKLAB_UPDATE_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "tracklab" / "updates"
    return Path.home() / ".cache" / "tracklab" / "updates"


def _raise_if_cancelled(cancel: CancelToken | None) -> None:
    if cancel is not None and cancel.cancelled:
        raise SelfUpdateCancelled("已取消更新")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_url(
    url: str,
    dest: Path,
    *,
    current: str,
    progress: ProgressCb | None = None,
    cancel: CancelToken | None = None,
    timeout_s: float = 60.0,
) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    part.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": _user_agent(current),
            "Accept": "application/octet-stream",
        },
    )
    try:
        _raise_if_cancelled(cancel)
        with urllib.request.urlopen(
            request, timeout=timeout_s, context=_ssl_context()
        ) as response:
            total = int(response.headers.get("Content-Length") or 0)
            received = 0
            if progress is not None:
                progress(received, total)
            with part.open("wb") as fh:
                while True:
                    _raise_if_cancelled(cancel)
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    received += len(chunk)
                    if progress is not None:
                        progress(received, total)
        part.replace(dest)
    except SelfUpdateCancelled:
        part.unlink(missing_ok=True)
        dest.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001
        part.unlink(missing_ok=True)
        dest.unlink(missing_ok=True)
        raise SelfUpdateError(f"下载失败：{exc}") from exc


def find_bundled_app(mount: Path) -> Path:
    preferred = mount / "TrackLab.app"
    if (preferred / "Contents").is_dir():
        return preferred
    apps = [path for path in mount.glob("*.app") if (path / "Contents").is_dir()]
    if len(apps) == 1:
        return apps[0]
    raise SelfUpdateError("安装包里找不到 TrackLab.app")


def extract_app_from_dmg(
    dmg: Path,
    dest_app: Path,
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Path:
    if dest_app.exists():
        shutil.rmtree(dest_app)
    mount = Path(tempfile.mkdtemp(prefix="tracklab-dmg-"))
    attached = False
    try:
        result = run(
            [
                "hdiutil",
                "attach",
                "-nobrowse",
                "-readonly",
                "-mountpoint",
                str(mount),
                str(dmg),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise SelfUpdateError(f"无法打开安装包：{detail or result.returncode}")
        attached = True
        source = find_bundled_app(mount)
        dest_app.parent.mkdir(parents=True, exist_ok=True)
        copy = run(
            ["/usr/bin/ditto", str(source), str(dest_app)],
            check=False,
            capture_output=True,
            text=True,
        )
        if copy.returncode != 0:
            detail = (copy.stderr or copy.stdout or "").strip()
            raise SelfUpdateError(f"无法复制新应用：{detail or copy.returncode}")
        return dest_app
    finally:
        if attached:
            run(
                ["hdiutil", "detach", str(mount), "-quiet", "-force"],
                check=False,
                capture_output=True,
                text=True,
            )
        shutil.rmtree(mount, ignore_errors=True)


def write_replace_script(directory: Path) -> Path:
    path = directory / "replace.sh"
    path.write_text(REPLACE_SCRIPT, encoding="utf-8")
    path.chmod(0o755)
    return path


def launch_replacer(*, pid: int, bundle: Path, new_app: Path) -> None:
    script = write_replace_script(new_app.parent)
    subprocess.Popen(
        ["/bin/bash", str(script), str(pid), str(bundle), str(new_app)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )


def prepare_update(
    info: UpdateInfo,
    *,
    bundle: Path,
    current: str,
    machine: str | None = None,
    progress: ProgressCb | None = None,
    stage: StageCb | None = None,
    cancel: CancelToken | None = None,
    download: Callable[..., None] | None = None,
) -> Path:
    """Download and extract the new .app. Does not quit or replace yet."""
    if info.status is not UpdateStatus.AVAILABLE:
        raise SelfUpdateError(info.message or "没有可安装的新版本。")
    if not Path(bundle).exists():
        raise SelfUpdateError("找不到当前安装包。")
    asset = info.installer_for(machine)
    if asset is None:
        want = dmg_filename(machine) or "对应架构的 DMG"
        raise SelfUpdateError(f"此版本没有 {want}，请到网页下载。")
    sums_text = ""
    listed = checksums_asset(info.assets)
    fetch = download or _download_url
    work = cache_dir() / (info.latest or "update").lstrip("v")
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    if listed is not None:
        sums_path = work / listed.name
        if stage is not None:
            stage("checksums")
        fetch(
            listed.url,
            sums_path,
            current=current,
            progress=None,
            cancel=cancel,
        )
        sums_text = sums_path.read_text(encoding="utf-8")
    expected = asset_sha256(asset, parse_sha256sums(sums_text))
    if not expected:
        raise SelfUpdateError("安装包缺少 SHA-256 校验和，已取消自动更新。")
    dmg = work / asset.name
    if stage is not None:
        stage("download")
    fetch(
        asset.url,
        dmg,
        current=current,
        progress=progress,
        cancel=cancel,
    )
    _raise_if_cancelled(cancel)
    if stage is not None:
        stage("verify")
    actual = sha256_file(dmg)
    if actual != expected:
        dmg.unlink(missing_ok=True)
        raise SelfUpdateError("安装包校验失败，已删除下载文件。")
    if stage is not None:
        stage("extract")
    new_app = work / "TrackLab.app"
    extract_app_from_dmg(dmg, new_app)
    if not (new_app / "Contents" / "MacOS").is_dir():
        raise SelfUpdateError("解包后的应用不完整。")
    dmg.unlink(missing_ok=True)
    return new_app
