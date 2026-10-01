"""Download a GitHub DMG and replace the running macOS .app after quit."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

from ai.contracts import CancelToken
from app.patch_runtime import is_patch_path, runtime_fingerprint
from app.update_checker import (
    UpdateInfo,
    UpdateStatus,
    _ssl_context,
    _user_agent,
    asset_sha256,
    checksums_asset,
    dmg_filename,
    parse_sha256sums,
    release_download_url,
)

ProgressCb = Callable[[int, int], None]
StageCb = Callable[[str], None]


class SelfUpdateError(RuntimeError):
    """Download, checksum, or install failed."""


class SelfUpdateCancelled(SelfUpdateError):
    """User cancelled an in-app update."""


REPLACE_SCRIPT = r"""#!/bin/bash
set -u
PID="$1"
DEST="$2"
SRC="$3"
LOG="${HOME}/Library/Logs/TrackLab-update.log"
BACKUP_DIR="${HOME}/Library/Caches/tracklab/updates/backup"
DITTO="${TRACKLAB_DITTO:-/usr/bin/ditto}"
OPENER="${TRACKLAB_OPEN:-open}"
mkdir -p "$(dirname "$LOG")" "$BACKUP_DIR"
exec >>"$LOG" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) wait pid=$PID dest=$DEST"

alert() {
  local message="$1"
  if [[ -n "${TRACKLAB_ALERT_LOG:-}" ]]; then
    printf '%s\n' "$message" >>"$TRACKLAB_ALERT_LOG"
    return 0
  fi
  /usr/bin/osascript -e "display alert \"TrackLab 更新失败\" message \"$message\"" || true
}

fail() {
  local why="$1"
  echo "failed: $why"
  if [[ -n "${BACKUP:-}" && -d "$BACKUP" ]]; then
    rm -rf "$DEST"
    if ! mv "$BACKUP" "$DEST"; then
      echo "rollback failed"
    fi
  fi
  rm -rf "${DEST}.new"
  alert "更新失败，已恢复原版本。"
  exit 1
}

alive=1
LIMIT="${TRACKLAB_WAIT_TICKS:-600}"
for _ in $(seq 1 "$LIMIT"); do
  if ! kill -0 "$PID" 2>/dev/null; then
    alive=0
    break
  fi
  sleep 0.25
done
if [[ "$alive" -eq 1 ]]; then
  echo "old process still running"
  alert "更新失败，旧版本仍在运行。请完全退出 TrackLab 后再打开一次。"
  exit 1
fi
sleep 0.4
if [[ ! -d "$SRC" ]]; then
  echo "missing new app $SRC"
  alert "更新失败，已恢复原版本。"
  exit 1
fi
chmod -R u+w "$DEST" 2>/dev/null || true
rm -rf "${DEST}.new"
if ! "$DITTO" "$SRC" "${DEST}.new"; then
  rm -rf "${DEST}.new"
  fail "copy"
fi
BACKUP=""
if [[ -d "$DEST" ]]; then
  BACKUP="$BACKUP_DIR/TrackLab-$(date -u +%Y%m%dT%H%M%SZ).app"
  if ! mv "$DEST" "$BACKUP"; then
    rm -rf "${DEST}.new"
    BACKUP=""
    fail "backup"
  fi
fi
if ! mv "${DEST}.new" "$DEST"; then
  fail "install"
fi
xattr -dr com.apple.quarantine "$DEST" 2>/dev/null || true
if ! "$OPENER" "$DEST"; then
  fail "open"
fi
if [[ -n "$BACKUP" ]]; then
  rm -rf "$BACKUP"
fi
rm -rf "$(dirname "$SRC")"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) replaced"
"""


def update_log_path() -> Path:
    return Path.home() / "Library" / "Logs" / "TrackLab-update.log"


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
    args = [str(script), str(pid), str(bundle), str(new_app)]
    log = update_log_path()
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{_utc_now()} launch pid={pid} dest={bundle} src={new_app}\n"
            )
    except OSError:
        pass
    # The script must outlive TrackLab. A child is killed when this process
    # crashes, which is what left the old app in place.
    _spawn_detached(["/bin/bash", *args])


def _utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _spawn_detached(argv: list[str]) -> None:
    import shlex

    command = "nohup " + " ".join(shlex.quote(part) for part in argv) + " >/dev/null 2>&1 &"
    # AppleScript strings use double quotes. Shell quotes here are a syntax error,
    # so the replacer never started and the old app stayed in place.
    source = "do shell script " + _applescript_string(command)
    try:
        subprocess.Popen(
            ["/usr/bin/osascript", "-e", source],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        return
    except OSError:
        pass
    subprocess.Popen(
        argv,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )


def _applescript_string(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + escaped + '"'


def _safe_patch_member(name: str) -> str | None:
    text = name.replace("\\", "/").lstrip("/")
    if not text or text.endswith("/") or ".." in text.split("/"):
        return None
    if not is_patch_path(text):
        return None
    return text


def apply_patch_zip(bundle: Path, archive_path: Path) -> None:
    """Overlay a code patch onto a copy of the installed app."""
    wrote = False
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            relative = _safe_patch_member(info.filename)
            if relative is None or info.is_dir():
                continue
            dest = bundle / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(info))
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode:
                dest.chmod(mode)
            elif relative == "Contents/MacOS/TrackLab":
                dest.chmod(dest.stat().st_mode | stat.S_IXUSR)
            wrote = True
    if not wrote or not (bundle / "Contents" / "MacOS" / "TrackLab").is_file():
        raise SelfUpdateError("补丁里没有可执行文件。")


def codesign_app(
    bundle: Path,
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    result = run(
        ["codesign", "--force", "--deep", "--sign", "-", "--timestamp=none", str(bundle)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise SelfUpdateError(f"无法签名新应用：{detail or result.returncode}")


def _copy_bundle(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["/usr/bin/ditto", str(source), str(dest)],
        check=True,
        capture_output=True,
        text=True,
    )


def _try_patch(
    info: UpdateInfo,
    *,
    bundle: Path,
    current: str,
    machine: str | None,
    progress: ProgressCb | None,
    stage: StageCb | None,
    cancel: CancelToken | None,
    download: Callable[..., None] | None,
    sign: Callable[[Path], None] | None,
    copy_bundle: Callable[[Path, Path], None] | None,
) -> Path | None:
    spec = info.patch_for(machine)
    if spec is None:
        return None
    try:
        local = runtime_fingerprint(bundle)
    except OSError:
        return None
    if local != spec.runtime:
        return None
    tag = info.latest or ""
    url = release_download_url(tag, spec.asset)
    if url is None:
        return None
    fetch = download or _download_url
    work = cache_dir() / (info.latest or "update").lstrip("v") / "patch"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    archive_path = work / spec.asset
    new_app = work / "TrackLab.app"
    try:
        if stage is not None:
            stage("download")
        fetch(
            url,
            archive_path,
            current=current,
            progress=progress,
            cancel=cancel,
        )
        _raise_if_cancelled(cancel)
        if stage is not None:
            stage("verify")
        if sha256_file(archive_path) != spec.sha256:
            raise SelfUpdateError("补丁校验失败，已删除下载文件。")
        if stage is not None:
            stage("extract")
        (copy_bundle or _copy_bundle)(bundle, new_app)
        apply_patch_zip(new_app, archive_path)
        if sign is None:
            codesign_app(new_app)
        else:
            sign(new_app)
    except SelfUpdateCancelled:
        shutil.rmtree(work, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        return None
    archive_path.unlink(missing_ok=True)
    return new_app


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
    sign: Callable[[Path], None] | None = None,
    copy_bundle: Callable[[Path, Path], None] | None = None,
) -> Path:
    """Download and extract the new .app. Does not quit or replace yet."""
    if info.status is not UpdateStatus.AVAILABLE:
        raise SelfUpdateError(info.message or "没有可安装的新版本。")
    if not Path(bundle).exists():
        raise SelfUpdateError("找不到当前安装包。")
    patched = _try_patch(
        info,
        bundle=Path(bundle),
        current=current,
        machine=machine,
        progress=progress,
        stage=stage,
        cancel=cancel,
        download=download,
        sign=sign,
        copy_bundle=copy_bundle,
    )
    if patched is not None:
        return patched
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
