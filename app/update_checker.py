"""GitHub Releases update detection. Network I/O is injectable; no Qt here."""

from __future__ import annotations

import json
import os
import socket
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Sequence

from app import GITHUB_REPO, __version__

GITHUB_API_LATEST = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
GITHUB_RELEASES_PAGE = f"https://github.com/{GITHUB_REPO}/releases/latest"
ALLOWED_DOWNLOAD_HOSTS = {"github.com", "www.github.com"}
CHECK_INTERVAL_S = 24 * 3600
DEFAULT_TIMEOUT_S = 8.0

SETTINGS_AUTO_CHECK = "update/auto_check"
SETTINGS_LAST_CHECK = "update/last_check"
SETTINGS_SKIPPED = "update/skipped_version"

Transport = Callable[[str, dict[str, str], float], tuple[int, Mapping[str, str], bytes]]


class UpdateStatus(str, Enum):
    LATEST = "latest"
    AVAILABLE = "available"
    SKIPPED = "skipped"
    NO_RELEASE = "no_release"
    RATE_LIMITED = "rate_limited"
    OFFLINE = "offline"
    TIMEOUT = "timeout"
    INVALID = "invalid"
    ERROR = "error"


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    url: str
    size: int = 0
    digest: str = ""


@dataclass(frozen=True)
class UpdateInfo:
    status: UpdateStatus
    current: str
    latest: str | None = None
    html_url: str | None = None
    notes: str = ""
    published_at: str = ""
    message: str = ""
    assets: tuple[ReleaseAsset, ...] = field(default_factory=tuple)

    @property
    def download_url(self) -> str:
        return safe_release_url(self.html_url) or GITHUB_RELEASES_PAGE

    @property
    def installer(self) -> ReleaseAsset | None:
        return self.installer_for()

    def installer_for(self, machine: str | None = None) -> ReleaseAsset | None:
        want = dmg_filename(machine)
        if not want:
            return None
        for asset in self.assets:
            if asset.name == want:
                return asset
        return None


def parse_version(tag: str) -> tuple[int, int, int] | None:
    """Strict `vMAJOR.MINOR.PATCH` (leading v optional)."""
    text = str(tag or "").strip()
    if text.startswith(("v", "V")):
        text = text[1:]
    parts = text.split(".")
    if len(parts) != 3:
        return None
    try:
        major, minor, patch = (int(part) for part in parts)
    except ValueError:
        return None
    if major < 0 or minor < 0 or patch < 0:
        return None
    return major, minor, patch


def is_newer(remote: str, current: str) -> bool:
    parsed_remote = parse_version(remote)
    parsed_current = parse_version(current)
    if parsed_remote is None or parsed_current is None:
        return False
    return parsed_remote > parsed_current


def should_auto_check(
    last_check_ts: float,
    now: float,
    *,
    interval_s: float = CHECK_INTERVAL_S,
) -> bool:
    if last_check_ts <= 0:
        return True
    return (now - last_check_ts) >= interval_s


def update_checks_allowed() -> bool:
    flag = os.environ.get("TRACKLAB_SKIP_UPDATE_CHECK", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return False
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return False
    if "--smoke" in sys.argv:
        return False
    return True


def format_published_at(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return text


def dmg_filename(machine: str | None = None) -> str | None:
    """Installer name shipped by ``release-macos.yml`` for this CPU."""
    kind = (machine or os.uname().machine).strip().lower()
    if kind in {"arm64", "aarch64"}:
        return "TrackLab-arm64.dmg"
    if kind in {"x86_64", "amd64"}:
        return "TrackLab-x86_64.dmg"
    return None


def parse_sha256sums(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        digest, name = parts[0], parts[-1]
        if name.startswith("*"):
            name = name[1:]
        name = name.rsplit("/", 1)[-1]
        if len(digest) != 64:
            continue
        if any(char not in "0123456789abcdefABCDEF" for char in digest):
            continue
        result[name] = digest.lower()
    return result


def parse_release_assets(payload: Mapping[str, object]) -> tuple[ReleaseAsset, ...]:
    raw = payload.get("assets") or []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    assets: list[ReleaseAsset] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").strip()
        url = safe_release_url(str(item.get("browser_download_url") or "") or None)
        if not name or not url:
            continue
        try:
            size = int(item.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        assets.append(
            ReleaseAsset(
                name=name,
                url=url,
                size=max(0, size),
                digest=str(item.get("digest") or "").strip(),
            )
        )
    return tuple(assets)


def asset_sha256(asset: ReleaseAsset, sums: Mapping[str, str] | None = None) -> str | None:
    digest = asset.digest.strip()
    lowered = digest.lower()
    if lowered.startswith("sha256:"):
        value = digest.split(":", 1)[1].strip().lower()
        if len(value) == 64:
            return value
    if sums and asset.name in sums:
        return sums[asset.name]
    return None


def checksums_asset(assets: Sequence[ReleaseAsset]) -> ReleaseAsset | None:
    for asset in assets:
        if asset.name.upper() == "SHA256SUMS.TXT":
            return asset
    return None


def can_self_update(
    info: UpdateInfo,
    *,
    frozen: bool | None = None,
    system: str | None = None,
    machine: str | None = None,
    bundle=None,  # noqa: ANN001
) -> bool:
    """True when the packaged macOS app can download and replace itself."""
    if info.status is not UpdateStatus.AVAILABLE:
        return False
    want = dmg_filename(machine)
    if not want or not any(asset.name == want for asset in info.assets):
        return False
    if system is None:
        system = sys.platform
    if system != "darwin":
        return False
    if frozen is None:
        from app.paths import is_frozen

        frozen = is_frozen()
    if not frozen:
        return False
    if bundle is None:
        from app.paths import frozen_app_bundle

        bundle = frozen_app_bundle()
    if bundle is None:
        return False
    try:
        return os.access(str(bundle.parent), os.W_OK)
    except OSError:
        return False


def safe_release_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urllib.parse.urlparse(str(url).strip())
    host = parsed.netloc.lower()
    if parsed.scheme != "https" or host not in ALLOWED_DOWNLOAD_HOSTS:
        return None
    return urllib.parse.urlunparse(parsed)


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _default_transport(url: str, headers: dict[str, str], timeout_s: float) -> tuple[int, Mapping[str, str], bytes]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s, context=_ssl_context()) as response:
            return int(response.status), dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        body = b""
        try:
            body = exc.read()
        except Exception:
            body = b""
        headers_map: dict[str, str] = {}
        if exc.headers is not None:
            headers_map = dict(exc.headers.items())
        return int(exc.code), headers_map, body


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, socket.timeout):
        return True
    reason = getattr(exc, "reason", None)
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return True
    text = str(reason or exc).lower()
    return "timed out" in text or "timeout" in text


def _user_agent(current: str) -> str:
    return f"TrackLab/{current} (+https://github.com/{GITHUB_REPO})"


def evaluate_release(
    payload: Mapping[str, object],
    current: str,
    *,
    skipped: str = "",
    honor_skip: bool = True,
) -> UpdateInfo:
    assets = parse_release_assets(payload)
    common = {
        "html_url": safe_release_url(str(payload.get("html_url") or "") or None),
        "notes": str(payload.get("body") or "").strip(),
        "published_at": format_published_at(str(payload.get("published_at") or "")),
        "assets": assets,
    }
    if payload.get("draft") or payload.get("prerelease"):
        return UpdateInfo(
            status=UpdateStatus.NO_RELEASE,
            current=current,
            message="尚未发布正式版本。",
            **common,
        )
    tag = str(payload.get("tag_name") or "").strip()
    parsed = parse_version(tag)
    if parsed is None:
        return UpdateInfo(
            status=UpdateStatus.INVALID,
            current=current,
            latest=tag or None,
            message="远端版本号无效。",
            **common,
        )
    if not is_newer(tag, current):
        return UpdateInfo(
            status=UpdateStatus.LATEST,
            current=current,
            latest=tag,
            message=f"当前已是最新版本 {tag}。",
            **common,
        )
    skipped_parsed = parse_version(skipped) if skipped else None
    if honor_skip and skipped_parsed is not None and skipped_parsed == parsed:
        return UpdateInfo(
            status=UpdateStatus.SKIPPED,
            current=current,
            latest=tag,
            message=f"已跳过版本 {tag}。",
            **common,
        )
    return UpdateInfo(
        status=UpdateStatus.AVAILABLE,
        current=current,
        latest=tag,
        message=f"发现新版本 {tag}。",
        **common,
    )


def check_for_update(
    current: str = __version__,
    *,
    skipped: str = "",
    honor_skip: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    transport: Transport | None = None,
) -> UpdateInfo:
    headers = {
        "User-Agent": _user_agent(current),
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    fetch = transport or _default_transport
    try:
        status, response_headers, body = fetch(GITHUB_API_LATEST, headers, timeout_s)
    except Exception as exc:  # noqa: BLE001
        if _is_timeout(exc):
            return UpdateInfo(
                status=UpdateStatus.TIMEOUT,
                current=current,
                message="检查更新超时。",
            )
        return UpdateInfo(
            status=UpdateStatus.OFFLINE,
            current=current,
            message="无法连接网络，已跳过更新检查。",
        )

    if status in {403, 429}:
        return UpdateInfo(
            status=UpdateStatus.RATE_LIMITED,
            current=current,
            message="GitHub 接口暂时限流，请稍后再试。",
        )
    if status == 404:
        return UpdateInfo(
            status=UpdateStatus.NO_RELEASE,
            current=current,
            message="尚未发布正式版本。",
        )
    if status != 200:
        remaining = str(response_headers.get("X-RateLimit-Remaining") or "")
        if remaining == "0":
            return UpdateInfo(
                status=UpdateStatus.RATE_LIMITED,
                current=current,
                message="GitHub 接口暂时限流，请稍后再试。",
            )
        return UpdateInfo(
            status=UpdateStatus.ERROR,
            current=current,
            message=f"检查更新失败（HTTP {status}）。",
        )
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return UpdateInfo(
            status=UpdateStatus.INVALID,
            current=current,
            message="远端版本信息无法解析。",
        )
    if not isinstance(payload, dict):
        return UpdateInfo(
            status=UpdateStatus.INVALID,
            current=current,
            message="远端版本信息无法解析。",
        )
    return evaluate_release(payload, current, skipped=skipped, honor_skip=honor_skip)


def status_bar_message(info: UpdateInfo) -> str:
    return info.message or {
        UpdateStatus.LATEST: "当前已是最新版本。",
        UpdateStatus.AVAILABLE: "发现新版本。",
        UpdateStatus.SKIPPED: "已跳过此版本。",
        UpdateStatus.NO_RELEASE: "尚未发布正式版本。",
        UpdateStatus.RATE_LIMITED: "GitHub 接口暂时限流。",
        UpdateStatus.OFFLINE: "无法连接网络。",
        UpdateStatus.TIMEOUT: "检查更新超时。",
        UpdateStatus.INVALID: "远端版本号无效。",
        UpdateStatus.ERROR: "检查更新失败。",
    }[info.status]
