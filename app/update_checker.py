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
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping

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
class UpdateInfo:
    status: UpdateStatus
    current: str
    latest: str | None = None
    html_url: str | None = None
    notes: str = ""
    published_at: str = ""
    message: str = ""

    @property
    def download_url(self) -> str:
        return safe_release_url(self.html_url) or GITHUB_RELEASES_PAGE


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
    if payload.get("draft") or payload.get("prerelease"):
        return UpdateInfo(
            status=UpdateStatus.NO_RELEASE,
            current=current,
            message="尚未发布正式版本。",
        )
    tag = str(payload.get("tag_name") or "").strip()
    parsed = parse_version(tag)
    html_url = safe_release_url(str(payload.get("html_url") or "") or None)
    notes = str(payload.get("body") or "").strip()
    published = format_published_at(str(payload.get("published_at") or ""))
    if parsed is None:
        return UpdateInfo(
            status=UpdateStatus.INVALID,
            current=current,
            latest=tag or None,
            html_url=html_url,
            notes=notes,
            published_at=published,
            message="远端版本号无效。",
        )
    if not is_newer(tag, current):
        return UpdateInfo(
            status=UpdateStatus.LATEST,
            current=current,
            latest=tag,
            html_url=html_url,
            notes=notes,
            published_at=published,
            message=f"当前已是最新版本 {tag}。",
        )
    skipped_parsed = parse_version(skipped) if skipped else None
    if honor_skip and skipped_parsed is not None and skipped_parsed == parsed:
        return UpdateInfo(
            status=UpdateStatus.SKIPPED,
            current=current,
            latest=tag,
            html_url=html_url,
            notes=notes,
            published_at=published,
            message=f"已跳过版本 {tag}。",
        )
    return UpdateInfo(
        status=UpdateStatus.AVAILABLE,
        current=current,
        latest=tag,
        html_url=html_url,
        notes=notes,
        published_at=published,
        message=f"发现新版本 {tag}。",
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
