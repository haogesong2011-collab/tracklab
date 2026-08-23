"""DeepSeek API key storage. The key never enters project JSON or reports."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SERVICE_NAME = "TrackLab.DeepSeek"
USERNAME = "api_key"
ENV_NAME = "DEEPSEEK_API_KEY"
CONFIG_DIR_ENV = "TRACKLAB_CONFIG_DIR"
KEY_FILENAME = "deepseek_api_key"

_session_key: str | None = None


def normalize_api_key(key: str | None) -> str:
    """Keep only the ASCII token. Strips Chinese labels copied from the console."""
    if not key:
        return ""
    cleaned = key.strip().strip("\"'").replace("\ufeff", "")
    tokens = cleaned.replace("\n", " ").replace("\r", " ").split()
    for token in tokens:
        if token.startswith("sk-") and token.isascii():
            return token
    ascii_only = "".join(ch for ch in cleaned if 32 < ord(ch) < 127)
    return ascii_only.strip()


def config_dir() -> Path:
    override = os.environ.get(CONFIG_DIR_ENV, "").strip()
    if override:
        return Path(override)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "TrackLab"
    if sys.platform == "win32":
        root = os.environ.get("APPDATA") or str(Path.home())
        return Path(root) / "TrackLab"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "tracklab"
    return Path.home() / ".config" / "tracklab"


def get_api_key() -> str | None:
    env = normalize_api_key(os.environ.get(ENV_NAME, ""))
    if env:
        return env
    stored = _keyring_get()
    if stored:
        return stored
    stored = _file_get()
    if stored:
        return stored
    if _session_key:
        return _session_key
    return None


def set_api_key(key: str, *, persist: bool = True) -> bool:
    """Save the key. Returns True if it will survive a restart."""
    global _session_key
    cleaned = normalize_api_key(key)
    _session_key = cleaned or None
    if not persist:
        return False
    if not cleaned:
        return clear_api_key()
    if _keyring_set(cleaned):
        return True
    return _file_set(cleaned)


def clear_api_key() -> bool:
    global _session_key
    _session_key = None
    keyring_ok = _keyring_delete()
    file_ok = _file_delete()
    return keyring_ok and file_ok


def has_api_key() -> bool:
    return get_api_key() is not None


def mask_key(key: str | None) -> str:
    if not key:
        return "未配置"
    trimmed = key.strip()
    if len(trimmed) <= 4:
        return "••••"
    return "••••" + trimmed[-4:]


def _key_file() -> Path:
    return config_dir() / KEY_FILENAME


def _file_get() -> str | None:
    path = _key_file()
    try:
        if not path.is_file():
            return None
        return normalize_api_key(path.read_text(encoding="utf-8")) or None
    except OSError:
        return None


def _file_set(key: str) -> bool:
    path = _key_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key + "\n", encoding="utf-8")
        os.chmod(path, 0o600)
        return True
    except OSError:
        return False


def _file_delete() -> bool:
    path = _key_file()
    try:
        if path.is_file():
            path.unlink()
        return True
    except OSError:
        return False


def _use_keyring() -> bool:
    return not os.environ.get(CONFIG_DIR_ENV, "").strip()


def _keyring_get() -> str | None:
    if not _use_keyring():
        return None
    try:
        import keyring
    except Exception:
        return None
    try:
        value = keyring.get_password(SERVICE_NAME, USERNAME)
    except Exception:
        return None
    if not value:
        return None
    return normalize_api_key(value) or None


def _keyring_set(key: str) -> bool:
    if not _use_keyring():
        return False
    try:
        import keyring
        keyring.set_password(SERVICE_NAME, USERNAME, key)
        return True
    except Exception:
        return False


def _keyring_delete() -> bool:
    if not _use_keyring():
        return True
    try:
        import keyring
        keyring.delete_password(SERVICE_NAME, USERNAME)
        return True
    except Exception:
        return False
