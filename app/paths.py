"""Locate bundled resources in source trees and frozen PyInstaller apps."""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    """Directory that contains collected `datas` (source repo or `_MEIPASS`)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if is_frozen() and meipass:
        return Path(meipass)
    return Path(__file__).resolve().parents[1]


def app_dir() -> Path:
    if is_frozen():
        return resource_root() / "app"
    return Path(__file__).resolve().parent


def style_path() -> Path:
    return app_dir() / "style.qss"


def bundled_model_dir() -> Path | None:
    """Frozen app directory for shipped SAM weights, if present."""
    if not is_frozen():
        return None
    path = resource_root() / "models"
    return path if path.is_dir() else None
