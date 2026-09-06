# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the macOS TrackLab.app (fast + precise SAM 2)."""

from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
sys.path.insert(0, str(ROOT))

from ai.model_manager import DEFAULT_SPEC, checkpoint_path  # noqa: E402
from app import BUNDLE_IDENTIFIER, __version__  # noqa: E402

ICON = ROOT / "macos-packaging" / "TrackLab.icns"

# MoGe / OpenCV stay out of the installer. Torch + SAM 2 are required.
EXCLUDES = [
    "moge",
    "cv2",
    "tkinter",
    "matplotlib",
    "IPython",
    "notebook",
    "pytest",
    "scipy",
    "pandas",
    "PIL",
    "openai",
    "torchaudio",
    "PySide6.QtWebEngine",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DExtras",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtPositioning",
    "PySide6.QtLocation",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickControls2",
    "PySide6.QtQuickWidgets",
    "PySide6.QtQml",
    "PySide6.QtRemoteObjects",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtTextToSpeech",
    "PySide6.QtWebChannel",
    "PySide6.QtWebSockets",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
    "PySide6.QtHttpServer",
    "PySide6.QtSpatialAudio",
    "PySide6.QtStateMachine",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    "PySide6.QtUiTools",
]

DROP_HINTS = (
    "WebEngine",
    "Qt3D",
    "QtQuick",
    "QtQml",
    "QtPdf",
    "QtMultimedia",
    "QtDesigner",
    "QtSql",
    "QtBluetooth",
    "QtNfc",
    "QtPositioning",
    "QtLocation",
    "QtSensors",
    "QtSerialPort",
    "QtTextToSpeech",
    "QtWebChannel",
    "QtWebSockets",
    "QtSvg",
)

datas = [(str(ROOT / "app" / "style.qss"), "app")]
binaries = []
hiddenimports = [
    "PySide6.QtCharts",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtNetwork",
    "shiboken6",
    "av",
    "numpy",
    "certifi",
    "keyring",
    "keyring.backends.macOS",
    "keyring.backends.fail",
    "keyring.backends.null",
    "torch",
    "torchvision",
    "sam2",
    "hydra",
    "omegaconf",
    "iopath",
    "app.paths",
    "app.update_checker",
    "app.update_dialog",
    "app.download_toast",
    "app.chart_ticks",
    "engine.decoder",
    "engine.video_index",
    "ai.sam_runtime",
    "ai.plane",
    "ai.charuco",
    "ai.depth_audit",
    "ai.autotracker",
]
hiddenimports += collect_submodules("app")
hiddenimports += collect_submodules("engine")
hiddenimports += collect_submodules("ai")


def _keep(item) -> bool:  # noqa: ANN001
    name = item[0] if isinstance(item, (tuple, list)) else str(item)
    return not any(hint in str(name) for hint in DROP_HINTS)


for pkg in (
    "PySide6",
    "av",
    "numpy",
    "certifi",
    "keyring",
    "torch",
    "torchvision",
    "sam2",
    "hydra",
    "omegaconf",
    "iopath",
):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += [entry for entry in pkg_datas if _keep(entry)]
    binaries += [entry for entry in pkg_binaries if _keep(entry)]
    hiddenimports += [entry for entry in pkg_hidden if _keep(entry)]

ckpt = checkpoint_path(DEFAULT_SPEC)
if not ckpt.is_file():
    ckpt = Path.home() / ".cache" / "tracklab" / "models" / DEFAULT_SPEC.filename
if ckpt.is_file():
    datas.append((str(ckpt), "models"))
else:
    raise SystemExit(
        f"missing SAM 2 checkpoint {ckpt}; run the app once or "
        "python -c \"from ai.model_manager import ensure_checkpoint; ensure_checkpoint(download=True)\""
    )

icon = str(ICON) if ICON.is_file() else None

a = Analysis(
    [str(ROOT / "app" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TrackLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    icon=icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TrackLab",
)

app = BUNDLE(
    coll,
    name="TrackLab.app",
    icon=icon,
    bundle_identifier=BUNDLE_IDENTIFIER,
    info_plist={
        "CFBundleName": "TrackLab",
        "CFBundleDisplayName": "TrackLab",
        "CFBundleGetInfoString": "TrackLab",
        "CFBundleIdentifier": BUNDLE_IDENTIFIER,
        "CFBundleVersion": __version__,
        "CFBundleShortVersionString": __version__,
        "NSPrincipalClass": "NSApplication",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "13.0",
        "LSApplicationCategoryType": "public.app-category.education",
        "NSHumanReadableCopyright": "Copyright © 2026 TrackLab",
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "Movie",
                "CFBundleTypeRole": "Viewer",
                "LSHandlerRank": "Alternate",
                "CFBundleTypeExtensions": [
                    "mp4",
                    "mov",
                    "m4v",
                    "avi",
                    "mkv",
                    "webm",
                    "mpg",
                    "mpeg",
                ],
            }
        ],
    },
)
