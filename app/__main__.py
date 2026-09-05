import os
import sys
from pathlib import Path


def _print_version() -> int:
    from app import __version__

    print(__version__)
    return 0


def _smoke_check(window) -> int:  # noqa: ANN001
    from ai.calibration import CalibrationMode, planar_state
    from ai.contracts import TrackMode
    from ai.desktop import create_tracker
    from ai.schema import Point2D
    from app import __version__
    from app.paths import is_frozen, style_path

    if not window.styleSheet():
        print("SMOKE FAIL: stylesheet empty", file=sys.stderr)
        return 1
    if not style_path().is_file():
        print(f"SMOKE FAIL: missing stylesheet {style_path()}", file=sys.stderr)
        return 1
    create_tracker(TrackMode.FAST)
    state = planar_state(
        [Point2D(0, 0), Point2D(400, 0), Point2D(400, 300), Point2D(0, 300)],
        width_m=1.0,
        height_m=0.75,
    )
    if state.mode is not CalibrationMode.PLANAR:
        print(f"SMOKE FAIL: planar calibration {state.warning}", file=sys.stderr)
        return 1
    video = os.environ.get("TRACKLAB_SMOKE_VIDEO", "").strip()
    if video:
        from engine.decoder import FrameDecoder
        from engine.video_index import index_video

        info = index_video(Path(video))
        decoder = FrameDecoder(info)
        try:
            frame = decoder.frame(0)
        finally:
            decoder.close()
        if frame is None or getattr(frame, "size", 0) == 0:
            print("SMOKE FAIL: could not decode sample video", file=sys.stderr)
            return 1
    exe = Path(sys.executable)
    if is_frozen():
        text = str(exe).replace("\\", "/")
        if "Contents/MacOS" not in text:
            print(f"SMOKE FAIL: frozen executable not in app bundle: {exe}", file=sys.stderr)
            return 1
        joined = os.pathsep.join(sys.path)
        if "/.venv/" in joined or joined.endswith(".venv"):
            print("SMOKE FAIL: virtualenv appeared on frozen sys.path", file=sys.stderr)
            return 1
    print(
        f"SMOKE OK version={__version__} frozen={is_frozen()} exe={exe}",
        flush=True,
    )
    return 0


def main() -> None:
    if "--version" in sys.argv:
        raise SystemExit(_print_version())
    from PySide6.QtWidgets import QApplication

    from app import __version__
    from app.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("TrackLab")
    app.setOrganizationName("TrackLab")
    app.setOrganizationDomain("tracklab.app")
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")
    window = MainWindow()
    if "--smoke" in sys.argv:
        code = _smoke_check(window)
        window.close()
        raise SystemExit(code)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
