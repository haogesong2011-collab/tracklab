"""Optional ChArUco helpers. OpenCV is not a required TrackLab dependency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai.calibration import CameraProfile
from ai.schema import Point2D

DEFAULT_SQUARES = (5, 4)
DEFAULT_SQUARE_M = 0.04
DEFAULT_MARKER_M = 0.03


@dataclass
class CharucoDetection:
    corners: list[Point2D]
    width_m: float
    height_m: float
    ids: list[int]
    reprojection_rms_px: float = 0.0


def opencv_available() -> bool:
    try:
        import cv2  # noqa: F401
    except Exception:
        return False
    return True


def detect_plane_from_frame(
    rgb: np.ndarray,
    *,
    squares: tuple[int, int] = DEFAULT_SQUARES,
    square_m: float = DEFAULT_SQUARE_M,
    marker_m: float = DEFAULT_MARKER_M,
    camera: CameraProfile | None = None,
) -> tuple[CharucoDetection | None, str]:
    board, gray, err = _board_and_gray(rgb, squares, square_m, marker_m)
    if board is None or gray is None:
        return None, err
    pts, ids, message = _detect_charuco(gray, board, camera)
    if pts is None or ids is None or len(pts) < 4:
        return None, message or "当前帧没有检测到足够的棋盘格角点"
    rectangle = _ordered_plane_corners(pts, ids, squares)
    width_m = (squares[0] - 1) * square_m
    height_m = (squares[1] - 1) * square_m
    return (
        CharucoDetection(
            corners=rectangle,
            width_m=width_m,
            height_m=height_m,
            ids=[int(i) for i in ids.reshape(-1)],
        ),
        "",
    )


def collect_calibration_view(
    rgb: np.ndarray,
    *,
    squares: tuple[int, int] = DEFAULT_SQUARES,
    square_m: float = DEFAULT_SQUARE_M,
    marker_m: float = DEFAULT_MARKER_M,
) -> tuple[object, object, str]:
    board, gray, err = _board_and_gray(rgb, squares, square_m, marker_m)
    if board is None or gray is None:
        return None, None, err
    pts, ids, message = _detect_charuco(gray, board, None)
    if pts is None or ids is None or len(pts) < 6:
        return None, None, message or "角点太少，换一个角度再采集"
    return pts, ids, ""


def calibrate_camera(
    views: list[tuple[object, object]],
    image_size: tuple[int, int],
    *,
    squares: tuple[int, int] = DEFAULT_SQUARES,
    square_m: float = DEFAULT_SQUARE_M,
    marker_m: float = DEFAULT_MARKER_M,
) -> tuple[CameraProfile | None, str]:
    if not opencv_available():
        return None, "未安装 OpenCV，无法标定镜头。可选：python -m pip install opencv-contrib-python"
    if len(views) < 3:
        return None, "镜头标定至少需要 3 张不同角度的棋盘格画面"
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        return None, f"无法导入 OpenCV：{exc}"
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(squares, square_m, marker_m, dictionary)
    all_corners = [item[0] for item in views]
    all_ids = [item[1] for item in views]
    try:
        rms, camera_matrix, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, image_size, None, None
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"镜头标定失败：{exc}"
    dist = np.array(dist, dtype=np.float64).reshape(-1)
    k = np.array(camera_matrix, dtype=np.float64)
    profile = CameraProfile(
        width=int(image_size[0]),
        height=int(image_size[1]),
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
        k1=float(dist[0]) if len(dist) > 0 else 0.0,
        k2=float(dist[1]) if len(dist) > 1 else 0.0,
        p1=float(dist[2]) if len(dist) > 2 else 0.0,
        p2=float(dist[3]) if len(dist) > 3 else 0.0,
        k3=float(dist[4]) if len(dist) > 4 else 0.0,
        rms=float(rms),
    )
    return profile, ""


def _board_and_gray(
    rgb: np.ndarray,
    squares: tuple[int, int],
    square_m: float,
    marker_m: float,
):
    if not opencv_available():
        return None, None, "未安装 OpenCV。可选：python -m pip install opencv-contrib-python"
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        return None, None, f"无法导入 OpenCV：{exc}"
    if rgb.ndim != 3:
        return None, None, "图像格式无效"
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(squares, square_m, marker_m, dictionary)
    return board, gray, ""


def _detect_charuco(gray, board, camera: CameraProfile | None):
    import cv2

    camera_matrix = None if camera is None else camera.camera_matrix()
    dist = None
    if camera is not None and camera.has_intrinsics:
        dist = np.array([[camera.k1, camera.k2, camera.p1, camera.p2, camera.k3]], dtype=np.float64)
    if hasattr(cv2.aruco, "CharucoDetector"):
        detector = cv2.aruco.CharucoDetector(board)
        corners, ids, _, _ = detector.detectBoard(gray)
        if corners is None or ids is None:
            return None, None, "未检测到 ChArUco 棋盘"
        return np.array(corners).reshape(-1, 2), np.array(ids).reshape(-1), ""
    dictionary = board.getDictionary() if hasattr(board, "getDictionary") else None
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if marker_ids is None or len(marker_ids) < 4:
        return None, None, "未检测到足够的 ArUco 标记"
    count, corners, ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board, cameraMatrix=camera_matrix, distCoeffs=dist
    )
    if count < 4 or corners is None or ids is None:
        return None, None, "棋盘格角点不足"
    return np.array(corners).reshape(-1, 2), np.array(ids).reshape(-1), ""


def _ordered_plane_corners(
    pts: np.ndarray, ids: np.ndarray, squares: tuple[int, int]
) -> list[Point2D]:
    nx = max(int(squares[0]) - 1, 1)
    ny = max(int(squares[1]) - 1, 1)
    wanted = [0, nx - 1, (ny - 1) * nx + (nx - 1), (ny - 1) * nx]
    by_id = {int(item): pts[index] for index, item in enumerate(np.asarray(ids).reshape(-1))}
    if all(cid in by_id for cid in wanted):
        return [
            Point2D(float(by_id[cid][0]), float(by_id[cid][1])) for cid in wanted
        ]
    xs, ys = pts[:, 0], pts[:, 1]
    return [
        Point2D(float(xs.min()), float(ys.max())),
        Point2D(float(xs.max()), float(ys.max())),
        Point2D(float(xs.max()), float(ys.min())),
        Point2D(float(xs.min()), float(ys.min())),
    ]
