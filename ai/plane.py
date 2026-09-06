"""Planar homography, undistortion, and local error propagation.

Pixel ↔ plane mapping uses a normalised DLT homography. Camera intrinsics are
optional: when absent the homography still works, but depth and undistortion
degrade to a pinhole guess for quality checks only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ai.schema import Point2D

MIN_QUAD_AREA_PX = 400.0
MIN_INTERIOR_ANGLE_DEG = 12.0
MAX_REPROJ_RMS_PX = 4.0
WARN_REPROJ_RMS_PX = 1.5
MIN_COVERAGE = 0.04
DEFAULT_PIXEL_SIGMA = 0.5
HOMOGRAPHY_EPS = 1e-12


@dataclass(frozen=True)
class HomographyFit:
    matrix: np.ndarray
    inverse: np.ndarray
    reprojection_rms_px: float
    area_px: float
    min_angle_deg: float
    condition: float


@dataclass(frozen=True)
class PlanePose:
    rotation: np.ndarray
    translation: np.ndarray
    camera_matrix: np.ndarray
    assumed_intrinsics: bool


def homography_pixel_to_world(
    pixels: list[Point2D],
    world: list[Point2D],
) -> HomographyFit:
    if len(pixels) < 4 or len(world) < 4:
        raise ValueError("平面标定至少需要四个对应点")
    src = np.array([[p.x, p.y] for p in pixels[:4]], dtype=np.float64)
    dst = np.array([[p.x, p.y] for p in world[:4]], dtype=np.float64)
    H = _dlt(src, dst)
    H_inv = np.linalg.inv(H)
    errors = []
    for (u, v), (x, y) in zip(src, dst):
        mapped = apply_homography(H_inv, x, y)
        if mapped is None:
            errors.append(1e3)
            continue
        errors.append(math.hypot(mapped[0] - u, mapped[1] - v))
    rms = math.sqrt(sum(e * e for e in errors) / max(len(errors), 1))
    area = abs(_quad_area(src))
    angles = _interior_angles(src)
    cond = float(np.linalg.cond(H))
    return HomographyFit(
        matrix=H,
        inverse=H_inv,
        reprojection_rms_px=rms,
        area_px=area,
        min_angle_deg=min(angles) if angles else 0.0,
        condition=cond,
    )


def apply_homography(H: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    vec = H @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(vec[2]) < HOMOGRAPHY_EPS:
        return None
    return float(vec[0] / vec[2]), float(vec[1] / vec[2])


def jacobian_world_wrt_pixel(H: np.ndarray, u: float, v: float, eps: float = 0.25) -> np.ndarray:
    """2×2 Jacobian of plane (x, y) with respect to pixel (u, v)."""
    c = apply_homography(H, u, v)
    if c is None:
        return np.zeros((2, 2), dtype=np.float64)
    du = apply_homography(H, u + eps, v)
    dv = apply_homography(H, u, v + eps)
    if du is None or dv is None:
        return np.zeros((2, 2), dtype=np.float64)
    return np.array(
        [
            [(du[0] - c[0]) / eps, (dv[0] - c[0]) / eps],
            [(du[1] - c[1]) / eps, (dv[1] - c[1]) / eps],
        ],
        dtype=np.float64,
    )


def world_sigma(
    H: np.ndarray,
    u: float,
    v: float,
    *,
    pixel_sigma: float = DEFAULT_PIXEL_SIGMA,
    control_rms_px: float = 0.0,
) -> tuple[float, float]:
    sigma = math.hypot(pixel_sigma, control_rms_px)
    jac = jacobian_world_wrt_pixel(H, u, v)
    cov = (sigma * sigma) * (jac @ jac.T)
    sx = math.sqrt(max(float(cov[0, 0]), 0.0))
    sy = math.sqrt(max(float(cov[1, 1]), 0.0))
    return sx, sy


def point_in_quad(x: float, y: float, corners: list[Point2D]) -> bool:
    pts = [(p.x, p.y) for p in corners]
    if len(pts) < 4:
        return False
    signs: list[float] = []
    for i in range(4):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % 4]
        cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
        signs.append(cross)
    pos = sum(1 for s in signs if s > 1e-9)
    neg = sum(1 for s in signs if s < -1e-9)
    return pos == 0 or neg == 0


def quad_quality(corners: list[Point2D], *, image_area: float = 0.0) -> tuple[bool, str, dict[str, float]]:
    pts = np.array([[p.x, p.y] for p in corners], dtype=np.float64)
    metrics = {
        "area_px": abs(_quad_area(pts)),
        "min_angle_deg": min(_interior_angles(pts) or [0.0]),
        "coverage": 0.0,
    }
    if not _convex(pts):
        return False, "四个角点不能构成凸四边形", metrics
    if metrics["area_px"] < MIN_QUAD_AREA_PX:
        return False, "标定平面在画面中太小", metrics
    if metrics["min_angle_deg"] < MIN_INTERIOR_ANGLE_DEG:
        return False, "角点过于接近共线，请拉开四边形", metrics
    if image_area > 1.0:
        metrics["coverage"] = metrics["area_px"] / image_area
        if metrics["coverage"] < MIN_COVERAGE:
            return False, "标定平面覆盖画面过小", metrics
    return True, "", metrics


def undistort_point(
    u: float,
    v: float,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float = 0.0,
    k2: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
    k3: float = 0.0,
) -> tuple[float, float]:
    if abs(k1) + abs(k2) + abs(p1) + abs(p2) + abs(k3) < 1e-18:
        return u, v
    x = (u - cx) / fx
    y = (v - cy) / fy
    x0, y0 = x, y
    for _ in range(10):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        if abs(radial) < 1e-12:
            break
        x = (x0 - 2.0 * p1 * x * y - p2 * (r2 + 2.0 * x * x)) / radial
        y = (y0 - p1 * (r2 + 2.0 * y * y) - 2.0 * p2 * x * y) / radial
    return fx * x + cx, fy * y + cy


def distort_point(
    u: float,
    v: float,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float = 0.0,
    k2: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
    k3: float = 0.0,
) -> tuple[float, float]:
    if abs(k1) + abs(k2) + abs(p1) + abs(p2) + abs(k3) < 1e-18:
        return u, v
    x = (u - cx) / fx
    y = (v - cy) / fy
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return fx * xd + cx, fy * yd + cy


def default_camera_matrix(width: float, height: float) -> np.ndarray:
    f = max(float(width), float(height), 1.0)
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def pose_from_world_to_pixel(
    H_wp: np.ndarray,
    camera_matrix: np.ndarray | None,
    *,
    width: float = 0.0,
    height: float = 0.0,
) -> PlanePose:
    assumed = camera_matrix is None
    if camera_matrix is None:
        camera_matrix = default_camera_matrix(width or 1280.0, height or 720.0)
    k_inv = np.linalg.inv(camera_matrix)
    h1 = H_wp[:, 0]
    h2 = H_wp[:, 1]
    h3 = H_wp[:, 2]
    lam = 1.0 / max(np.linalg.norm(k_inv @ h1), 1e-12)
    r1 = lam * (k_inv @ h1)
    r2 = lam * (k_inv @ h2)
    r3 = np.cross(r1, r2)
    t = lam * (k_inv @ h3)
    rotation = np.column_stack([r1, r2, r3])
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1.0
        t = -t
    cam_z = float((rotation @ np.array([0.0, 0.0, 0.0]) + t)[2])
    if cam_z < 0:
        rotation[:, :2] *= -1.0
        t = -t
    return PlanePose(
        rotation=rotation,
        translation=t,
        camera_matrix=camera_matrix,
        assumed_intrinsics=assumed,
    )


def expected_camera_depth(
    pose: PlanePose,
    x_m: float,
    y_m: float,
) -> float | None:
    point = pose.rotation @ np.array([x_m, y_m, 0.0], dtype=np.float64) + pose.translation
    z = float(point[2])
    if z <= 1e-6:
        return None
    return z


def ray_plane_intersect(
    u: float,
    v: float,
    pose: PlanePose,
) -> tuple[float, float, float] | None:
    """Intersect the camera ray of pixel (u, v) with Z_world = 0."""
    k_inv = np.linalg.inv(pose.camera_matrix)
    direction = k_inv @ np.array([u, v, 1.0], dtype=np.float64)
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    # Camera origin in world: X = R^T (P_cam - t); origin is P_cam = 0
    r_t = pose.rotation.T
    origin_world = -r_t @ pose.translation
    dir_world = r_t @ direction
    if abs(dir_world[2]) < 1e-12:
        return None
    s = -origin_world[2] / dir_world[2]
    if s <= 0:
        return None
    hit = origin_world + s * dir_world
    return float(hit[0]), float(hit[1]), 0.0


def project_world_point(
    x_m: float,
    y_m: float,
    z_m: float,
    pose: PlanePose,
) -> tuple[float, float] | None:
    cam = pose.rotation @ np.array([x_m, y_m, z_m], dtype=np.float64) + pose.translation
    if cam[2] <= 1e-9:
        return None
    uvw = pose.camera_matrix @ cam
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])


def _dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    t_src, src_n = _normalise(src)
    t_dst, dst_n = _normalise(dst)
    rows: list[list[float]] = []
    for (u, v), (x, y) in zip(src_n, dst_n):
        rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, x * u, x * v, x])
        rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y])
    matrix = np.array(rows, dtype=np.float64)
    _, _, vh = np.linalg.svd(matrix)
    h = vh[-1].reshape(3, 3)
    H = np.linalg.inv(t_dst) @ h @ t_src
    if abs(H[2, 2]) > HOMOGRAPHY_EPS:
        H = H / H[2, 2]
    return H


def _normalise(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = pts.mean(axis=0)
    dist = np.sqrt(((pts - mean) ** 2).sum(axis=1)).mean()
    scale = math.sqrt(2.0) / max(float(dist), 1e-12)
    transform = np.array(
        [[scale, 0.0, -scale * mean[0]], [0.0, scale, -scale * mean[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homogen = transform @ np.hstack([pts, ones]).T
    return transform, homogen[:2].T


def _quad_area(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(
        x[0] * y[1]
        + x[1] * y[2]
        + x[2] * y[3]
        + x[3] * y[0]
        - (y[0] * x[1] + y[1] * x[2] + y[2] * x[3] + y[3] * x[0])
    )


def _interior_angles(pts: np.ndarray) -> list[float]:
    angles: list[float] = []
    for i in range(4):
        prev = pts[(i - 1) % 4] - pts[i]
        nxt = pts[(i + 1) % 4] - pts[i]
        n0 = np.linalg.norm(prev)
        n1 = np.linalg.norm(nxt)
        if n0 < 1e-9 or n1 < 1e-9:
            angles.append(0.0)
            continue
        cos = float(np.clip(np.dot(prev, nxt) / (n0 * n1), -1.0, 1.0))
        angles.append(math.degrees(math.acos(cos)))
    return angles


def _convex(pts: np.ndarray) -> bool:
    signs: list[int] = []
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        c = pts[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) < 1e-9:
            return False
        signs.append(1 if cross > 0 else -1)
    return len(set(signs)) == 1
