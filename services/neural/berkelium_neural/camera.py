"""Canonical camera math (OpenCV axes, T_cw world->camera) and explicit convention adapters.

All conversions are explicit; nothing infers an axis convention from a file extension.
"""

from __future__ import annotations

import math

import numpy as np

from .contracts import CameraSpec

# OpenGL camera axes (+X right, +Y up, -Z forward) -> OpenCV camera axes (+X right, +Y down, +Z forward)
F_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """World->camera (OpenCV) 4x4 for a camera at ``eye`` looking at ``target`` with world +Y up."""
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    z = target - eye
    n = np.linalg.norm(z)
    if n < 1e-12:
        raise ValueError("eye and target coincide")
    z /= n                                   # camera +Z (forward)
    x = np.cross(z, up)                      # camera +X (right) = forward x up (OpenCV: y down)
    if np.linalg.norm(x) < 1e-9:
        raise ValueError("up vector parallel to view direction")
    x /= np.linalg.norm(x)
    y = np.cross(z, x)                       # camera +Y (down)
    R = np.stack([x, y, z])                  # rows = camera axes in world coords -> R_cw
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ eye
    return T


def intrinsics_from_fov(width: int, height: int, fov_y_deg: float) -> tuple[float, float, float, float]:
    fy = 0.5 * height / math.tan(math.radians(fov_y_deg) / 2)
    return fy, fy, width / 2.0, height / 2.0


def make_camera(eye, target=(0, 0, 0), width=64, height=64, fov_y_deg=40.0, near=0.01, far=100.0) -> CameraSpec:
    fx, fy, cx, cy = intrinsics_from_fov(width, height, fov_y_deg)
    T = look_at(eye, target)
    return CameraSpec(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy,
                      T_cw=tuple(tuple(float(v) for v in r) for r in T), near=near, far=far)


def orbit_cameras(n: int, radius: float, elevation_deg: float = 20.0, width=64, height=64, fov_y_deg=40.0,
                  target=(0.0, 0.0, 0.0), phase_deg: float = 0.0) -> list[CameraSpec]:
    """``n`` cameras evenly spaced in azimuth on a ring around ``target`` (world +Y up)."""
    out = []
    el = math.radians(elevation_deg)
    for i in range(n):
        az = math.radians(phase_deg + 360.0 * i / n)
        eye = (target[0] + radius * math.cos(el) * math.sin(az), target[1] + radius * math.sin(el),
               target[2] + radius * math.cos(el) * math.cos(az))
        out.append(make_camera(eye, target, width, height, fov_y_deg))
    return out


def T_cw(cam: CameraSpec) -> np.ndarray:
    return np.asarray(cam.T_cw, dtype=np.float64)


def K(cam: CameraSpec) -> np.ndarray:
    return np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1.0]])


def camera_center(cam: CameraSpec) -> np.ndarray:
    T = T_cw(cam)
    return -T[:3, :3].T @ T[:3, 3]


def from_opengl_c2w(T_wc_gl: np.ndarray, width, height, fx, fy, cx, cy, near=0.01, far=100.0) -> CameraSpec:
    """OpenGL camera-to-world (Blender/NeRF-synthetic style) -> canonical. Same world basis:
    T_wc_cv = T_wc_gl @ F ; T_cw = inverse(T_wc_cv)."""
    T_wc_cv = np.asarray(T_wc_gl, dtype=np.float64) @ F_GL_TO_CV
    T = np.linalg.inv(T_wc_cv)
    T[3] = (0, 0, 0, 1)
    return CameraSpec(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy,
                      T_cw=tuple(tuple(float(v) for v in r) for r in T), near=near, far=far)


def resized(cam: CameraSpec, width: int, height: int) -> CameraSpec:
    """Resize under the edge-origin convention: fx,cx scale by sx; fy,cy by sy."""
    sx, sy = width / cam.width, height / cam.height
    return cam.model_copy(update={"width": width, "height": height, "fx": cam.fx * sx, "fy": cam.fy * sy,
                                  "cx": cam.cx * sx, "cy": cam.cy * sy})


def project_points(cam: CameraSpec, pts_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates (u, v) and camera depth Z for world points (no culling)."""
    T = T_cw(cam)
    pc = pts_w @ T[:3, :3].T + T[:3, 3]
    z = pc[:, 2]
    uv = np.stack([cam.fx * pc[:, 0] / z + cam.cx, cam.fy * pc[:, 1] / z + cam.cy], -1)
    return uv, z
