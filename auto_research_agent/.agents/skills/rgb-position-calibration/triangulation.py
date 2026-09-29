"""Self-contained RGB camera models, pixel rays, projection and multi-view triangulation.

NumPy only. Import it from this skill folder, or copy the file into a bundle:

    sys.path.insert(0, "/workspace/.agents/skills/rgb-position-calibration")
    from triangulation import triangulate, project, pixel_ray, ray_plane

or run ``python triangulation.py views.json`` (a JSON list of views, see triangulate).

The constants are the simulator-verified pinhole models in CALIBRATION.md: ideal
640 x 480 pinholes, square pixels, OpenCV camera axes (+x right, +y down, +z forward).
``cam_high`` is fixed to the world; each wrist camera is the measured link6 pose of its
arm composed with a fixed mount. Pass the ``meta`` of the reply the image came with
(``eef_positions`` and ``eef_quaternions_wxyz``, ordered [left, right]), so a wrist
camera's pose matches its image.
"""
from __future__ import annotations

from itertools import combinations
import json
import sys

import numpy as np

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
WIDTH, HEIGHT = 640, 480
# fx = focal_length * width / horizontal_aperture; fy = fx (Isaac renders square pixels).
K_HIGH = np.array([[288.1325, 0.0, 320.0], [0.0, 288.1325, 240.0], [0.0, 0.0, 1.0]])
K_WRIST = np.array([[397.0413, 0.0, 320.0], [0.0, 397.0413, 240.0], [0.0, 0.0, 1.0]])
T_ENV_FROM_HIGH = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, -0.866025, 0.5, -0.41],
                            [0.0, -0.5, -0.866025, 1.308],
                            [0.0, 0.0, 0.0, 1.0]])
T_LINK6_FROM_WRIST = np.array([[0.0, -0.50003, 0.866008, 0.084842],
                               [-1.0, 0.0, 0.0, 0.0],
                               [0.0, -0.866008, -0.50003, 0.05094],
                               [0.0, 0.0, 0.0, 1.0]])

# Advisory limits for triangulate() warnings; tune from your own validation.
MIN_RAY_ANGLE_DEG = 10.0
MAX_RAY_DISTANCE_M = 0.005
MAX_REPROJECTION_PX = 3.0


def transform(position, quaternion_wxyz):
    """4x4 pose matrix from a position and a wxyz quaternion."""
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=float) / np.linalg.norm(quaternion_wxyz)
    matrix = np.eye(4)
    matrix[:3, :3] = [[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                      [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                      [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]]
    matrix[:3, 3] = position
    return matrix


def intrinsics(camera):
    """3x3 K of one camera."""
    if camera not in CAMERAS:
        raise ValueError(f"camera must be one of {CAMERAS}")
    return (K_HIGH if camera == "cam_high" else K_WRIST).copy()


def camera_pose(camera, meta=None):
    """4x4 T_env_from_cam. Wrist cameras need the observation ``meta`` of their image."""
    if camera == "cam_high":
        return T_ENV_FROM_HIGH.copy()
    if camera not in CAMERAS:
        raise ValueError(f"camera must be one of {CAMERAS}")
    if meta is None:
        raise ValueError("Wrist cameras need the observation meta (measured link6 pose)")
    arm = 0 if camera == "cam_left_wrist" else 1
    link6 = transform(np.asarray(meta["eef_positions"][arm], dtype=float),
                      np.asarray(meta["eef_quaternions_wxyz"][arm], dtype=float))
    return link6 @ T_LINK6_FROM_WRIST


def pixel_ray(camera, pixel, meta=None):
    """(origin, unit direction) in the environment frame for pixel (u, v)."""
    pose = camera_pose(camera, meta)
    u, v = np.asarray(pixel, dtype=float).reshape(2)
    direction = pose[:3, :3] @ np.linalg.solve(intrinsics(camera), [u, v, 1.0])
    return pose[:3, 3].copy(), direction / np.linalg.norm(direction)


def ray_plane(camera, pixel, height, meta=None):
    """Point where a pixel ray meets the horizontal plane z = height (single-view estimate)."""
    origin, direction = pixel_ray(camera, pixel, meta)
    if abs(direction[2]) < 1e-9 or (height - origin[2]) / direction[2] <= 0:
        raise ValueError("Ray does not reach that plane in front of the camera")
    return origin + direction * (height - origin[2]) / direction[2]


def project(camera, point, meta=None):
    """((u, v), depth_m) of an environment point; depth <= 0 means behind the camera."""
    pose = camera_pose(camera, meta)
    local = pose[:3, :3].T @ (np.asarray(point, dtype=float).reshape(3) - pose[:3, 3])
    uvw = intrinsics(camera) @ local
    return (uvw[:2] / uvw[2]).tolist() if abs(uvw[2]) > 1e-12 else [float("nan")] * 2, float(local[2])


def triangulate(views):
    """Least-squares 3-D point from two or more views of the same physical feature.

    ``views``: [{"camera": name, "pixel": [u, v], "meta": meta}, ...]; ``meta`` is the
    observation the image came with (required for wrist cameras). Views taken at
    different times are valid only if the feature did not move in between.

    Returns the point with its evidence: per-view ray distance (m) and reprojection
    error (px), depth, the largest pairwise ray angle and advisory ``warnings``. An
    empty warning list is necessary, not sufficient: a wrong pixel correspondence can
    still agree by chance, so confirm with a third view or a measured-motion check.
    """
    if len(views) < 2:
        raise ValueError("Triangulation needs at least two views")
    rays = [pixel_ray(view["camera"], view["pixel"], view.get("meta")) for view in views]
    system, target = np.zeros((3, 3)), np.zeros(3)
    for origin, direction in rays:
        orthogonal = np.eye(3) - np.outer(direction, direction)
        system += orthogonal
        target += orthogonal @ origin
    angles = [float(np.degrees(np.arccos(np.clip(abs(a[1] @ b[1]), 0.0, 1.0))))
              for a, b in combinations(rays, 2)]
    if max(angles) < 0.5:
        raise ValueError("Rays are nearly parallel; take a view from a different direction")
    point = np.linalg.solve(system, target)
    distances = [float(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (point - o))) for o, d in rays]
    reprojection, depths = [], []
    for view in views:
        pixel, depth = project(view["camera"], point, view.get("meta"))
        reprojection.append(float(np.linalg.norm(np.subtract(pixel, view["pixel"]))))
        depths.append(depth)
    warnings = []
    if max(angles) < MIN_RAY_ANGLE_DEG:
        warnings.append(f"max ray angle {max(angles):.1f} deg < {MIN_RAY_ANGLE_DEG:g}: depth is poorly constrained")
    if min(depths) <= 0:
        warnings.append("point is behind a camera: wrong correspondence or stale pose")
    if max(distances) > MAX_RAY_DISTANCE_M:
        warnings.append(f"rays miss by {1000*max(distances):.1f} mm: check correspondence and image/pose pairing")
    if max(reprojection) > MAX_REPROJECTION_PX:
        warnings.append(f"reprojection error {max(reprojection):.1f} px > {MAX_REPROJECTION_PX:g}")
    return {"point": point.tolist(), "ray_distances_m": distances, "reprojection_px": reprojection,
            "depths_m": depths, "max_ray_angle_deg": max(angles), "warnings": warnings}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python triangulation.py views.json")
    with open(sys.argv[1]) as stream:
        print(json.dumps(triangulate(json.load(stream)), indent=2))
