"""Visible-pixel geometry using only camera observations and calibration."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _calibration(observation: dict[str, Any], camera: str) -> dict[str, Any]:
    if camera not in CAMERAS:
        raise ValueError(f"Unsupported camera: {camera}")
    parameters = observation.get("camera_parameters", {}).get(camera)
    if parameters is None:
        raise ValueError(
            f"Camera calibration is unavailable for {camera}; initialize the "
            "episode with include_camera_parameters=true"
        )
    if not isinstance(parameters, dict):
        raise TypeError(f"Camera calibration for {camera} must be an object")
    intrinsic = _finite_array(
        parameters.get("intrinsic_matrix"), (3, 3), f"{camera} intrinsic matrix"
    )
    transform = _finite_array(
        parameters.get("camera_to_world_usd"),
        (4, 4),
        f"{camera} camera-to-world transform",
    )
    resolution = parameters.get("resolution")
    if (
        not isinstance(resolution, (list, tuple))
        or len(resolution) != 2
        or int(resolution[0]) <= 0
        or int(resolution[1]) <= 0
    ):
        raise ValueError(f"Invalid resolution for {camera}: {resolution}")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"Invalid homogeneous transform for {camera}")
    if abs(np.linalg.det(transform[:3, :3])) < 1e-8:
        raise ValueError(f"Singular camera rotation for {camera}")
    return {
        "intrinsic": intrinsic,
        "camera_to_world_usd": transform,
        "resolution": (int(resolution[0]), int(resolution[1])),
    }


def _pixel_uv(
    selection: dict[str, Any],
    resolution: tuple[int, int],
    coordinate_space: str,
) -> np.ndarray:
    pixel = _finite_array(selection.get("pixel"), (2,), "selection pixel")
    width, height = resolution
    if coordinate_space == "normalized_0_1":
        if np.any(pixel < 0) or np.any(pixel > 1):
            raise ValueError("Normalized pixel coordinates must be in [0, 1]")
        pixel = pixel * np.array([width - 1, height - 1], dtype=np.float64)
    elif coordinate_space != "image_pixels":
        raise ValueError("coordinate_space must be image_pixels or normalized_0_1")
    if not 0 <= pixel[0] <= width - 1 or not 0 <= pixel[1] <= height - 1:
        raise ValueError(
            f"Pixel {pixel.tolist()} is outside image resolution {[width, height]}"
        )
    return pixel


def pixel_ray_world(
    observation: dict[str, Any],
    selection: dict[str, Any],
    *,
    coordinate_space: str,
) -> dict[str, Any]:
    camera = str(selection.get("camera", ""))
    calibration = _calibration(observation, camera)
    pixel = _pixel_uv(selection, calibration["resolution"], coordinate_space)
    intrinsic = calibration["intrinsic"]
    transform = calibration["camera_to_world_usd"]
    ray_cv = np.linalg.solve(intrinsic, np.array([pixel[0], pixel[1], 1.0]))
    if not np.isfinite(ray_cv).all() or ray_cv[2] <= 0:
        raise ValueError(f"Invalid pinhole ray for {camera}")
    # RGB/depth intrinsics use CV optical axes (+X right, +Y down, +Z
    # forward); the live camera transform uses USD axes (+X right, +Y up,
    # -Z forward).
    ray_usd = np.array([ray_cv[0], -ray_cv[1], -ray_cv[2]])
    direction = transform[:3, :3] @ ray_usd
    direction /= np.linalg.norm(direction)
    return {
        "camera": camera,
        "pixel_uv": pixel,
        "rounded_pixel_uv": np.rint(pixel).astype(int),
        "resolution": calibration["resolution"],
        "origin_world_m": transform[:3, 3].copy(),
        "direction_world": direction,
        "ray_cv": ray_cv,
        "camera_to_world_usd": transform,
    }


def _depth_sample(
    observation: dict[str, Any], ray: dict[str, Any], radius: int
) -> dict[str, Any] | None:
    key = f"{ray['camera']}_depth_m"
    if key not in observation:
        return None
    depth = np.asarray(observation[key], dtype=np.float64)
    width, height = ray["resolution"]
    if depth.shape != (height, width):
        raise ValueError(
            f"Depth shape for {ray['camera']} is {depth.shape}, expected {(height, width)}"
        )
    u, v = (int(value) for value in ray["rounded_pixel_uv"])
    u0, u1 = max(0, u - radius), min(width, u + radius + 1)
    v0, v1 = max(0, v - radius), min(height, v + radius + 1)
    window = depth[v0:v1, u0:u1]
    valid = window[np.isfinite(window) & (window > 0)]
    if not valid.size:
        return {
            "available": True,
            "valid": False,
            "window_uv_bounds": [u0, v0, u1 - 1, v1 - 1],
            "valid_sample_count": 0,
        }
    value = float(np.median(valid))
    median_absolute_deviation = float(np.median(np.abs(valid - value)))
    ray_cv = ray["ray_cv"] / ray["ray_cv"][2]
    point_cv = ray_cv * value
    point_usd = np.array([point_cv[0], -point_cv[1], -point_cv[2]])
    transform = ray["camera_to_world_usd"]
    point_world = transform[:3, :3] @ point_usd + transform[:3, 3]
    return {
        "available": True,
        "valid": True,
        "depth_m": value,
        "depth_median_absolute_deviation_m": median_absolute_deviation,
        "window_uv_bounds": [u0, v0, u1 - 1, v1 - 1],
        "valid_sample_count": int(valid.size),
        "point_world_m": point_world,
    }


def _max_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return max(
        float(np.linalg.norm(points[first] - points[second]))
        for first in range(len(points))
        for second in range(first + 1, len(points))
    )


def _triangulate(rays: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    if len(rays) < 2:
        raise ValueError(
            "Triangulation requires matching pixels for the same physical point "
            "in at least two camera views"
        )
    identity = np.eye(3, dtype=np.float64)
    matrix = np.zeros((3, 3), dtype=np.float64)
    vector = np.zeros(3, dtype=np.float64)
    for ray in rays:
        direction = ray["direction_world"]
        projector = identity - np.outer(direction, direction)
        matrix += projector
        vector += projector @ ray["origin_world_m"]
    point, _, _, _ = np.linalg.lstsq(matrix, vector, rcond=None)
    condition_number = float(np.linalg.cond(matrix))
    closest_points = []
    distances = []
    forward_distances = []
    for ray in rays:
        offset = point - ray["origin_world_m"]
        forward = float(np.dot(offset, ray["direction_world"]))
        closest = ray["origin_world_m"] + forward * ray["direction_world"]
        closest_points.append(closest)
        distances.append(float(np.linalg.norm(point - closest)))
        forward_distances.append(forward)
    angles = []
    for first in range(len(rays)):
        for second in range(first + 1, len(rays)):
            cosine = float(
                np.clip(
                    np.dot(
                        rays[first]["direction_world"],
                        rays[second]["direction_world"],
                    ),
                    -1.0,
                    1.0,
                )
            )
            angles.append(math.degrees(math.acos(cosine)))
    return point, {
        "ray_count": len(rays),
        "condition_number": condition_number,
        "rms_point_to_ray_residual_m": float(np.sqrt(np.mean(np.square(distances)))),
        "max_point_to_ray_residual_m": max(distances),
        "closest_points_max_spread_m": _max_distance(np.asarray(closest_points)),
        "minimum_ray_angle_degrees": min(angles),
        "maximum_ray_angle_degrees": max(angles),
        "forward_distances_m": forward_distances,
        "all_rays_forward": all(value > 0 for value in forward_distances),
    }


def locate_pixel_selections(
    observation: dict[str, Any],
    selections: list[dict[str, Any]],
    *,
    coordinate_space: str = "image_pixels",
    method: str = "auto",
    depth_window_radius: int = 1,
    max_position_spread_m: float = 0.05,
    max_triangulation_residual_m: float = 0.03,
) -> dict[str, Any]:
    """Estimate a visible surface point without consulting task-object state."""
    if not isinstance(selections, list) or not 1 <= len(selections) <= 3:
        raise ValueError("selections must contain one to three camera pixels")
    cameras = [str(selection.get("camera", "")) for selection in selections]
    if len(set(cameras)) != len(cameras):
        raise ValueError("Provide at most one selected pixel per camera")
    if method not in ("auto", "depth", "triangulation"):
        raise ValueError("method must be auto, depth, or triangulation")
    radius = int(depth_window_radius)
    if not 0 <= radius <= 5:
        raise ValueError("depth_window_radius must be between 0 and 5")
    if not 0 < max_position_spread_m <= 1:
        raise ValueError("max_position_spread_m must be in (0, 1]")
    if not 0 < max_triangulation_residual_m <= 1:
        raise ValueError("max_triangulation_residual_m must be in (0, 1]")

    rays = [
        pixel_ray_world(observation, selection, coordinate_space=coordinate_space)
        for selection in selections
    ]
    depth_samples = [_depth_sample(observation, ray, radius) for ray in rays]
    valid_depth = [
        (ray, sample)
        for ray, sample in zip(rays, depth_samples, strict=True)
        if sample is not None and sample.get("valid")
    ]
    environment_origin = np.asarray(
        observation.get("environment_origin_world_m", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if environment_origin.shape != (3,) or not np.isfinite(environment_origin).all():
        raise ValueError("Invalid environment origin in observation")

    diagnostics = []
    for ray, sample in zip(rays, depth_samples, strict=True):
        row = {
            "camera": ray["camera"],
            "pixel_uv": ray["pixel_uv"].tolist(),
            "rounded_pixel_uv": ray["rounded_pixel_uv"].tolist(),
            "resolution": list(ray["resolution"]),
            "ray_origin_world_m": ray["origin_world_m"].tolist(),
            "ray_direction_world": ray["direction_world"].tolist(),
            "depth": None,
        }
        if sample is not None:
            row["depth"] = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in sample.items()
                if key != "point_world_m"
            }
        diagnostics.append(row)

    use_depth = method == "depth" or (method == "auto" and valid_depth)
    if use_depth:
        if not valid_depth:
            raise ValueError(
                "No valid selected depth sample; select a valid surface pixel or "
                "provide corresponding pixels in at least two views for triangulation"
            )
        points = np.asarray(
            [sample["point_world_m"] for _, sample in valid_depth],
            dtype=np.float64,
        )
        point_world = np.median(points, axis=0)
        spread = _max_distance(points)
        quality_label = "good" if spread <= max_position_spread_m else "low"
        quality = {
            "label": quality_label,
            "valid_depth_selection_count": len(valid_depth),
            "selected_camera_count": len(rays),
            "max_depth_point_spread_m": spread,
            "max_allowed_position_spread_m": max_position_spread_m,
            "depth_window_radius_pixels": radius,
        }
        used_cameras = [ray["camera"] for ray, _ in valid_depth]
        resolved_method = "depth_unprojection"
    else:
        point_world, triangulation = _triangulate(rays)
        weak = (
            not triangulation["all_rays_forward"]
            or triangulation["minimum_ray_angle_degrees"] < 2.0
            or triangulation["condition_number"] > 1e6
            or triangulation["max_point_to_ray_residual_m"]
            > max_triangulation_residual_m
        )
        quality = {
            "label": "low" if weak else "good",
            **triangulation,
            "max_allowed_triangulation_residual_m": (max_triangulation_residual_m),
        }
        used_cameras = [ray["camera"] for ray in rays]
        resolved_method = "multi_view_triangulation"
        if method == "auto" and any(sample is not None for sample in depth_samples):
            resolved_method = "multi_view_triangulation_after_invalid_depth"

    point_environment = point_world - environment_origin
    return {
        "status": "Success" if quality["label"] == "good" else "Low_Confidence",
        "method": resolved_method,
        "position_environment_m": point_environment.tolist(),
        "position_world_m": point_world.tolist(),
        "environment_origin_world_m": environment_origin.tolist(),
        "coordinate_frame_for_robot_commands": "environment_origin",
        "position_semantics": "visible_surface_point_not_object_center",
        "used_cameras": used_cameras,
        "quality": quality,
        "selections": diagnostics,
        "privilege_boundary": {
            "uses_rgb_selected_pixels": True,
            "uses_camera_calibration": True,
            "uses_observed_depth": resolved_method == "depth_unprojection",
            "uses_task_object_pose": False,
            "uses_scene_manager": False,
            "uses_segmentation": False,
        },
    }
