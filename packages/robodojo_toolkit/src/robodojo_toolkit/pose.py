"""Pose math with the same dictionary API as the harness robodojo_pose_math tool.

NumPy/SciPy only, so it runs on any policy server. Conventions match the harness
(Isaac Lab isaaclab.utils.math): wxyz quaternions sign-standardized to w >= 0,
fixed-axis/extrinsic XYZ Euler angles in (-pi, pi], rotation vectors in radians,
row-major rotation matrices. ``compose_pose`` applies T_base * T_delta for a local
delta; an environment delta adds translation in environment axes and applies
q_delta * q_base. ``relative_pose`` is the exact inverse of ``compose_pose``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ROTATION_REPRESENTATIONS = (
    "quaternion_wxyz",
    "quaternion_xyzw",
    "euler_xyz_extrinsic_rad",
    "axis_angle_vector_rad",
    "rotation_matrix_row_major",
)


def _vector(value: Any, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite {size}-vector")
    return array


def canonical(quaternion_wxyz) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Quaternion norm must be non-zero")
    q = q / norm
    return -q if q[0] < 0 else q


def _rotation(q_wxyz) -> Rotation:
    q = canonical(q_wxyz)
    return Rotation.from_quat(q[[1, 2, 3, 0]])


def _quaternion(rotation: Rotation) -> np.ndarray:
    return canonical(rotation.as_quat()[[3, 0, 1, 2]])


def to_quaternion(rotation: dict[str, Any], label: str = "rotation") -> np.ndarray:
    """Decode one public rotation value to canonical unit wxyz."""
    if not isinstance(rotation, dict):
        raise TypeError(f"{label} must contain representation and value")
    representation = str(rotation.get("representation", ""))
    value = rotation.get("value")
    if representation not in ROTATION_REPRESENTATIONS:
        raise ValueError(f"{label}.representation must be one of {ROTATION_REPRESENTATIONS}")
    if representation == "quaternion_wxyz":
        return canonical(_vector(value, 4, f"{label}.value"))
    if representation == "quaternion_xyzw":
        return canonical(_vector(value, 4, f"{label}.value")[[3, 0, 1, 2]])
    if representation == "euler_xyz_extrinsic_rad":
        return _quaternion(Rotation.from_euler("xyz", _vector(value, 3, f"{label}.value")))
    if representation == "axis_angle_vector_rad":
        vector = _vector(value, 3, f"{label}.value")
        if np.linalg.norm(vector) < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return _quaternion(Rotation.from_rotvec(vector))
    matrix = _vector(value, 9, f"{label}.value").reshape(3, 3)
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5, rtol=0.0):
        raise ValueError(f"{label}.value must be an orthonormal rotation matrix")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError(f"{label}.value rotation-matrix determinant must be +1")
    return _quaternion(Rotation.from_matrix(matrix))


def from_quaternion(quaternion_wxyz, representation: str) -> dict[str, Any]:
    """Encode canonical wxyz in one public representation."""
    if representation not in ROTATION_REPRESENTATIONS:
        raise ValueError(f"output_representation must be one of {ROTATION_REPRESENTATIONS}")
    q = canonical(quaternion_wxyz)
    if representation == "quaternion_wxyz":
        value = q.tolist()
    elif representation == "quaternion_xyzw":
        value = q[[1, 2, 3, 0]].tolist()
    elif representation == "euler_xyz_extrinsic_rad":
        value = _rotation(q).as_euler("xyz").tolist()
    elif representation == "axis_angle_vector_rad":
        value = _rotation(q).as_rotvec().tolist()
    else:
        value = _rotation(q).as_matrix().reshape(-1).tolist()
    return {"representation": representation, "value": value}


def _full_pose(value: dict[str, Any], label: str):
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return (_vector(value.get("position_m"), 3, f"{label}.position_m"),
            to_quaternion(value.get("rotation"), f"{label}.rotation"))


def _delta_pose(value: dict[str, Any]):
    if not isinstance(value, dict):
        raise ValueError("delta_pose must be an object")
    position = _vector(value["position_m"], 3, "delta_pose.position_m") if "position_m" in value else np.zeros(3)
    quaternion = (to_quaternion(value["rotation"], "delta_pose.rotation")
                  if "rotation" in value else np.array([1.0, 0.0, 0.0, 0.0]))
    return position, quaternion


def _result(position, quaternion, representation):
    q = canonical(quaternion)
    return {"position_m": np.asarray(position, dtype=float).reshape(3).tolist(),
            "rotation": from_quaternion(q, representation), "quaternion_wxyz": q.tolist()}


def compose(base_position, base_quaternion, delta_position, delta_quaternion, frame="local"):
    """Apply a delta to a base pose; returns (position, wxyz)."""
    base_r, delta_r = _rotation(base_quaternion), _rotation(delta_quaternion)
    if frame == "local":
        return (np.asarray(base_position) + base_r.apply(delta_position),
                _quaternion(base_r * delta_r))
    if frame == "environment":
        return np.asarray(base_position) + np.asarray(delta_position), _quaternion(delta_r * base_r)
    raise ValueError("delta_frame must be local or environment")


def relative(base_position, base_quaternion, target_position, target_quaternion, frame="local"):
    """Delta such that compose(base, delta, frame) == target; returns (position, wxyz)."""
    base_r, target_r = _rotation(base_quaternion), _rotation(target_quaternion)
    offset = np.asarray(target_position) - np.asarray(base_position)
    if frame == "local":
        return base_r.inv().apply(offset), _quaternion(base_r.inv() * target_r)
    if frame == "environment":
        return offset, _quaternion(target_r * base_r.inv())
    raise ValueError("delta_frame must be local or environment")


def _opening(value, label):
    opening = float(value)
    if not np.isfinite(opening) or not 0.0 <= opening <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return opening


def pose_math(arguments: dict[str, Any]) -> dict[str, Any]:
    """Drop-in for ctx.call('robodojo_pose_math', **arguments) returning the parsed result."""
    operation = str(arguments.get("operation", ""))
    output = str(arguments.get("output_representation", "quaternion_wxyz"))
    if operation == "convert_rotation":
        q = to_quaternion(arguments.get("rotation"))
        result: dict[str, Any] = {"rotation": from_quaternion(q, output), "quaternion_wxyz": q.tolist()}
    elif operation in ("compose_pose", "relative_pose"):
        frame = str(arguments.get("delta_frame", "local"))
        base = _full_pose(arguments.get("base_pose"), "base_pose")
        if operation == "compose_pose":
            position, q = compose(*base, *_delta_pose(arguments.get("delta_pose")), frame=frame)
            result = {"pose": _result(position, q, output), "delta_frame": frame}
        else:
            position, q = relative(*base, *_full_pose(arguments.get("target_pose"), "target_pose"), frame=frame)
            result = {"delta_pose": _result(position, q, output), "delta_frame": frame}
    elif operation == "format_target":
        position, q = _full_pose(arguments.get("pose"), "pose")
        kind = str(arguments.get("target_kind", "step_eef"))
        target: dict[str, Any] = {"position": position.tolist(), "quaternion_wxyz": q.tolist()}
        if kind == "step_eef":
            if type(arguments.get("gripper_closed")) is not bool:
                raise ValueError("step_eef format requires boolean gripper_closed")
            target["gripper_closed"] = arguments["gripper_closed"]
        elif kind != "free_space_move":
            raise ValueError("target_kind must be step_eef or free_space_move")
        if "gripper_opening" in arguments:
            target["gripper_opening"] = _opening(arguments["gripper_opening"], "gripper_opening")
        result = {"target_kind": kind, "target": target}
    elif operation == "extract_target":
        target = arguments.get("eef_target")
        if not isinstance(target, dict):
            raise ValueError("eef_target must be an object")
        position = _vector(target.get("position"), 3, "eef_target.position")
        q = to_quaternion({"representation": "quaternion_wxyz", "value": target.get("quaternion_wxyz")},
                          "eef_target.quaternion_wxyz")
        result = {"pose": _result(position, q, output)}
        if "gripper_closed" in target:
            if type(target["gripper_closed"]) is not bool:
                raise ValueError("eef_target.gripper_closed must be boolean")
            result["gripper_closed"] = target["gripper_closed"]
        if "gripper_opening" in target:
            result["gripper_opening"] = _opening(target["gripper_opening"], "eef_target.gripper_opening")
    else:
        raise ValueError("operation must be convert_rotation, compose_pose, relative_pose, "
                         "format_target, or extract_target")
    result.update(operation=operation, math_backend="robodojo_toolkit.pose (numpy/scipy)",
                  scene_state_accessed=False, physical_steps=0)
    return result
