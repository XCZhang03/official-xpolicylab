"""High-level RoboDojo pose operations backed by Isaac Lab math utilities."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

ROTATION_REPRESENTATIONS = (
    "quaternion_wxyz",
    "quaternion_xyzw",
    "euler_xyz_extrinsic_rad",
    "axis_angle_vector_rad",
    "rotation_matrix_row_major",
)


@lru_cache(maxsize=4)
def _load_isaac_math(project_root: str) -> ModuleType:
    """Load the vendored pure-math module without importing Isaac/Kit packages."""

    path = (
        Path(project_root)
        / "RoboDojo"
        / "third_party"
        / "IsaacLab"
        / "source"
        / "isaaclab"
        / "isaaclab"
        / "utils"
        / "math.py"
    )
    if not path.is_file():
        raise RuntimeError(f"Missing reviewed Isaac Lab math module: {path}")
    module_name = f"_robodojo_isaaclab_math_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Isaac Lab math module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite {size}-vector")
    return array


def _canonical_quaternion(math_module: ModuleType, value: torch.Tensor) -> torch.Tensor:
    value = value.reshape(1, 4).to(dtype=torch.float64, device="cpu")
    if float(torch.linalg.vector_norm(value, dim=-1)[0]) < 1e-12:
        raise ValueError("Quaternion norm must be non-zero")
    return math_module.quat_unique(math_module.normalize(value))[0]


def _rotation_to_quaternion(
    math_module: ModuleType, rotation: dict[str, Any], label: str = "rotation"
) -> torch.Tensor:
    """Decode one public rotation value to canonical unit ``wxyz``.

    Euler input is fixed-axis/extrinsic XYZ in radians. Axis-angle input is a
    rotation vector whose norm is the angle. Matrix input is flat row-major and
    must be orthonormal with determinant +1. Non-zero quaternion input is
    normalized; all output is sign-standardized to a non-negative scalar part.
    """

    if not isinstance(rotation, dict):
        raise TypeError(f"{label} must contain representation and value")
    representation = str(rotation.get("representation", ""))
    value = rotation.get("value")
    if representation not in ROTATION_REPRESENTATIONS:
        raise ValueError(
            f"{label}.representation must be one of {ROTATION_REPRESENTATIONS}"
        )

    if representation == "quaternion_wxyz":
        quaternion = torch.as_tensor(
            _finite_vector(value, 4, f"{label}.value"), dtype=torch.float64
        )
    elif representation == "quaternion_xyzw":
        xyzw = _finite_vector(value, 4, f"{label}.value")
        quaternion = torch.as_tensor(
            math_module.convert_quat(xyzw, to="wxyz"), dtype=torch.float64
        )
    elif representation == "euler_xyz_extrinsic_rad":
        euler = torch.as_tensor(
            _finite_vector(value, 3, f"{label}.value"), dtype=torch.float64
        )
        quaternion = math_module.quat_from_euler_xyz(
            euler[0:1], euler[1:2], euler[2:3]
        )[0]
    elif representation == "axis_angle_vector_rad":
        vector = torch.as_tensor(
            _finite_vector(value, 3, f"{label}.value"), dtype=torch.float64
        )
        angle = torch.linalg.vector_norm(vector).reshape(1)
        if float(angle[0]) < 1e-12:
            quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        else:
            quaternion = math_module.quat_from_angle_axis(
                angle, (vector / angle[0]).reshape(1, 3)
            )[0]
    else:
        matrix = _finite_vector(value, 9, f"{label}.value").reshape(3, 3)
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5, rtol=0.0):
            raise ValueError(f"{label}.value must be an orthonormal rotation matrix")
        if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-5, rtol=0.0):
            raise ValueError(f"{label}.value rotation-matrix determinant must be +1")
        quaternion = math_module.quat_from_matrix(
            torch.as_tensor(matrix, dtype=torch.float64)
        )
    return _canonical_quaternion(math_module, quaternion)


def _quaternion_to_rotation(
    math_module: ModuleType, quaternion: torch.Tensor, representation: str
) -> dict[str, Any]:
    """Encode canonical ``wxyz`` in one unambiguous public representation."""

    if representation not in ROTATION_REPRESENTATIONS:
        raise ValueError(
            f"output_representation must be one of {ROTATION_REPRESENTATIONS}"
        )
    quaternion = _canonical_quaternion(math_module, quaternion)
    batched = quaternion.reshape(1, 4)
    if representation == "quaternion_wxyz":
        value = quaternion.tolist()
    elif representation == "quaternion_xyzw":
        value = math_module.convert_quat(
            quaternion.detach().cpu().numpy(), to="xyzw"
        ).tolist()
    elif representation == "euler_xyz_extrinsic_rad":
        roll, pitch, yaw = math_module.euler_xyz_from_quat(batched)
        value = [float(roll[0]), float(pitch[0]), float(yaw[0])]
    elif representation == "axis_angle_vector_rad":
        value = math_module.axis_angle_from_quat(batched)[0].tolist()
    else:
        value = math_module.matrix_from_quat(batched)[0].reshape(-1).tolist()
    return {"representation": representation, "value": value}


def _full_pose(
    math_module: ModuleType, value: dict[str, Any], label: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a complete ``position_m`` plus ``rotation`` pose."""

    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    position = torch.as_tensor(
        _finite_vector(value.get("position_m"), 3, f"{label}.position_m"),
        dtype=torch.float64,
    )
    quaternion = _rotation_to_quaternion(
        math_module, value.get("rotation"), f"{label}.rotation"
    )
    return position, quaternion


def _delta_pose(
    math_module: ModuleType, value: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a partial delta, using zero translation and identity rotation defaults."""

    if not isinstance(value, dict):
        raise ValueError("delta_pose must be an object")
    position = torch.zeros(3, dtype=torch.float64)
    quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    if "position_m" in value:
        position = torch.as_tensor(
            _finite_vector(value["position_m"], 3, "delta_pose.position_m"),
            dtype=torch.float64,
        )
    if "rotation" in value:
        quaternion = _rotation_to_quaternion(
            math_module, value["rotation"], "delta_pose.rotation"
        )
    return position, quaternion


def _pose_result(
    math_module: ModuleType,
    position: torch.Tensor,
    quaternion: torch.Tensor,
    representation: str,
) -> dict[str, Any]:
    """Return a pose in the requested representation plus control-ready ``wxyz``."""

    quaternion = _canonical_quaternion(math_module, quaternion)
    return {
        "position_m": position.reshape(3).tolist(),
        "rotation": _quaternion_to_rotation(math_module, quaternion, representation),
        "quaternion_wxyz": quaternion.tolist(),
    }


def pose_math(project_root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute one validated, scene-independent high-level pose operation.

    ``compose_pose`` applies a local delta as ``T_base * T_delta``. An
    environment delta adds translation directly in environment axes and
    left-multiplies orientation as ``q_delta * q_base`` without rotating the
    base position about the environment origin. ``relative_pose`` is the exact
    inverse under the selected convention. ``format_target`` and
    ``extract_target`` translate between general poses and the named RoboDojo
    EEF/cuRobo target fields without reading or advancing simulation.
    """

    math_module = _load_isaac_math(str(Path(project_root).resolve()))
    operation = str(arguments.get("operation", ""))
    output_representation = str(
        arguments.get("output_representation", "quaternion_wxyz")
    )

    if operation == "convert_rotation":
        quaternion = _rotation_to_quaternion(math_module, arguments.get("rotation"))
        result: dict[str, Any] = {
            "rotation": _quaternion_to_rotation(
                math_module, quaternion, output_representation
            ),
            "quaternion_wxyz": quaternion.tolist(),
        }
    elif operation == "compose_pose":
        base_position, base_quaternion = _full_pose(
            math_module, arguments.get("base_pose"), "base_pose"
        )
        delta_position, delta_quaternion = _delta_pose(
            math_module, arguments.get("delta_pose")
        )
        delta_frame = str(arguments.get("delta_frame", "local"))
        if delta_frame == "local":
            position, quaternion = math_module.combine_frame_transforms(
                base_position.reshape(1, 3),
                base_quaternion.reshape(1, 4),
                delta_position.reshape(1, 3),
                delta_quaternion.reshape(1, 4),
            )
            position, quaternion = position[0], quaternion[0]
        elif delta_frame == "environment":
            position = base_position + delta_position
            quaternion = math_module.quat_mul(
                delta_quaternion.reshape(1, 4), base_quaternion.reshape(1, 4)
            )[0]
        else:
            raise ValueError("delta_frame must be local or environment")
        result = {
            "pose": _pose_result(
                math_module, position, quaternion, output_representation
            ),
            "delta_frame": delta_frame,
        }
    elif operation == "relative_pose":
        base_position, base_quaternion = _full_pose(
            math_module, arguments.get("base_pose"), "base_pose"
        )
        target_position, target_quaternion = _full_pose(
            math_module, arguments.get("target_pose"), "target_pose"
        )
        delta_frame = str(arguments.get("delta_frame", "local"))
        if delta_frame == "local":
            position, quaternion = math_module.subtract_frame_transforms(
                base_position.reshape(1, 3),
                base_quaternion.reshape(1, 4),
                target_position.reshape(1, 3),
                target_quaternion.reshape(1, 4),
            )
            position, quaternion = position[0], quaternion[0]
        elif delta_frame == "environment":
            position = target_position - base_position
            quaternion = math_module.quat_mul(
                target_quaternion.reshape(1, 4),
                math_module.quat_inv(base_quaternion.reshape(1, 4)),
            )[0]
        else:
            raise ValueError("delta_frame must be local or environment")
        result = {
            "delta_pose": _pose_result(
                math_module, position, quaternion, output_representation
            ),
            "delta_frame": delta_frame,
        }
    elif operation == "format_target":
        position, quaternion = _full_pose(math_module, arguments.get("pose"), "pose")
        target_kind = str(arguments.get("target_kind", "step_eef"))
        target: dict[str, Any] = {
            "position": position.tolist(),
            "quaternion_wxyz": quaternion.tolist(),
        }
        if target_kind == "step_eef":
            if type(arguments.get("gripper_closed")) is not bool:
                raise ValueError("step_eef format requires boolean gripper_closed")
            target["gripper_closed"] = arguments["gripper_closed"]
        elif target_kind != "free_space_move":
            raise ValueError("target_kind must be step_eef or free_space_move")
        if "gripper_opening" in arguments:
            opening = float(arguments["gripper_opening"])
            if not np.isfinite(opening) or not 0.0 <= opening <= 1.0:
                raise ValueError("gripper_opening must be in [0, 1]")
            target["gripper_opening"] = opening
        result = {"target_kind": target_kind, "target": target}
    elif operation == "extract_target":
        target = arguments.get("eef_target")
        if not isinstance(target, dict):
            raise ValueError("eef_target must be an object")
        position = torch.as_tensor(
            _finite_vector(target.get("position"), 3, "eef_target.position"),
            dtype=torch.float64,
        )
        quaternion = _rotation_to_quaternion(
            math_module,
            {
                "representation": "quaternion_wxyz",
                "value": target.get("quaternion_wxyz"),
            },
            "eef_target.quaternion_wxyz",
        )
        result = {
            "pose": _pose_result(
                math_module, position, quaternion, output_representation
            )
        }
        if "gripper_closed" in target:
            if type(target["gripper_closed"]) is not bool:
                raise ValueError("eef_target.gripper_closed must be boolean")
            result["gripper_closed"] = target["gripper_closed"]
        if "gripper_opening" in target:
            opening = float(target["gripper_opening"])
            if not np.isfinite(opening) or not 0.0 <= opening <= 1.0:
                raise ValueError("eef_target.gripper_opening must be in [0, 1]")
            result["gripper_opening"] = opening
    else:
        raise ValueError(
            "operation must be convert_rotation, compose_pose, relative_pose, "
            "format_target, or extract_target"
        )

    result.update(
        operation=operation,
        math_backend="Isaac Lab isaaclab.utils.math",
        scene_state_accessed=False,
        physical_steps=0,
    )
    return result
