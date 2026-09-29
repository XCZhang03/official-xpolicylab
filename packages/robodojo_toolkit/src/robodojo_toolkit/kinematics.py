"""Robot-only URDF forward kinematics and the bounded local DLS step behind step_eef.

Pure NumPy/SciPy; no simulator or scene access. This module is the single source for
the harness MCP tools and for bundles running under the official policy interface.
"""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

DATA = Path(__file__).resolve().parent / "data"
ARMS = ("left", "right")


def validate_eef_target(target, *, require_gripper=False):
    """Validate a public pose before any simulator RPC or action."""
    required = {"position", "quaternion_wxyz"}
    if require_gripper:
        required.add("gripper_closed")
    if not isinstance(target, dict) or not required <= set(target):
        raise ValueError("Target is missing required pose/gripper fields")
    if set(target) - (required | {"gripper_opening"}):
        raise ValueError("Unknown target fields")
    position = np.asarray(target["position"], dtype=float)
    quaternion = np.asarray(target["quaternion_wxyz"], dtype=float)
    if (
        position.shape != (3,) or quaternion.shape != (4,)
        or not np.isfinite(np.r_[position, quaternion]).all()
        or abs(np.linalg.norm(quaternion) - 1) > 1e-4
    ):
        raise ValueError("Finite pose and unit wxyz required")
    if require_gripper and type(target["gripper_closed"]) is not bool:
        raise ValueError("Explicit boolean gripper required")
    if "gripper_opening" in target:
        opening = target["gripper_opening"]
        if (
            isinstance(opening, (bool, str)) or not np.isscalar(opening)
            or not np.isfinite(opening) or not 0 <= opening <= 1
        ):
            raise ValueError("gripper_opening must be in [0, 1]")
    return target


def transform(position, quaternion_wxyz):
    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = position
    return result


def pose(matrix):
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return {
        "position": matrix[:3, 3].tolist(),
        "quaternion_wxyz": quaternion[[3, 0, 1, 2]].tolist(),
    }


class ArmFK:
    def __init__(self, urdf, joint_names, base="base_link", tip="link6"):
        self.names = list(joint_names)
        edges = {
            joint.find("child").get("link"): joint
            for joint in ET.parse(urdf).getroot().findall("joint")
        }
        self.chain = []
        while tip != base:
            joint = edges[tip]
            origin = joint.find("origin")
            xyz = (
                np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                if origin is not None
                else np.zeros(3)
            )
            rpy = (
                np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                if origin is not None
                else np.zeros(3)
            )
            origin_transform = np.eye(4)
            origin_transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            origin_transform[:3, 3] = xyz
            axis_element = joint.find("axis")
            axis = (
                np.fromstring(axis_element.get("xyz"), sep=" ")
                if axis_element is not None
                else np.array([1.0, 0.0, 0.0])
            )
            self.chain.insert(
                0, (joint.get("name"), joint.get("type"), origin_transform, axis)
            )
            tip = joint.find("parent").get("link")
        moving = [name for name, kind, _, _ in self.chain if kind != "fixed"]
        if set(moving) != set(self.names):
            raise ValueError(f"URDF chain/arm joint mismatch: {moving}, {self.names}")

    def matrix(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self.names),) or not np.isfinite(q).all():
            raise ValueError("Invalid arm joint vector")
        values = dict(zip(self.names, q, strict=True))
        result = np.eye(4)
        for name, kind, origin, axis in self.chain:
            motion = np.eye(4)
            if kind in ("revolute", "continuous"):
                motion[:3, :3] = Rotation.from_rotvec(axis * values[name]).as_matrix()
            elif kind != "fixed":
                raise ValueError("Only fixed/revolute arm chains are supported")
            result = result @ origin @ motion
        return result

    def bounded_target(self, q, limits, root, target):
        q = np.asarray(q, dtype=float)
        now = root @ self.matrix(q)
        goal = transform(target["position"], target["quaternion_wxyz"])
        translation = goal[:3, 3] - now[:3, 3]
        rotation = Rotation.from_matrix(goal[:3, :3] @ now[:3, :3].T).as_rotvec()

        def bounded(value, cap):
            return value * min(1.0, cap / max(np.linalg.norm(value), 1e-12))

        error = np.r_[bounded(translation, 0.02), bounded(rotation, 0.1)]
        jacobian = np.zeros((6, len(q)))
        for index in range(len(q)):
            shifted = q.copy()
            shifted[index] += 1e-5
            following = root @ self.matrix(shifted)
            jacobian[:3, index] = (following[:3, 3] - now[:3, 3]) / 1e-5
            jacobian[3:, index] = (
                Rotation.from_matrix(following[:3, :3] @ now[:3, :3].T).as_rotvec()
                / 1e-5
            )
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 0.05**2 * np.eye(6), error
        )
        limits = np.asarray(limits, dtype=float)
        low, high = (
            np.maximum(limits[:, 0], q - 0.05),
            np.minimum(limits[:, 1], q + 0.05),
        )
        if np.any(low > high):
            raise ValueError("Joint outside bounded valid interval")
        result = np.clip(q + delta, low, high).astype(np.float32)
        return result, {
            "target_position_error_m": float(np.linalg.norm(translation)),
            "target_rotation_error_rad": float(np.linalg.norm(rotation)),
            "proposed_joint_delta": (result - q).tolist(),
            "bounded_task_delta": error.tolist(),
            "method": "robot_only_numerical_jacobian_dls",
            "physical_tracking_verified": False,
        }


def robot_spec(name="x5"):
    """Packaged robot facts (URDF, joint names, links, limits, dual-arm roots)."""
    spec = json.loads((DATA / name / "robot.json").read_text())
    spec["urdf"] = str(DATA / name / spec["urdf"])
    return spec


class DualArm:
    """Dual X5 kinematics in the environment frame used by observations and targets."""

    def __init__(self, name="x5"):
        self.spec = robot_spec(name)
        self.fk = {arm: ArmFK(self.spec["urdf"], self.spec["arm_joints"], self.spec["base_link"],
                              self.spec["ee_link"]) for arm in ARMS}
        self.limits = np.asarray(self.spec["joint_limits_rad"], dtype=float)
        self.roots = {arm: transform(r["position"], r["quaternion_wxyz"])
                      for arm, r in self.spec["roots"].items()}

    @staticmethod
    def _offset(arm):
        if arm not in ARMS:
            raise ValueError("arm must be left or right")
        return 0 if arm == "left" else 7

    def eef(self, arm, joints):
        """Environment-frame link6 pose {position, quaternion_wxyz} for 6 arm joints."""
        return pose(self.roots[arm] @ self.fk[arm].matrix(joints))

    def check(self, states, eef_positions, eef_quaternions_wxyz):
        """Compare packaged roots + FK against one observation's measured link6 poses."""
        states = np.asarray(states, dtype=float)
        result = {}
        for index, arm in enumerate(ARMS):
            offset = self._offset(arm)
            predicted = self.roots[arm] @ self.fk[arm].matrix(states[offset:offset + 6])
            measured = transform(eef_positions[index], eef_quaternions_wxyz[index])
            result[arm] = {
                "position_error_m": float(np.linalg.norm(predicted[:3, 3] - measured[:3, 3])),
                "rotation_error_rad": float(Rotation.from_matrix(
                    predicted[:3, :3] @ measured[:3, :3].T).magnitude()),
            }
        return result

    def step_toward(self, arm, joints, target):
        """One bounded DLS update toward a pose (the harness step_eef semantics)."""
        return self.fk[arm].bounded_target(joints, self.limits, self.roots[arm], target)

    def preview(self, actions):
        """Predicted link6 poses for 14-D joint rows (the robodojo_fk_preview result).

        Kinematic targets only: no physics, contact or task objects. Rows of any length
        (the harness tool takes exactly 50); gripper_closed means opening < 0.5.
        """
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all():
            raise ValueError("Expected N x 14 finite joint rows")
        return {
            "trajectory": [
                dict(index=index, **{
                    arm: dict(**self.eef(arm, row[offset:offset + 6]),
                              gripper_closed=bool(row[offset + 6] < 0.5),
                              gripper_opening=float(row[offset + 6]))
                    for arm, offset in (("left", 0), ("right", 7))
                })
                for index, row in enumerate(actions)
            ],
            "physical_steps": 0,
            "frame": "environment_origin",
            "interpretation": "kinematic_targets_not_object_future",
        }
