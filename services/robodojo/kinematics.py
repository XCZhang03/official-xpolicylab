"""Robot-only URDF FK and bounded local DLS; no scene-object access."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


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


def measured_joint_state(manager):
    """Read physical arm joints and normalized grippers, never cached commands."""
    robots = {
        robot.arm_name.split("_")[0]: robot
        for robot in manager.robot_list if robot.type == "target"
    }
    values = []
    for arm in ("left", "right"):
        robot = robots[arm]
        joints = manager.get_joint(robot, env_idx_list=[0])[0]
        primary = manager.get_end_effector_real_val(robot, env_idx_list=[0])[0][0]
        low, high = robot.gripper_scale
        opening = (primary - low) / (high - low)
        if robot.gripper_move["sign"] != 1:
            opening = 1 - opening
        values.extend([*joints, float(np.clip(opening, 0, 1))])
    state = np.asarray(values, dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("Invalid measured dual-arm state")
    return state


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


class DualKinematics:
    def __init__(self, env):
        self.env, self.manager = env, env.robot_manager
        self.robots = {
            robot.arm_name.split("_")[0]: robot
            for robot in self.manager.robot_list
            if robot.type == "target"
        }
        if set(self.robots) != {"left", "right"}:
            raise ValueError("Expected left/right target X5 arms")
        self.fk = {
            arm: ArmFK(
                robot.urdf_path,
                robot.arm_joints_name,
                robot.base_link,
                robot.ee_link_name,
            )
            for arm, robot in self.robots.items()
        }

    def root(self, arm):
        robot = self.robots[arm]
        value = self.manager.get_link_pose(robot, robot.base_link, is_relative=True)[0]
        return transform(value[:3], value[3:])

    def check(self):
        checks = {}
        for arm, robot in self.robots.items():
            q = self.manager.get_joint(robot)[0]
            predicted = self.root(arm) @ self.fk[arm].matrix(q)
            measured = self.manager.get_real_endpose(robot)[0]
            position_error = np.linalg.norm(predicted[:3, 3] - measured[:3])
            rotation_error = Rotation.from_matrix(
                predicted[:3, :3] @ transform(measured[:3], measured[3:])[:3, :3].T
            ).magnitude()
            checks[arm] = {
                "position_error_m": float(position_error),
                "rotation_error_rad": float(rotation_error),
                "passed": bool(position_error < 0.002 and rotation_error < 0.01),
            }
        if not all(check["passed"] for check in checks.values()):
            raise ValueError(f"Robot FK does not match measured link6: {checks}")
        return {"arms": checks, "passed": True, "physical_steps": 0}

    def preview(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError("Expected H50 x 14 proposal")
        checks = self.check()
        roots = {arm: self.root(arm) for arm in ("left", "right")}
        trajectory = [
            dict(
                index=index,
                **{
                    arm: dict(
                        **pose(
                            roots[arm] @ self.fk[arm].matrix(row[offset : offset + 6])
                        ),
                        gripper_closed=bool(row[offset + 6] < 0.5),
                        gripper_opening=float(row[offset + 6]),
                    )
                    for arm, offset in (("left", 0), ("right", 7))
                },
            )
            for index, row in enumerate(actions)
        ]
        return {
            "trajectory": trajectory,
            "measured_fk_check": checks,
            "physical_steps": 0,
            "frame": "environment_origin",
            "interpretation": "kinematic_targets_not_object_future",
        }

    def target(self, targets):
        if set(targets) != {"left", "right"}:
            raise ValueError("Explicit left/right targets required")
        self.check()
        action, diagnostics = [], {}
        for arm in ("left", "right"):
            robot, target = self.robots[arm], targets[arm]
            validate_eef_target(target, require_gripper=True)
            key = self.manager.robot_key[self.manager.robot_list.index(robot)]
            limits = (
                key.data.soft_joint_pos_limits[0, robot.arm_joint_indices].cpu().numpy()
            )
            joints, diagnostics[arm] = self.fk[arm].bounded_target(
                self.manager.get_joint(robot)[0], limits, self.root(arm), target
            )
            opening = target.get("gripper_opening", float(not target["gripper_closed"]))
            action.extend([*joints.tolist(), opening])
        return {"action": action, "diagnostics": diagnostics, "physical_steps": 0}
