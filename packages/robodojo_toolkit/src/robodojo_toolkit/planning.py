"""cuRobo free-space planning that returns executable 25 Hz joint rows.

Runs in-process on the policy side (no simulator): the same vendored RoboDojo planner,
construction and acceptance rules as the harness ``robodojo_free_space_move`` tool.
It is robot- and table-aware only; task objects are not in the collision model.

Requires the ``planning`` extra (torch, warp-lang and the pinned cuRobo build). Warp
compiles kernels on first use, so construct and ``warmup()`` a Planner at load time,
outside any per-step time limit.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np

from .kinematics import ARMS, DualArm, pose, transform
from .trajectory import (GOAL_POSITION_TOLERANCE_M, GOAL_ROTATION_TOLERANCE_RAD,
                         MAX_TRAJECTORY_ACTIONS, policy_actions)

CONTROL_DT_S = 0.04
CONFIGS = ("official", "harness")


def curobo_config(kind="official", directory=None):
    """Materialize a cuRobo robot config whose paths point at the packaged URDF.

    ``official`` uses the published RoboDojo template (curobo_tmp.yml), substituted the
    way upstream utils/update_embodiment_config_path.py does. Returns the file path.
    """
    if kind not in CONFIGS:
        raise ValueError(f"config must be one of {CONFIGS}")
    data = Path(__file__).resolve().parent / "data" / "x5"
    directory = Path(directory or tempfile.mkdtemp(prefix="robodojo-toolkit-"))
    directory.mkdir(parents=True, exist_ok=True)
    if kind == "official":
        text = (data / "curobo_tmp.yml").read_text()
        # The template's placeholder is the RoboDojo root; its paths are <root>/Assets/Robots/x5/*.
        text = text.replace("${ASSETS_PATH}/Assets/Robots/x5", str(data)).replace("$ASSETS_PATH/Assets/Robots/x5", str(data))
        if "ASSETS_PATH" in text:
            raise ValueError("Unresolved ASSETS_PATH placeholder in the cuRobo template")
    else:
        # The vendored cuRobo fork's x5_v2 config bakes in its author's paths.
        text = (data / "curobo_harness_x5_v2.yml").read_text().replace(
            "/home/kaslensu/workspace/isaacsim_project/curobo/x5", str(data))
    path = directory / f"curobo_{kind}.yml"
    path.write_text(text)
    return path


class Planner:
    """Per-arm cuRobo planners built exactly as RoboDojo's robot manager builds them."""

    def __init__(self, config="official", arm=None):
        cache = Path(os.environ.setdefault("WARP_CACHE_PATH", str(Path(tempfile.gettempdir()) / "robodojo-toolkit-warp")))
        seed = os.environ.get("ROBODOJO_TOOLKIT_WARP_SEED")
        if seed and Path(seed).is_dir() and not (cache.exists() and any(cache.iterdir())):
            # A read-only prebuilt kernel cache (image or checkpoint): copy it to a writable one.
            import shutil
            shutil.copytree(seed, cache, dirs_exist_ok=True)
        import warp as wp
        # Portable kernels: PTX for a baseline architecture that newer drivers JIT-load,
        # so a cache built once (shipped with a checkpoint) works on any Ampere+ GPU.
        wp.config.cuda_output = os.environ.get("ROBODOJO_TOOLKIT_WARP_OUTPUT", "ptx")
        wp.config.ptx_target_arch = int(os.environ.get("ROBODOJO_TOOLKIT_PTX_ARCH", "80"))
        from ._vendor.curobo_planner import CuroboPlanner

        self.kinematics = DualArm()
        spec = self.kinematics.spec
        self.config = config
        self.yml_path = curobo_config(config)
        self.planners = {}
        for name in (ARMS if arm is None else (arm,)):
            root = spec["roots"][name]
            self.planners[name] = CuroboPlanner(
                robot_origin_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                active_joints_name=spec["arm_joints"],
                all_joints=spec["arm_joints"],
                dt=spec["simulation_dt_s"],
                yml_path=str(self.yml_path),
                table_height=spec["table_height_m"] - root["position"][2],
            )

    def _root_pose(self, arm):
        root = self.kinematics.spec["roots"][arm]
        return list(root["position"]) + list(root["quaternion_wxyz"])

    def warmup(self):
        """Plan one short move per arm so JIT compilation happens now, not mid-episode."""
        for arm in self.planners:
            joints = np.zeros(6)
            start = self.kinematics.eef(arm, joints)
            self.planners[arm].plan_path(joints, start["position"] + start["quaternion_wxyz"],
                                         real_robot_pose=self._root_pose(arm))

    def plan(self, arm, state, target, gripper_opening=None):
        """Plan ``arm`` from a 14-D state to ``target`` = {position, quaternion_wxyz}.

        Returns {"status": "Success", "actions": N x 14 rows, ...} or a failure dict.
        Rows hold the other arm and ramp only this arm's gripper, like the harness tool.
        """
        state = np.asarray(state, dtype=np.float32).reshape(14)
        if arm not in self.planners:
            raise ValueError(f"No planner for arm {arm!r}")
        offset = 0 if arm == "left" else 7
        joints = state[offset:offset + 6]
        opening = float(state[offset + 6] if gripper_opening is None else gripper_opening)
        position = [float(x) for x in target["position"]]
        quaternion = [float(x) for x in target["quaternion_wxyz"]]
        result = self.planners[arm].plan_path(joints, position + quaternion,
                                              real_robot_pose=self._root_pose(arm))
        if result.get("status") != "Success":
            return {"status": "Planning_Failed", "arm": arm, "planner_result": result.get("status"),
                    "failure_stage": result.get("failure_stage", "planning"),
                    "reason": result.get("reason", "Planner returned no feasible trajectory")}
        positions = np.asarray(result["position"], dtype=np.float32)
        actions = policy_actions(positions, state, arm, opening,
                                 planner_dt=float(result["interpolation_dt"]), control_dt=CONTROL_DT_S)
        final = self.kinematics.eef(arm, positions[-1])
        goal, reached = transform(position, quaternion), transform(final["position"], final["quaternion_wxyz"])
        from scipy.spatial.transform import Rotation
        diagnostics = {
            "final_position_error_m": float(np.linalg.norm(goal[:3, 3] - reached[:3, 3])),
            "final_rotation_error_rad": float(Rotation.from_matrix(goal[:3, :3] @ reached[:3, :3].T).magnitude()),
            "final_joint_state": positions[-1].tolist(),
        }
        if (diagnostics["final_position_error_m"] > GOAL_POSITION_TOLERANCE_M
                or diagnostics["final_rotation_error_rad"] > GOAL_ROTATION_TOLERANCE_RAD):
            return {"status": "Planning_Failed", "arm": arm, "reason": "Planned endpoint exceeds goal tolerances",
                    "diagnostics": diagnostics}
        return {"status": "Success", "arm": arm, "actions": actions, "action_count": len(actions),
                "fits_one_step_call": len(actions) <= MAX_TRAJECTORY_ACTIONS, "diagnostics": diagnostics,
                "config": self.config}


_SHARED = {}


def shared_planner(config="official"):
    """One warmed Planner per process: the adapter builds it at load, bundles reuse it.

    With ROBODOJO_TOOLKIT_PLANNER_SOCKET set (the development container), this is a
    client of one long-lived planner process instead, started on first use.
    """
    if config not in _SHARED:
        from .service import SOCKET_ENV, connect
        if os.environ.get(SOCKET_ENV):
            _SHARED[config] = connect(os.environ[SOCKET_ENV], config)
        else:
            planner = Planner(config=config)
            planner.warmup()
            _SHARED[config] = planner
    return _SHARED[config]
