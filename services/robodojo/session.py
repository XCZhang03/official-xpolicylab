"""One no-rollback RoboDojo episode owned by the native simulator process."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .io import require, write_json
from .kinematics import DualKinematics, measured_joint_state, transform, validate_eef_target
from .trajectory import (
    GOAL_POSITION_TOLERANCE_M,
    GOAL_ROTATION_TOLERANCE_RAD,
    JOINT_TARGET_CONTRACT,
    MAX_TRAJECTORY_ACTIONS,
    policy_actions,
)

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def register_native_evaluation(env):
    """Run the same task-check registration used by ``EvalEnv.run_eval``."""
    env.run_reward()
    if hasattr(env, "get_score"):
        env.get_score()
    names = ("check_list", "final_check_list", "trigger_check_list")
    counts = {
        name: [len(group) for group in getattr(env.reward_manager, name)]
        for name in names
    }
    require(
        all(
            sum(counts[name][index] for name in names) > 0
            for index in range(env.num_envs)
        ),
        "Native task completion conditions are empty; refusing vacuous success",
    )
    support_envs = []
    if getattr(env, "interact", False) and hasattr(env, "query_support_arm_traj"):
        support_envs = list(env.get_running_env_idx_list())
        for env_idx in support_envs:
            env.query_support_arm_traj(env_idx=env_idx)
    return {
        "registered": True,
        "source": "native EvalEnv.run_eval preamble",
        "condition_group_counts": counts,
        "initial_support_arm_query_envs": support_envs,
    }


class RoboDojoSession:
    """Simulator-side controller, planner cache, recorder, and RPC surface."""

    def __init__(self, env, output, task, *, enforce_step_limit=True):
        self.env, self.output = env, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode_id, self.step_id = None, 0
        self.terminated = self.truncated = self.success = False
        self.poisoned, self.finished = True, False
        self.video_writer, self.video_frames = None, 0
        self.finish_reason = None
        self._motion_plans = OrderedDict()
        self._motion_plan_limit = 16
        self.native_step_limit = int(env.step_lim)
        require(self.native_step_limit > 0, "Native task step limit must be positive")
        # The legacy setup flag selects episode mode, not whether time is bounded.
        self.episode_mode = "formal" if enforce_step_limit else "exploration"
        self.episode_step_limit = self.native_step_limit
        env.enforce_step_limit = True
        physics_dt = float(env.dt)
        ticks_per_observation = int(env.obs_manager.collect_interval)
        control_dt = physics_dt * ticks_per_observation
        require(
            np.isclose(
                control_dt,
                1.0 / float(env.obs_manager.collect_freq),
                atol=1e-9,
                rtol=0.0,
            ),
            "RoboDojo observation frequency and collect interval disagree",
        )
        self.metadata = {
            "task": task,
            "instruction": task,
            "simulator": "RoboDojo",
            "robot_adapter": "dual_arx_x5",
            "control_dt": control_dt,
            "control_frequency_hz": 1.0 / control_dt,
            "physics_dt": physics_dt,
            "physics_frequency_hz": 1.0 / physics_dt,
            "inner_control_ticks_per_observation": ticks_per_observation,
            "action_layer": "25Hz absolute dual-arm joint targets",
            "curobo_execution_layer": "25Hz absolute dual-arm joint targets via native take_action",
            "torque_layer": "PhysX implicit-actuator PD",
            "step_limit_enforced": True,
            "native_step_limit_enforced": True,
            "episode_mode": self.episode_mode,
            "native_reference_step_limit": self.native_step_limit,
            "episode_step_limit": self.episode_step_limit,
            "interaction_step_guidance": (
                "The original native task step limit is enforced in every episode; "
                "conserve interaction steps and batch predictable motion."
            ),
            "action_dim": 14,
            "action_horizon": 50,
            "frame": "environment_origin",
            "eef_link": "link6",
            "no_rollback": True,
            "supports_fk_preview": True,
            "supports_temporal_observations": True,
            "temporal_observation_frequency_hz": 1.0 / control_dt,
            "cameras": list(CAMERAS),
            "gripper_semantics": "continuous_0_closed_1_open",
            "free_space_world_model": {
                "privileged_task_objects": False,
                "depth_reconstruction": False,
                "static_table": True,
                "robot_self_collision": True,
                "other_arm_collision": False,
                "dynamic_object_collision": False,
                "planner_gripper_configuration": "fixed cuRobo retract/default state",
            },
        }
        vision_cfg = getattr(env.obs_manager, "obs_config", {}).get("vision", {})
        self.metadata.update(
            supports_depth=bool(vision_cfg.get("depth", False)),
            supports_camera_calibration=bool(
                vision_cfg.get("intrinsic_matrix", False)
                or vision_cfg.get("extrinsic_matrix", False)
            ),
            depth_representation="distance_to_image_plane_meters",
            camera_extrinsic_convention="camera_to_world_usd",
            camera_optical_convention=(
                "USD +Y up/-Z forward; RGB/depth intrinsics use CV +Y down/+Z forward"
            ),
        )
        # The session records one frame per actual 25 Hz action/control block.
        # Disable only EvalEnv's duplicate stream, not task checks or rendering.
        env._stream_vision = lambda *args, **kwargs: None

    def _operator_reward_state(self):
        """Return native evaluation state for the trusted operator only.

        This payload is persisted beside the simulator run, outside the agent
        workspace.  It must never be returned by the MCP bridge.
        """

        manager = self.env.reward_manager
        gated_scores = getattr(manager, "_gated_score_lst", None)
        score_percent = None
        if gated_scores is not None and len(gated_scores) > 0:
            score_percent = float(gated_scores[0])
        groups = {}
        for name in ("check_list", "final_check_list", "trigger_check_list"):
            values = getattr(manager, name, None)
            groups[name] = len(values[0]) if values and len(values) > 0 else 0
        return {
            "native_reward": 1.0 if self.success else 0.0,
            "native_success": bool(self.success),
            "native_score_percent": score_percent,
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "native_end_flag": bool(self.env.end_flag[0]),
            "valid_for_success_rate": bool(
                (self.terminated or self.truncated)
                and 0 not in getattr(self.env, "unstable_envs", set())
                and not self.poisoned
            ),
            "unstable_native_layout": bool(
                0 in getattr(self.env, "unstable_envs", set())
            ),
            "condition_group_counts": groups,
            "score_completed_count": int(
                getattr(manager, "score_completed_count", [0])[0]
            ),
            "final_score_completed_count": int(
                getattr(manager, "final_score_completed_count", [0])[0]
            ),
        }

    def _write_operator_state(self, reason=None):
        """Publish an atomic, mode-0600 snapshot for the operator dashboard."""

        if self.episode_id is None:
            return
        observation = (
            f"{self.episode_id}/observations/{self.step_id:06d}.npz"
            if hasattr(self, "obs")
            else None
        )
        state = {
            "schema": "robodojo_operator_state_v1",
            "simulator_pid": os.getpid(),
            "updated_at_unix_s": time.time(),
            "task": self.metadata["task"],
            "instruction": self.metadata.get("instruction"),
            "episode_id": self.episode_id,
            "step_id": int(self.step_id),
            "native_reference_step_limit": self.native_step_limit,
            "step_limit_enforced": True,
            "native_step_limit_enforced": bool(self.env.enforce_step_limit),
            "episode_step_limit": self.episode_step_limit,
            "episode_mode": self.episode_mode,
            "exploration_limit_reached": bool(self.episode_mode == "exploration" and self.truncated),
            "observation_file": observation,
            "cameras": list(CAMERAS),
            "reason": reason,
            "reward": self._operator_reward_state(),
        }
        path = self.output / "operator_state.json"
        write_json(path, state)
        path.chmod(0o600)

    def _check_identity(self, episode_id, step_id):
        require(
            not self.poisoned
            and episode_id == self.episode_id
            and step_id == self.step_id,
            "Stale episode/tick or uncertain simulator state",
        )

    def _observe(self, record=False):
        raw = self.env.get_obs()
        state = raw["state"]
        # Upstream puts last commanded grippers in raw["state"]; use physical
        # proprioception so state and action labels remain distinct.
        states = measured_joint_state(self.env.robot_manager)
        images, depth_fields, camera_parameters = {}, {}, {}
        aliases = {
            "cam_high": ("cam_high", "cam_head", "head_camera", "top_camera"),
            "cam_left_wrist": ("cam_left_wrist", "left_camera"),
            "cam_right_wrist": ("cam_right_wrist", "right_camera"),
        }
        for key, names in aliases.items():
            source = next((name for name in names if name in raw["vision"]), None)
            require(
                source is not None,
                f"Missing {key}; native cameras={list(raw['vision'])}",
            )
            camera_data = raw["vision"][source]
            images[key] = np.asarray(camera_data["color"], dtype=np.uint8)[..., :3]
            if "depth" in camera_data:
                depth = np.asarray(camera_data["depth"], dtype=np.float32)
                if depth.ndim >= 1 and depth.shape[-1] == 1:
                    depth = depth.squeeze(-1)
                require(
                    depth.shape == images[key].shape[:2],
                    f"Invalid {key} depth shape: {depth.shape}",
                )
                depth_fields[f"{key}_depth_m"] = depth

            camera_manager = getattr(self.env, "camera_manager", None)
            camera_names = (
                list(getattr(camera_manager, "camera_names", [[]])[0])
                if camera_manager
                else []
            )
            if source in camera_names and self.metadata["supports_camera_calibration"]:
                camera_id = camera_names.index(source)
                camera_parameters[key] = {
                    "camera_name": source,
                    "resolution": [
                        int(images[key].shape[1]),
                        int(images[key].shape[0]),
                    ],
                    "intrinsic_matrix": np.asarray(
                        camera_manager.get_camera_intrinsics(camera_id, 0),
                        dtype=np.float64,
                    ).tolist(),
                    "camera_to_world_usd": np.asarray(
                        camera_manager.get_camera_extrinsics(camera_id, 0),
                        dtype=np.float64,
                    ).tolist(),
                    "extrinsic_convention": "camera_to_world_usd",
                    "optical_convention": (
                        "USD +Y up/-Z forward; convert to CV +Y down/+Z forward for pinhole projection"
                    ),
                    "depth": {
                        "available": f"{key}_depth_m" in depth_fields,
                        "type": "distance_to_image_plane",
                        "unit": "meters",
                        "invalid_values": "non-finite or <= 0",
                    },
                }
        poses = np.asarray(
            [state[f"{arm}_ee_pose"] for arm in ("left", "right")], dtype=np.float32
        )
        environment_origins = getattr(self.env, "env_origins", None)
        require(environment_origins is not None, "Missing native environment origins")
        environment_origin = environment_origins[0]
        if hasattr(environment_origin, "detach"):
            environment_origin = environment_origin.detach()
        if hasattr(environment_origin, "cpu"):
            environment_origin = environment_origin.cpu()
        environment_origin = np.asarray(environment_origin, dtype=np.float64)
        require(
            environment_origin.shape == (3,) and np.isfinite(environment_origin).all(),
            "Invalid native environment origin",
        )
        self.obs = dict(
            **images,
            **depth_fields,
            states=states,
            eef_positions=poses[:, :3],
            eef_quaternions_wxyz=poses[:, 3:],
            environment_origin_world_m=environment_origin,
            instruction=raw["instruction"],
            total_interaction_steps=self.step_id,
            camera_parameters=camera_parameters,
            # The official policy observation reports these last commanded grippers;
            # the contract substitutes them only for the official profile.
            commanded_gripper_openings=np.asarray(
                [np.asarray(state.get(f"{arm}_ee_joint_state", [np.nan]), dtype=np.float64).reshape(-1)[0]
                 for arm in ("left", "right")], dtype=np.float64),
        )
        if record:
            directory = self.episode_dir / "observations"
            directory.mkdir(exist_ok=True)
            recorded = {
                key: value
                for key, value in self.obs.items()
                if key != "camera_parameters"
            }
            np.savez_compressed(directory / f"{self.step_id:06d}.npz", **recorded)
            if camera_parameters:
                write_json(
                    directory / f"{self.step_id:06d}_camera_parameters.json",
                    camera_parameters,
                )
            if self.video_writer is None:
                import imageio.v2 as imageio

                self.video_writer = imageio.get_writer(
                    str(self.output / "sensors.mp4"),
                    fps=1.0 / self.metadata["control_dt"],
                    codec="libx264",
                    pixelformat="yuv420p",
                    macro_block_size=2,
                    output_params=["-movflags", "+faststart"],
                )
            frames = [
                np.asarray(Image.fromarray(images[key]).resize((640, 360)))
                for key in CAMERAS
            ]
            self.video_writer.append_data(np.concatenate(frames, axis=1))
            self.video_frames += 1
            self._write_operator_state()
        return self.obs

    def _motion_state(self):
        state = np.asarray(self.obs["states"], dtype=np.float32).reshape(14)
        require(np.isfinite(state).all(), "Invalid current state for motion planning")
        return state.copy()

    def _ensure_curobo_planner(self, arm):
        require(arm in ("left", "right"), f"Invalid motion-planning arm: {arm}")
        robot = self.kinematics.robots[arm]
        manager = self.env.robot_manager
        planner = manager.planner.get(robot.robot_name)
        if planner is None:
            manager._setup_planner(robot)
            planner = manager.planner.get(robot.robot_name)
        require(planner is not None, f"No cuRobo planner available for {arm}")
        return robot, planner

    def _motion_plan_diagnostics(self, arm, joints, target):
        actual = self.kinematics.root(arm) @ self.kinematics.fk[arm].matrix(joints)
        goal = transform(target["position"], target["quaternion_wxyz"])
        return {
            "final_position_error_m": float(
                np.linalg.norm(actual[:3, 3] - goal[:3, 3])
            ),
            "final_rotation_error_rad": float(
                Rotation.from_matrix(goal[:3, :3] @ actual[:3, :3].T).magnitude()
            ),
            "final_joint_state": np.asarray(joints, dtype=np.float32).tolist(),
            "physical_tracking_verified": False,
        }

    def _store_motion_plan(self, entry):
        plan_id = uuid.uuid4().hex
        entry = dict(entry, motion_plan_id=plan_id)
        self._motion_plans[plan_id] = entry
        self._motion_plans.move_to_end(plan_id)
        while len(self._motion_plans) > self._motion_plan_limit:
            self._motion_plans.popitem(last=False)
        return entry

    def _update_terminal_state(self):
        ended = bool(self.env.end_flag[0])
        limit_reached = self.step_id >= self.episode_step_limit
        self.success = bool(ended and self.env.success[0])
        self.truncated = bool(limit_reached and not self.success)
        self.terminated = bool(ended and not self.truncated)
        return ended or limit_reached

    def _execute_motion_plan(self, plan_id):
        plan = self._motion_plans.get(str(plan_id))
        if plan is None:
            return {
                "status": "Plan_Not_Found",
                "motion_plan_id": str(plan_id),
                "executed": False,
                "reason": "Motion plan is missing or expired from the bounded cache.",
            }
        current = self._motion_state()
        if self.step_id != plan["step_id"] or not np.allclose(
            current, plan["start_state"], atol=2e-3, rtol=0.0
        ):
            return {
                "status": "Plan_Stale",
                "motion_plan_id": plan["motion_plan_id"],
                "executed": False,
                "reason": "An action ran or measured state changed since preview; re-plan before execution.",
                "planned_start_state": plan["start_state"].tolist(),
                "current_state": current.tolist(),
            }

        frames, steps = [], []
        actions = plan["actions"]
        for index, action in enumerate(actions):
            if self.terminated or self.truncated or self.finished:
                break
            result = self.chunk_step(
                [action],
                source="curobo_free_space",
                motion_plan_id=plan["motion_plan_id"],
                control_block_index=index,
            )
            frames.extend(result["frames"])
            steps.extend(result["steps"])

        self._motion_plans.pop(plan["motion_plan_id"], None)
        complete = len(steps) == len(actions)
        measured = self._motion_state()
        offset = 0 if plan["arm"] == "left" else 7
        tracking = self._motion_plan_diagnostics(
            plan["arm"], measured[offset:offset + 6], plan["target"]
        )
        tracking["physical_tracking_verified"] = True
        goal_reached = (
            tracking["final_position_error_m"] <= GOAL_POSITION_TOLERANCE_M
            and tracking["final_rotation_error_rad"] <= GOAL_ROTATION_TOLERANCE_RAD
        )
        return {
            # Success means all commands ran, not that the robot reached the goal.
            "status": "Success" if complete else "Partial",
            "motion_plan_id": plan["motion_plan_id"],
            "arm": plan["arm"],
            "executed": bool(steps),
            "executed_control_blocks": len(steps),
            "planned_control_blocks": len(actions),
            "goal_reached": bool(goal_reached),
            "measured_goal_error": tracking,
            "goal_tolerances": {
                "position_m": GOAL_POSITION_TOLERANCE_M,
                "rotation_rad": GOAL_ROTATION_TOLERANCE_RAD,
            },
            "observation_frequency_hz": self.metadata["control_frequency_hz"],
            "frame_count": len(frames),
            "frames": frames,
            "steps": steps,
            "joint_target_contract": JOINT_TARGET_CONTRACT,
            "step_id": self.step_id,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "success": self.success,
        }

    def free_space_move(
        self,
        *,
        arm,
        target,
        preview_only=False,
        include_trajectory=False,
    ):
        """Plan internally with cuRobo; execute ordinary 25Hz joint actions."""
        arm = str(arm).strip().lower()
        require(
            type(preview_only) is bool and type(include_trajectory) is bool,
            "Mode flags must be boolean",
        )
        target = validate_eef_target(target)
        robot, planner = self._ensure_curobo_planner(arm)
        start_state = self._motion_state()
        offset = 0 if arm == "left" else 7
        current_joints = start_state[offset : offset + 6]
        start_opening = float(start_state[offset + 6])
        target_opening = (
            start_opening
            if target.get("gripper_opening") is None
            else float(target["gripper_opening"])
        )
        result = planner.plan_path(
            current_joints,
            target["position"] + target["quaternion_wxyz"],
            real_robot_pose=robot.entity_origin_pose,
        )
        if result.get("status") != "Success":
            return {
                "status": "Planning_Failed",
                "planner": "curobo",
                "arm": arm,
                "executed": False,
                "planner_result": result.get("status"),
                "failure_stage": result.get("failure_stage", "planning"),
                "reason": result.get("reason", "Planner returned no feasible trajectory"),
                "diagnostics": result.get("diagnostics", {}),
            }
        positions = np.asarray(result.get("position"), dtype=np.float32)
        actions = policy_actions(
            positions, start_state, arm, target_opening,
            # Pristine upstream plans omit the sampling period; read it from the solver.
            planner_dt=float(result.get("interpolation_dt")
                             or planner.motion_planner.trajopt_solver.config.interpolation_dt),
            control_dt=self.metadata["control_dt"],
        )
        diagnostics = self._motion_plan_diagnostics(arm, positions[-1], target)
        if (
            diagnostics["final_position_error_m"] > GOAL_POSITION_TOLERANCE_M
            or diagnostics["final_rotation_error_rad"] > GOAL_ROTATION_TOLERANCE_RAD
        ):
            return {
                "status": "Planning_Failed", "planner": "curobo", "arm": arm,
                "executed": False, "reason": "Planned endpoint exceeds goal tolerances",
                "diagnostics": diagnostics,
            }
        if len(actions) > MAX_TRAJECTORY_ACTIONS:
            return {
                "status": "Planning_Failed", "executed": False, "arm": arm,
                "failure_stage": "action_budget",
                "reason": f"Trajectory exceeds the {MAX_TRAJECTORY_ACTIONS}-step planning/preview limit; choose a nearer waypoint",
                "action_count": len(actions), "max_actions": MAX_TRAJECTORY_ACTIONS,
            }
        entry = self._store_motion_plan(
            {
                "arm": arm,
                "target": target,
                "start_state": start_state,
                "actions": actions,
                "step_id": int(self.step_id),
            }
        )
        response = {
            "status": "Success",
            "planner": "curobo",
            "arm": arm,
            "motion_plan_id": entry["motion_plan_id"],
            "executed": False,
            "preview_only": True,
            "start_state": start_state.tolist(),
            "final_arm_joint_state": positions[-1].tolist(),
            "target": target,
            "diagnostics": diagnostics,
            "timing": {
                "control_blocks": len(actions),
                "remaining_episode_steps": max(0, self.episode_step_limit - self.step_id),
                "fits_remaining_budget": len(actions) <= self.episode_step_limit - self.step_id,
                "control_frequency_hz": self.metadata["control_frequency_hz"],
                "simulator_advance_duration_s": len(actions) * self.metadata["control_dt"],
            },
            "joint_target_contract": JOINT_TARGET_CONTRACT,
            "goal_tolerances": {
                "position_m": GOAL_POSITION_TOLERANCE_M,
                "rotation_rad": GOAL_ROTATION_TOLERANCE_RAD,
            },
            "execution_contract": {
                "environment_action_rows": True,
                "executor": "same native take_action as robodojo_step and robodojo_step_eef",
                "benchmark_step_accounting": "one 14D action row per 25Hz step",
                "planned_velocities_executed": False,
                "success_means": "commands completed; inspect transition.goal_reached separately",
            },
            "collision_model": self.metadata["free_space_world_model"],
            "execute_with": {"motion_plan_id": entry["motion_plan_id"]},
        }
        if include_trajectory:
            response["trajectory_preview"] = {
                "actions": actions.tolist(),
                "frequency_hz": self.metadata["control_frequency_hz"],
                "action_count": len(actions),
                "includes_initial_state": False,
                "note": f"Complete executable targets, without display subsampling. Replay them in robodojo_step batches of at most {MAX_TRAJECTORY_ACTIONS}.",
            }
        if not preview_only:
            response["execution"] = self._execute_motion_plan(entry["motion_plan_id"])
            response["executed"] = response["execution"]["executed"]
            response["preview_only"] = False
        return response

    def reset(self, seed, source, policy_version):
        require(self.episode_id is None, "One fresh episode only; no rollback/reset")
        self.env.reset(seed=[seed])
        registration = register_native_evaluation(self.env)
        write_json(self.output / "native_evaluation.json", registration)
        self.episode_id = uuid.uuid4().hex
        self.episode_dir = self.output / self.episode_id
        self.episode_dir.mkdir()
        self.poisoned = False
        self.kinematics = DualKinematics(self.env)
        write_json(self.output / "fk_validation.json", self.kinematics.check())
        self._observe(record=True)
        fingerprints = {}
        for key, value in self.obs.items():
            if key == "camera_parameters":
                fingerprints[key] = {
                    "schema": "camera_parameters_v1",
                    "sha256": hashlib.sha256(
                        json.dumps(value, sort_keys=True).encode()
                    ).hexdigest(),
                }
                continue
            array = np.asarray(value)
            fingerprints[key] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        write_json(
            self.output / "initial_observation_fingerprint.json",
            {
                "fields": fingerprints,
                "scope": "RGB, depth when enabled, proprio, EEF, and instruction; not complete physics state",
                "bitwise_physics_reproducibility_guaranteed": False,
            },
        )
        self.metadata["instruction"] = self.obs["instruction"]
        identity = hashlib.sha256(self.obs["states"].tobytes()).hexdigest()
        result = {
            "episode_id": self.episode_id,
            "step_id": 0,
            "initial_state_hash": identity,
            "instruction": self.obs["instruction"],
            "metadata": dict(
                self.metadata,
                layout_id=seed,
                eval_seed=getattr(self.env, "eval_seed", None),
                initial_state_hash_scope="robot_proprio_only_not_complete_scene",
                policy_version=policy_version,
                controller_source=source,
                native_reset_settling_not_counted_as_policy_controls=True,
            ),
        }
        write_json(self.output / "reset.json", result)
        return result

    def chunk_step(self, actions, action_type="joint", **kwargs):
        """Execute rows through native take_action, one environment step each.

        ``joint``: 14-D [left joints 1..6, left gripper, right joints 1..6, right gripper].
        ``ee``: 16-D [left link6 x,y,z,qw,qx,qy,qz, left gripper, right ..., right gripper],
        the official EEF action: EvalEnv solves each arm with its own cuRobo IK
        (robot_manager.solve_ik) and an arm whose IK fails keeps its previous target.
        """
        require(
            not self.finished and not self.terminated and not self.truncated,
            "Episode finished",
        )
        require(action_type in ("joint", "ee"), "action_type must be joint or ee")
        width, grippers = (14, [6, 13]) if action_type == "joint" else (16, [7, 15])
        actions = np.asarray(actions, dtype=np.float32)
        require(
            actions.ndim == 2
            and actions.shape[1] == width
            and 1 <= len(actions) <= MAX_TRAJECTORY_ACTIONS
            and np.isfinite(actions).all(),
            f"Expected 1..{MAX_TRAJECTORY_ACTIONS} finite {width}D absolute actions",
        )
        require(
            np.all((actions[:, grippers] >= 0) & (actions[:, grippers] <= 1)),
            "Invalid gripper opening",
        )
        if action_type == "ee":
            norms = np.linalg.norm(actions[:, [3, 4, 5, 6, 11, 12, 13, 14]].reshape(-1, 2, 4), axis=2)
            require(np.all(np.abs(norms - 1) <= 1e-3), "EEF quaternions must be unit wxyz")
            # The official env builds its cuRobo planners (and IK) at startup; this
            # server defers them, so build them before the first EEF action.
            for arm in ("left", "right"):
                self._ensure_curobo_planner(arm)
        rows, frames = [], []
        for action in actions:
            if action_type == "joint":
                command = {
                    key: value
                    for arm, offset in (("left", 0), ("right", 7))
                    for key, value in (
                        (f"{arm}_arm_joint_state", action[offset : offset + 6]),
                        (f"{arm}_ee_joint_state", action[offset + 6 : offset + 7]),
                    )
                }
            else:
                command = {
                    key: value
                    for arm, offset in (("left", 0), ("right", 8))
                    for key, value in (
                        (f"{arm}_ee_pose", action[offset : offset + 7]),
                        (f"{arm}_ee_joint_state", action[offset + 7 : offset + 8]),
                    )
                }
            before = int(self.env.take_action_cnt[0])
            self.env.take_action(command)
            require(
                int(self.env.take_action_cnt[0]) == before + 1,
                "Native action was not executed exactly once",
            )
            self.step_id += 1
            ended = self._update_terminal_state()
            observation = self._observe(record=True)
            frames.append(observation)
            row = {
                "valid": True,
                # ``executed_action`` stays the 14-D joint contract; EEF rows are
                # recorded as sent, with the joints the native IK actually commanded
                # visible in the post-step measured state.
                **({"executed_action": action.copy(), "execution_mode": "policy_joint_target"}
                   if action_type == "joint" else
                   {"ee_action": action.copy(), "execution_mode": "policy_ee_target"}),
                "observation_step_id": self.step_id - 1,
                "measured_state": observation["states"].copy(),
                "step_id": self.step_id,
                "terminated": self.terminated,
                "truncated": self.truncated,
                "success": self.success,
                "source": "mcp_joint_action" if action_type == "joint" else "mcp_ee_action",
            }
            row.update(kwargs)
            rows.append(row)
            write_json(self.episode_dir / f"action_{self.step_id - 1:06d}.json", row)
            if ended:
                self._write_summary("terminal")
                break
        return {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "observation_frequency_hz": self.metadata["control_frequency_hz"],
            "frame_count": len(frames),
            "frames": frames,
            "steps": rows,
            "joint_target_contract": JOINT_TARGET_CONTRACT,
        }

    def _write_summary(self, reason):
        if self.episode_id is None:
            return
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        if self.finish_reason is not None:
            reason = self.finish_reason
        complete = self.terminated or self.truncated
        invalid = 0 in getattr(self.env, "unstable_envs", set())
        eligible = complete and not invalid and not self.poisoned
        score = None
        # Native run_eval awards success=1, otherwise the task's gated process
        # score. Early controller exit/timeout does not erase earned partial
        # credit; binary completion eligibility remains a separate decision.
        if not invalid:
            score = (
                1.0
                if self.success
                else (
                    float(self.env.reward_manager.get_score()[0]) / 100
                    if hasattr(self.env, "get_score")
                    else 0.0
                )
            )
        write_json(
            self.output / "evaluation_outcome.json",
            {
                "complete": complete,
                "valid_for_success_rate": eligible,
                "native_success": self.success if eligible else None,
                "native_score": score,
                "valid_for_score": score is not None,
                "score_source": "native_eval" if score is not None else "invalid_native_layout",
                "control_uncertain": bool(self.poisoned),
                "status": (
                    "invalid_native_layout"
                    if invalid
                    else "native_completed"
                    if eligible
                    else "incomplete"
                ),
                "reason": reason,
                "native_control_steps": self.step_id,
                "native_reference_step_limit": self.native_step_limit,
                "step_limit_enforced": True,
                "native_step_limit_enforced": bool(self.env.enforce_step_limit),
                "episode_step_limit": self.episode_step_limit,
            },
        )
        write_json(
            self.output / "summary.json",
            {
                "episode_id": self.episode_id,
                "step_id": self.step_id,
                "success": self.success,
                "terminated": self.terminated,
                "truncated": self.truncated,
                "complete": complete,
                "reason": reason,
                "video_frames": self.video_frames,
                "control_dt": self.metadata["control_dt"],
                "no_rollback": True,
            },
        )
        self._write_operator_state(reason=reason)

    def dispatch(self, op, args):
        if op == "metadata":
            return self.metadata
        if op == "reset":
            return self.reset(**args)
        self._check_identity(args["episode_id"], args["step_id"])
        data = {
            key: value
            for key, value in args.items()
            if key not in ("episode_id", "step_id")
        }
        if op == "teacher_observation":
            return self.obs
        if op == "fk_preview":
            return self.kinematics.preview(data["actions"])
        if op == "eef_joint_target":
            return self.kinematics.target(data["targets"])
        if op == "free_space_move":
            return self.free_space_move(**data)
        if op == "execute_motion_plan":
            return self._execute_motion_plan(data["motion_plan_id"])
        if op == "chunk_step":
            return self.chunk_step(**data)
        if op == "finish_pilot":
            self.finished = True
            self.finish_reason = data["reason"]
            self._write_summary(data["reason"])
            return {
                "episode_id": self.episode_id,
                "step_id": self.step_id,
                "success": self.success,
                "terminated": self.terminated,
                "truncated": self.truncated,
                "video_frames": self.video_frames,
                "control_dt": self.metadata["control_dt"],
                "reason": data["reason"],
            }
        raise ValueError("Unsupported no-rollback operation: " + op)
