"""End-effector motion on the official interface.

The official evaluation executes two kinds of 25 Hz actions, both one step per row:

- joint rows (robodojo_step, 14-D), and
- native EEF rows (robodojo_step_ee, 16-D): link6 poses that the environment's own
  cuRobo IK turns into joint targets.

    ee_row(meta, targets)          one native EEF row (unset arms hold their measured pose)
    ee_rows(meta, waypoints)       several native EEF rows

The former harness tool robodojo_step_eef instead solved one *bounded* DLS update
from the measured joints per waypoint. It is rebuilt on joint rows:

    eef_row(state, targets)        one bounded update for both arms -> one row
    eef_rows(state, waypoints)     a chunk of rows, each predicted from the previous
                                   (open loop, one robodojo_step call)
    step_eef(ctx, waypoints)       the former tool exactly: one row per step, each
                                   solved from the measured state (closed loop)
    servo(ctx, arm, target)        repeat closed-loop updates until the measured pose
                                   reaches the target
    ik(arm, seed, target)          full IK solve (no step bound) for planning and checks
    execute(ctx, rows)             send joint (14) or EEF (16) rows in chunks of <= 50;
                                   stop when the episode ends

A waypoint is {"left": target?, "right": target?}. A target is {"position",
"quaternion_wxyz"} in the environment frame, plus optional "gripper_opening" in
[0, 1] or "gripper_closed" (bool). An arm without a target holds its measured
joints and commanded gripper.
"""
from __future__ import annotations

import json

import numpy as np

from .kinematics import ARMS, DualArm, transform
from .rows import parse_reply
from .trajectory import GOAL_POSITION_TOLERANCE_M, GOAL_ROTATION_TOLERANCE_RAD, MAX_TRAJECTORY_ACTIONS

OFFSET = {"left": 0, "right": 7}
_KINEMATICS = None


def _kinematics(kinematics=None):
    global _KINEMATICS
    if kinematics is not None:
        return kinematics
    if _KINEMATICS is None:
        _KINEMATICS = DualArm()
    return _KINEMATICS


def _opening(target, current):
    if "gripper_opening" in target:
        opening = float(target["gripper_opening"])
        if not 0.0 <= opening <= 1.0:
            raise ValueError("gripper_opening must be in [0, 1]")
        return opening
    if "gripper_closed" in target:
        if type(target["gripper_closed"]) is not bool:
            raise ValueError("gripper_closed must be a bool")
        return 0.0 if target["gripper_closed"] else 1.0
    return float(current)


def _check_waypoint(waypoint):
    if not isinstance(waypoint, dict) or not waypoint or set(waypoint) - set(ARMS):
        raise ValueError("A waypoint maps 'left' and/or 'right' to a target")


def eef_row(state, targets, *, kinematics=None):
    """One bounded DLS update toward ``targets`` from ``state``: (row, diagnostics).

    Per arm and step: the pose error is capped at 2 cm / 0.1 rad and each joint moves
    at most 0.05 rad (joint limits respected). This is exactly one former step_eef
    waypoint. ``diagnostics[arm]`` holds the error to the target *before* the step.
    """
    kinematics = _kinematics(kinematics)
    _check_waypoint(targets)
    row = np.asarray(state, dtype=np.float32).reshape(14).copy()
    diagnostics = {}
    for arm, target in targets.items():
        offset = OFFSET[arm]
        joints, diagnostics[arm] = kinematics.step_toward(arm, row[offset:offset + 6].astype(float), target)
        row[offset:offset + 6] = joints
        row[offset + 6] = _opening(target, row[offset + 6])
    return row, diagnostics


def eef_rows(state, waypoints, *, kinematics=None):
    """Open-loop chunk: one bounded update per waypoint, chained through predicted joints.

    One ``robodojo_step`` call executes it. Nothing is measured between rows, so
    tracking lag and contact are not corrected. Use step_eef or servo when accuracy
    near contact matters. Returns (N x 14 rows, per-row diagnostics).
    """
    rows, diagnostics = [], []
    current = np.asarray(state, dtype=np.float32).reshape(14)
    for waypoint in waypoints:
        current, info = eef_row(current, waypoint, kinematics=kinematics)
        rows.append(current)
        diagnostics.append(info)
    return np.asarray(rows, dtype=np.float32).reshape(-1, 14), diagnostics


def _gripper_index(arm):
    return OFFSET[arm] + 6


def ee_row(meta, targets):
    """One 16-D native EEF row: [left x,y,z,qw,qx,qy,qz, gripper, right ..., gripper].

    ``meta`` is parsed observation metadata (for arms without a target: their measured
    link6 pose and commanded gripper). The environment solves each arm with its own
    cuRobo IK, with no step bound: a far target is reached in one step (10 physics
    ticks, interpolated in joint space), and IK may pick a different arm configuration.
    If IK fails, that arm keeps its previous target. Keep consecutive targets close and
    check the measured pose.
    """
    if not isinstance(targets, dict) or set(targets) - set(ARMS):
        raise ValueError("targets maps 'left' and/or 'right' to a pose target")
    row = []
    for index, arm in enumerate(ARMS):
        target = targets.get(arm)
        if target is None:
            position, quaternion = meta["eef_positions"][index], meta["eef_quaternions_wxyz"][index]
            opening = float(meta["states"][_gripper_index(arm)])
        else:
            position, quaternion = target["position"], target["quaternion_wxyz"]
            opening = _opening(target, meta["states"][_gripper_index(arm)])
        quaternion = np.asarray(quaternion, dtype=float)
        norm = np.linalg.norm(quaternion)
        if len(position) != 3 or quaternion.shape != (4,) or not np.isfinite(norm) or abs(norm - 1) > 1e-2:
            raise ValueError(f"{arm}: position must be 3-D and quaternion_wxyz a unit quaternion")
        quaternion = quaternion / norm
        if quaternion[0] < 0:
            quaternion = -quaternion
        row += [float(v) for v in position] + quaternion.tolist() + [opening]
    return np.asarray(row, dtype=np.float32)


def ee_rows(meta, waypoints):
    """Native EEF rows for a list of waypoints; unset arms hold the pose in ``meta``."""
    return np.asarray([ee_row(meta, waypoint) for waypoint in waypoints], dtype=np.float32).reshape(-1, 16)


def episode_ended(reply):
    """True when a harness reply reports native termination or truncation.

    Official evaluation never reports this: it cancels the bundle instead (the next
    ctx.call raises), so code must not wait for this flag.
    """
    def ended(value):
        if isinstance(value, dict):
            return value.get("episode_ended") is True or any(ended(v) for v in value.values())
        return isinstance(value, list) and any(ended(v) for v in value)
    return any(ended(json.loads(block["text"])) for block in reply.get("content", [])
               if block.get("type") == "text")


def execute(ctx, rows, *, chunk=MAX_TRAJECTORY_ACTIONS):
    """Send rows in chunks of <= ``chunk`` (max 50): 14-D joint rows with robodojo_step,
    16-D native EEF rows with robodojo_step_ee.

    Returns (last reply, its metadata). Stops early if the episode ends. The reply of
    each call is the observation after its last row, so a smaller chunk means more
    frequent feedback at the cost of more calls (the same number of steps).
    """
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] not in (14, 16) or not len(rows):
        raise ValueError("Expected N x 14 joint rows or N x 16 native EEF rows")
    tool = "robodojo_step" if rows.shape[1] == 14 else "robodojo_step_ee"
    chunk = int(chunk)
    if not 1 <= chunk <= MAX_TRAJECTORY_ACTIONS:
        raise ValueError(f"chunk must be 1..{MAX_TRAJECTORY_ACTIONS}")
    reply = None
    for start in range(0, len(rows), chunk):
        reply = ctx.call(tool, actions=rows[start:start + chunk].tolist())
        if episode_ended(reply):
            break
    return reply, parse_reply(reply, images=False)[0]


def step_eef(ctx, waypoints, *, state=None, kinematics=None):
    """The former robodojo_step_eef tool: one row per waypoint, each from measured joints.

    Each waypoint costs one step and one robodojo_step call, and the reply's measured
    state seeds the next solve. ``state`` defaults to a fresh observation. Returns
    (last reply, its metadata, per-waypoint diagnostics).
    """
    if state is None:
        state = parse_reply(ctx.call("robodojo_observe"), images=False)[0]["states"]
    reply, meta, diagnostics = None, None, []
    for waypoint in waypoints:
        row, info = eef_row(state, waypoint, kinematics=kinematics)
        diagnostics.append(info)
        reply = ctx.call("robodojo_step", actions=[row.tolist()])
        meta = parse_reply(reply, images=False)[0]
        state = meta["states"]
        if episode_ended(reply):
            break
    return reply, meta, diagnostics


def pose_error(meta, arm, target):
    """(position m, rotation rad) error of the measured link6 pose in ``meta`` to ``target``."""
    from scipy.spatial.transform import Rotation
    index = ARMS.index(arm)
    measured = transform(meta["eef_positions"][index], meta["eef_quaternions_wxyz"][index])
    goal = transform(target["position"], target["quaternion_wxyz"])
    return (float(np.linalg.norm(goal[:3, 3] - measured[:3, 3])),
            float(Rotation.from_matrix(goal[:3, :3] @ measured[:3, :3].T).magnitude()))


def servo(ctx, arm, target, *, max_steps=50, rows_per_call=1,
          position_tol_m=GOAL_POSITION_TOLERANCE_M, rotation_tol_rad=GOAL_ROTATION_TOLERANCE_RAD,
          state=None, kinematics=None):
    """Closed-loop approach: bounded updates toward one pose until the *measured* pose is
    within tolerance or ``max_steps`` steps have run.

    ``rows_per_call`` > 1 predicts that many rows open loop per call (fewer calls,
    coarser feedback). Returns (last reply, metadata, {"reached", "steps",
    "position_error_m", "rotation_error_rad"}); errors are measured.
    """
    if state is None:
        meta = parse_reply(ctx.call("robodojo_observe"), images=False)[0]
        reply = None
    else:
        meta, reply = {"states": state}, None
    steps = 0
    while True:
        if "eef_positions" in meta:
            position_error, rotation_error = pose_error(meta, arm, target)
            if position_error <= position_tol_m and rotation_error <= rotation_tol_rad:
                break
        if steps >= max_steps or (reply is not None and episode_ended(reply)):
            break
        count = min(int(rows_per_call), max_steps - steps)
        rows, _ = eef_rows(meta["states"], [{arm: target}] * count, kinematics=kinematics)
        reply = ctx.call("robodojo_step", actions=rows.tolist())
        meta = parse_reply(reply, images=False)[0]
        steps += count
    if "eef_positions" not in meta:
        meta = parse_reply(ctx.call("robodojo_observe"), images=False)[0]
        position_error, rotation_error = pose_error(meta, arm, target)
    return reply, meta, {"reached": position_error <= position_tol_m and rotation_error <= rotation_tol_rad,
                         "steps": steps, "position_error_m": position_error,
                         "rotation_error_rad": rotation_error}


def ik(arm, seed, target, *, max_iterations=400, position_tol_m=1e-4, rotation_tol_rad=1e-3,
       kinematics=None):
    """Joint solution for a link6 pose by iterating the bounded update from ``seed``.

    Local and seed-dependent (the nearest solution). It does not check collisions.
    Returns (joints6, {"converged", "iterations", "position_error_m",
    "rotation_error_rad"}). Use it to test reachability or to pick a joint goal;
    move there with the planner or bounded rows, not one large jump.
    """
    kinematics = _kinematics(kinematics)
    joints = np.asarray(seed, dtype=float).reshape(6)
    info = {}
    for iteration in range(int(max_iterations) + 1):
        step, info = kinematics.step_toward(arm, joints, target)
        if (info["target_position_error_m"] <= position_tol_m
                and info["target_rotation_error_rad"] <= rotation_tol_rad):
            break
        if iteration < max_iterations:
            joints = np.asarray(step, dtype=float)
    converged = (info["target_position_error_m"] <= position_tol_m
                 and info["target_rotation_error_rad"] <= rotation_tol_rad)
    return joints, {"converged": bool(converged), "iterations": iteration,
                    "position_error_m": info["target_position_error_m"],
                    "rotation_error_rad": info["target_rotation_error_rad"]}
