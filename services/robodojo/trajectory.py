"""Convert internal cuRobo samples into the shared 25Hz policy action format."""

from __future__ import annotations

import numpy as np

# Shared plan acceptance and measured goal-reached thresholds.
# Bound accumulated RGB/depth frames in one native RPC response.
MAX_TRAJECTORY_ACTIONS = 50
GOAL_POSITION_TOLERANCE_M = 0.001
GOAL_ROTATION_TOLERANCE_RAD = 0.005

# Commands, never measured observations.
JOINT_TARGET_CONTRACT = {
    "version": 2,
    "field": "executed_action",
    "shape": [14],
    "order": ["left_joint_1..6", "left_gripper", "right_joint_1..6", "right_gripper"],
    "arm_units": "radians",
    "gripper_units": "normalized_opening_0_closed_1_open",
    "reference": "absolute_joint_position",
    "alignment": "command endpoint for interval (observation_step_id, step_id]; measured_state is post-execution",
    "frequency_hz": 25,
    "replay": f"Send executed_action rows to robodojo_step in batches of 1..{MAX_TRAJECTORY_ACTIONS}; all motion interfaces use the same native interpolation. Equivalent commands do not guarantee identical physics from different initial scene states.",
}


def policy_actions(
    positions, initial_state, arm, target_opening, *, planner_dt, control_dt=0.04
):
    """Select each 40ms endpoint and the final endpoint; t=0 is not an action.

    Native take_action owns interpolation and zero velocity targets. The
    inactive arm holds its initial target; gripper opening follows elapsed time.
    """
    positions = np.asarray(positions, dtype=np.float32)
    initial_state = np.asarray(initial_state, dtype=np.float32)
    if arm not in ("left", "right") or initial_state.shape != (14,):
        raise ValueError("Expected left/right arm and a 14D initial state")
    if positions.ndim != 2 or positions.shape[1] != 6 or len(positions) < 2:
        raise ValueError("Expected at least two 6D planner states including t=0")
    if not np.isfinite(positions).all() or not np.isfinite(initial_state).all():
        raise ValueError("Joint targets must be finite")
    if not np.isfinite(target_opening) or not 0 <= target_opening <= 1 or np.any(
        (initial_state[[6, 13]] < 0) | (initial_state[[6, 13]] > 1)
    ):
        raise ValueError("Gripper openings must be in [0, 1]")
    if not np.isfinite(planner_dt) or planner_dt <= 0 or not np.isclose(control_dt, 0.04):
        raise ValueError("Require positive planner_dt and 25Hz control_dt")
    stride = round(control_dt / planner_dt)
    if stride < 1 or not np.isclose(stride * planner_dt, control_dt, atol=1e-9, rtol=0):
        raise ValueError("Planner samples must evenly divide the 25Hz control interval")
    offset = 0 if arm == "left" else 7
    if not np.allclose(positions[0], initial_state[offset:offset + 6], atol=0.002, rtol=0):
        raise ValueError("Planner trajectory must start at measured joints")
    intervals = len(positions) - 1
    indices = np.minimum(np.arange(stride, intervals + stride, stride), intervals)
    actions = np.repeat(initial_state[None], len(indices), axis=0)
    actions[:, offset:offset + 6] = positions[indices]
    alpha = indices / intervals
    actions[:, offset + 6] = (1 - alpha) * initial_state[offset + 6] + alpha * target_opening
    return actions
