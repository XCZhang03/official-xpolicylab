"""Policy-side motion toolkit for RoboDojo agent bundles.

Entry points:
    pose_math(arguments)          same dict API and results as the robodojo_pose_math tool
    DualArm()                     X5 FK, observation check and one bounded step_eef update
    policy_actions(...)           planner samples -> 25 Hz 14-D joint rows
    parse_reply(reply)            (meta, {camera: image}) from a robodojo_* reply
    hold / set_gripper / approach rows for waiting, gripper changes and bounded IK steps
    ee_row / ee_rows               official native EEF rows (robodojo_step_ee; env IK)
    eef_row / eef_rows / step_eef / servo / ik / execute   bounded EEF motion on joint rows
    Planner(config="official")    cuRobo free-space planning (needs the `planning` extra)
    shared_planner()              the process-wide warmed Planner

Everything runs on the policy side with only observations as input, so bundles behave
the same in harness exploration, isolated rehearsal and official XPolicyLab evaluation.
"""
from .kinematics import ArmFK, DualArm, pose, robot_spec, transform, validate_eef_target
from .motion import (ee_row, ee_rows, eef_row, eef_rows, episode_ended, execute, ik, pose_error,
                     servo, step_eef)
from .pose import pose_math
from .rows import approach, hold, parse_reply, set_gripper
from .trajectory import (GOAL_POSITION_TOLERANCE_M, GOAL_ROTATION_TOLERANCE_RAD,
                         JOINT_TARGET_CONTRACT, MAX_TRAJECTORY_ACTIONS, policy_actions)

__version__ = "0.2.0"


def Planner(*args, **kwargs):  # noqa: N802 - lazy import keeps torch/cuRobo optional
    from .planning import Planner as _Planner
    return _Planner(*args, **kwargs)


def shared_planner(config="official"):
    """Process-wide warmed planner (built once at adapter load, reused by bundles)."""
    from .planning import shared_planner as _shared
    return _shared(config)
