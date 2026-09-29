"""Dense-to-policy endpoint alignment without starting a simulator."""
from types import SimpleNamespace
from collections import OrderedDict

import numpy as np
import pytest

from services.robodojo.session import RoboDojoSession
from services.robodojo.trajectory import policy_actions


def test_observations_use_measured_grippers_not_commanded_openings():
    from services.robodojo.kinematics import measured_joint_state
    left = SimpleNamespace(type="target", arm_name="left_arm", gripper_scale=[0, 2],
                           gripper_move={"sign": 1})
    right = SimpleNamespace(type="target", arm_name="right_arm", gripper_scale=[-1, 1],
                            gripper_move={"sign": -1})
    manager = SimpleNamespace(
        robot_list=[right, left],  # Explicit left/right output order.
        get_joint=lambda robot, **kwargs: {0: np.full(6, 0.1 if robot is left else 0.2)},
        get_end_effector_real_val=lambda robot, **kwargs: {0: [0.5, -0.5]},
        control_manager=SimpleNamespace(prev_control={"ignored_command": 1.0}),
    )
    state = measured_joint_state(manager)
    np.testing.assert_array_equal(state[[6, 13]], [0.25, 0.25])
    np.testing.assert_allclose(state[:6], 0.1)
    np.testing.assert_allclose(state[7:13], 0.2)


@pytest.mark.parametrize("arm,offset", [("left", 0), ("right", 7)])
@pytest.mark.parametrize("command_ticks", [1, 10, 12, 20, 21])
def test_policy_targets_use_40ms_endpoints_and_preserve_final_goal(arm, offset, command_ticks):
    start = np.arange(14, dtype=np.float32) / 20
    positions = start[offset:offset + 6] + np.arange(command_ticks + 1, dtype=np.float32)[:, None] / 100
    actions = policy_actions(positions, start, arm, 0.0, planner_dt=0.004)
    indices = [min(i, command_ticks) for i in range(10, command_ticks + 10, 10)]
    assert len(actions) == len(indices)
    np.testing.assert_array_equal(actions[:, offset:offset + 6], positions[indices])
    np.testing.assert_array_equal(actions[-1, offset:offset + 6], positions[-1])
    assert actions[-1, offset + 6] == 0.0
    other = 7 - offset
    np.testing.assert_array_equal(actions[:, other:other + 7],
                                  np.repeat(start[None, other:other + 7], len(actions), axis=0))
    np.testing.assert_allclose(actions[:, offset + 6],
                               (1 - np.asarray(indices) / command_ticks) * start[offset + 6])


def _session(tmp_path, limit=100):
    session = RoboDojoSession.__new__(RoboDojoSession)
    session.step_id = 7
    session.episode_id = "unit"
    session.episode_dir = tmp_path
    session.episode_step_limit = limit
    session.terminated = session.truncated = session.finished = session.success = False
    session._motion_plans = OrderedDict()
    session._motion_plan_limit = 16
    session.metadata = {"control_frequency_hz": 25, "control_dt": 0.04, "physics_dt": 0.004,
                        "inner_control_ticks_per_observation": 10, "free_space_world_model": {}}
    session.obs = {"states": np.zeros(14, dtype=np.float32)}
    submitted = []

    def take_action(command):
        submitted.append(command)
        session.env.take_action_cnt[0] += 1
        session.obs = {"states": np.concatenate([
            command["left_arm_joint_state"], command["left_ee_joint_state"],
            command["right_arm_joint_state"], command["right_ee_joint_state"]])}

    session.env = SimpleNamespace(take_action=take_action, take_action_cnt=[7],
                                  end_flag=[False], success=[False])
    session._observe = lambda **kwargs: session.obs
    session._write_summary = lambda reason: None
    session._motion_plan_diagnostics = lambda *args: {
        "final_position_error_m": 0.0, "final_rotation_error_rad": 0.0,
        "physical_tracking_verified": False}
    return session, submitted


@pytest.mark.parametrize("dt", [0, -0.004, 0.025, float("nan"), float("inf")])
def test_conversion_rejects_invalid_or_non_aligned_timing(dt):
    with pytest.raises(ValueError):
        policy_actions(np.zeros((11, 6)), np.zeros(14), "left", 0, planner_dt=dt)


def test_conversion_validates_t0_and_finite_six_joint_samples():
    for positions in (np.zeros((1, 6)), np.zeros((11, 5)),
                      np.ones((11, 6)), np.full((11, 6), np.nan)):
        with pytest.raises(ValueError):
            policy_actions(positions, np.zeros(14), "left", 0, planner_dt=0.004)


def _plan(session):
    actions = np.zeros((3, 14), dtype=np.float32)
    actions[:, 0] = [0.1, 0.2, 0.3]
    return session._store_motion_plan({
        "arm": "left", "start_state": session._motion_state(),
        "step_id": session.step_id, "target": {}, "actions": actions})


@pytest.mark.parametrize("limit,count,status", [(100, 3, "Success"), (8, 1, "Partial")])
def test_cached_plan_executes_native_joint_path_and_records_only_executed_prefix(tmp_path, limit, count, status):
    session, submitted = _session(tmp_path, limit)
    plan = _plan(session)
    result = session._execute_motion_plan(plan["motion_plan_id"])
    assert result["status"] == status
    assert result["planned_control_blocks"] == 3
    assert result["executed_control_blocks"] == len(submitted) == count
    assert len(result["frames"]) == len(result["steps"]) == count
    assert plan["motion_plan_id"] not in session._motion_plans
    for i, row in enumerate(result["steps"]):
        np.testing.assert_array_equal(row["executed_action"], plan["actions"][i])
        np.testing.assert_array_equal(submitted[i]["left_arm_joint_state"], row["executed_action"][:6])
        assert row["observation_step_id"] == 7 + i
        assert row["step_id"] == 8 + i
        assert row["execution_mode"] == "policy_joint_target"
        assert "inner_controls" not in row
        assert (tmp_path / f"action_{7 + i:06d}.json").is_file()
    assert result["joint_target_contract"]["frequency_hz"] == 25


@pytest.mark.parametrize("changed_step", [False, True])
def test_cached_plan_rejects_changed_state_or_any_intervening_action(tmp_path, changed_step):
    session, submitted = _session(tmp_path)
    plan = _plan(session)
    if changed_step:
        session.step_id += 1  # Even if measured joints returned to the same place.
    else:
        session.obs["states"][0] += 0.003
    result = session._execute_motion_plan(plan["motion_plan_id"])
    assert result["status"] == "Plan_Stale"
    assert not result["executed"] and not submitted


def test_execution_success_does_not_claim_goal_reached(tmp_path):
    session, _ = _session(tmp_path)
    session._motion_plan_diagnostics = lambda *args: {
        "final_position_error_m": 0.015, "final_rotation_error_rad": 0.08}
    result = session._execute_motion_plan(_plan(session)["motion_plan_id"])
    assert result["status"] == "Success"
    assert result["goal_reached"] is False
    assert result["measured_goal_error"]["physical_tracking_verified"]


@pytest.mark.parametrize("preview_only", [False, True])
def test_single_mode_flag_defaults_to_native_execution(tmp_path, preview_only):
    session, submitted = _session(tmp_path)
    planner = SimpleNamespace(plan_path=lambda *args, **kwargs: {
        "status": "Success", "position": np.zeros((12, 6)), "interpolation_dt": 0.004})
    session._ensure_curobo_planner = lambda arm: (SimpleNamespace(entity_origin_pose=None), planner)
    result = session.free_space_move(
        arm="left", target={"position": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
        include_trajectory=True, **({"preview_only": True} if preview_only else {}))
    assert result["status"] == "Success"
    assert result["preview_only"] is preview_only
    assert result["executed"] is not preview_only
    assert len(submitted) == (0 if preview_only else 2)
    assert len(result["trajectory_preview"]["actions"]) == 2
    assert result["timing"]["fits_remaining_budget"]
    assert result["timing"]["remaining_episode_steps"] == 93


def test_failed_plan_exposes_stage_reason_and_diagnostics_without_execution(tmp_path):
    session, submitted = _session(tmp_path)
    failure = {'status': 'Fail', 'failure_stage': 'trajectory_optimization',
               'reason': 'Path feasibility failed', 'diagnostics': {'min_position_error': 1e-7}}
    planner = SimpleNamespace(plan_path=lambda *args, **kwargs: failure)
    session._ensure_curobo_planner = lambda arm: (SimpleNamespace(entity_origin_pose=None), planner)
    result = session.free_space_move(arm='left',
        target={'position': [0, 0, 1], 'quaternion_wxyz': [1, 0, 0, 0]})
    assert result['status'] == 'Planning_Failed'
    for key in ('reason', 'failure_stage', 'diagnostics'):
        assert result[key] == failure[key]
    assert result['executed'] is False and not submitted and not session._motion_plans


def test_plan_and_preview_share_the_same_action_limit(tmp_path, monkeypatch):
    from services.robodojo.trajectory import MAX_TRAJECTORY_ACTIONS
    session, submitted = _session(tmp_path)
    planner = SimpleNamespace(plan_path=lambda *args, **kwargs: {
        "status": "Success", "position": np.zeros((11, 6)), "interpolation_dt": 0.004})
    session._ensure_curobo_planner = lambda arm: (SimpleNamespace(entity_origin_pose=None), planner)
    monkeypatch.setattr("services.robodojo.session.policy_actions",
                        lambda *args, **kwargs: np.zeros((MAX_TRAJECTORY_ACTIONS + 1, 14)))
    result = session.free_space_move(
        arm="left", target={"position": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]})
    assert result["status"] == "Planning_Failed"
    assert not submitted and not session._motion_plans


def test_seventy_step_plan_is_rejected_before_execution(tmp_path):
    session, submitted = _session(tmp_path)
    planner = SimpleNamespace(plan_path=lambda *args, **kwargs: {
        'status': 'Success', 'position': np.zeros((700, 6)), 'interpolation_dt': .004})
    session._ensure_curobo_planner = lambda arm: (SimpleNamespace(entity_origin_pose=None), planner)
    result = session.free_space_move(arm='left', preview_only=True, include_trajectory=True,
        target={'position': [0,0,0], 'quaternion_wxyz': [1,0,0,0]})
    assert result['status'] == 'Planning_Failed'
    assert result['action_count'] == 70 and result['max_actions'] == 50
    assert result['failure_stage'] == 'action_budget'
    assert not submitted and not session._motion_plans


@pytest.mark.parametrize("position_error,rotation_error", [(0.002, 0), (0, 0.006)])
def test_reject_inaccurate_plan_before_cache_or_execution(tmp_path, position_error, rotation_error):
    session, submitted = _session(tmp_path)
    planner = SimpleNamespace(plan_path=lambda *args, **kwargs: {
        "status": "Success", "position": np.zeros((11, 6)),
        "velocity": np.zeros((11, 6)), "interpolation_dt": 0.004})
    session._ensure_curobo_planner = lambda arm: (SimpleNamespace(entity_origin_pose=None), planner)
    session._motion_plan_diagnostics = lambda *args: {
        "final_position_error_m": position_error, "final_rotation_error_rad": rotation_error}
    result = session.free_space_move(arm="left",
                                    target={"position": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
                                    preview_only=False)
    assert result["status"] == "Planning_Failed"
    assert not result["executed"] and not submitted and not session._motion_plans


@pytest.mark.upstream
def test_motion_planner_config_has_explicit_accuracy_tolerances():
    import ast
    from pathlib import Path
    from services.robodojo.trajectory import GOAL_POSITION_TOLERANCE_M, GOAL_ROTATION_TOLERANCE_RAD
    path = Path(__file__).resolve().parents[2] / "RoboDojo/env/planner_manager/curobo_planner.py"
    if not path.exists():
        pytest.skip("Pinned RoboDojo source not available")
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_build_motion_planner_cfg")
    call = next(node for node in ast.walk(method)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "MotionPlannerCfg")
    tolerances = {keyword.arg: ast.literal_eval(keyword.value)
                  for keyword in call.keywords if keyword.arg in ("position_tolerance", "orientation_tolerance")}
    assert tolerances == {"position_tolerance": GOAL_POSITION_TOLERANCE_M,
                          "orientation_tolerance": GOAL_ROTATION_TOLERANCE_RAD}
    search = {keyword.arg: ast.literal_eval(keyword.value)
              for keyword in call.keywords if keyword.arg in ('num_ik_seeds', 'self_collision_check')}
    assert search == {'num_ik_seeds': 32, 'self_collision_check': True}
