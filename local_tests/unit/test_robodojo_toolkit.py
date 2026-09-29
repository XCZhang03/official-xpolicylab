"""Motion toolkit parity with the harness MCP implementations (no GPU)."""
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'packages/robodojo_toolkit/src'))

from robodojo_toolkit import DualArm, pose_math, policy_actions  # noqa: E402
from robodojo_toolkit.pose import ROTATION_REPRESENTATIONS  # noqa: E402


def _flat(value, out):
    if isinstance(value, dict):
        for key in sorted(value):
            if key != 'math_backend':
                _flat(value[key], out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _flat(item, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out.append(float(value))
    return out


def test_pose_math_matches_harness_tool():
    pytest.importorskip('torch')
    from services.robodojo.pose_math import pose_math as harness
    rng = np.random.default_rng(3)

    def rotation():
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        return {'representation': 'quaternion_wxyz', 'value': q.tolist()}
    for _ in range(200):
        for output in ROTATION_REPRESENTATIONS:
            base = {'position_m': rng.normal(size=3).tolist(), 'rotation': rotation()}
            delta = {'position_m': rng.normal(size=3).tolist(), 'rotation': rotation()}
            for frame in ('local', 'environment'):
                case = {'operation': 'compose_pose', 'base_pose': base, 'delta_pose': delta,
                        'delta_frame': frame, 'output_representation': output}
                ours, theirs = _flat(pose_math(case), []), _flat(harness(PROJECT, case), [])
                assert np.allclose(ours, theirs, atol=1e-9)


def test_dual_arm_matches_harness_fk_and_bounded_step():
    from services.robodojo.kinematics import ArmFK
    arm = DualArm()
    urdf = arm.spec['urdf']
    harness_fk = ArmFK(urdf, arm.spec['arm_joints'], 'base_link', 'link6')
    rng = np.random.default_rng(5)
    for _ in range(50):
        q = rng.uniform(-1.5, 1.5, 6)
        assert np.allclose(arm.fk['left'].matrix(q), harness_fk.matrix(q))
        target = arm.eef('left', q + rng.normal(scale=0.05, size=6))
        ours, _ = arm.step_toward('left', q, target)
        theirs, _ = harness_fk.bounded_target(q, arm.limits, arm.roots['left'], target)
        assert np.allclose(ours, theirs)
    # Start pose matches the simulator's reported link6 at episode start (0.5 mm).
    assert np.allclose(arm.eef('left', np.zeros(6))['position'], [-0.29953, -0.35230, 0.92150], atol=5e-4)


def test_policy_actions_is_the_harness_function():
    from services.robodojo import trajectory
    positions = np.linspace(np.zeros(6), np.ones(6) * 0.2, 101).astype(np.float32)
    state = np.zeros(14, np.float32); state[6] = state[13] = 1.0
    a = policy_actions(positions, state, 'left', 0.2, planner_dt=0.004)
    b = trajectory.policy_actions(positions, state, 'left', 0.2, planner_dt=0.004)
    assert a.shape == (10, 14) and np.array_equal(a, b)


def test_fk_preview_matches_harness_preview():
    from types import SimpleNamespace
    from services.robodojo.kinematics import DualKinematics, transform
    arm = DualArm()
    harness = DualKinematics.__new__(DualKinematics)  # Harness preview without a simulator.
    harness.fk = arm.fk
    harness.root = lambda name: arm.roots[name]
    harness.check = lambda: {'passed': True}
    rows = np.random.default_rng(9).uniform(-1, 1, (50, 14))
    rows[:, [6, 13]] = np.random.default_rng(10).uniform(0, 1, (50, 2))
    ours, theirs = arm.preview(rows)['trajectory'], harness.preview(rows)['trajectory']
    for a, b in zip(ours, theirs, strict=True):
        for name in ('left', 'right'):
            assert np.allclose(a[name]['position'], b[name]['position'])
            assert np.allclose(np.abs(np.dot(a[name]['quaternion_wxyz'], b[name]['quaternion_wxyz'])), 1.0)
            assert a[name]['gripper_closed'] == b[name]['gripper_closed']


class _KinematicRobot:
    """Fake ctx whose joints track commands exactly (no physics); counts calls."""

    def __init__(self, state):
        import robodojo_toolkit as tk
        self.kinematics, self.state, self.calls = tk.DualArm(), np.asarray(state, float), []

    def _reply(self):
        import json
        poses = [self.kinematics.eef(arm, self.state[o:o + 6]) for arm, o in (('left', 0), ('right', 7))]
        meta = {'step_id': len(self.calls), 'states': self.state.tolist(),
                'eef_positions': [p['position'] for p in poses], 'eef_quaternions_wxyz': [p['quaternion_wxyz'] for p in poses]}
        return {'content': [{'type': 'text', 'text': json.dumps(meta)}]}

    def call(self, tool, **arguments):
        self.calls.append((tool, len(arguments.get('actions', []))))
        if tool == 'robodojo_step':
            self.state = np.asarray(arguments['actions'][-1], float)
        return self._reply()


def _start_and_target():
    import robodojo_toolkit as tk
    state = np.zeros(14)
    state[[1, 2, 8, 9]] = [0.8, 0.9, 0.8, 0.9]
    state[6] = 1.0
    start = tk.DualArm().eef('left', state[:6])
    target = {'position': (np.asarray(start['position']) + [0.05, 0.03, -0.06]).tolist(),
              'quaternion_wxyz': start['quaternion_wxyz']}
    return state, target


def test_bounded_eef_motion_ik_and_servo():
    import robodojo_toolkit as tk
    state, target = _start_and_target()
    joints, info = tk.ik('left', state[:6], target)
    assert info['converged'] and info['position_error_m'] < 1e-4
    rows, _ = tk.eef_rows(state, [{'left': {**target, 'gripper_closed': True}}] * 5)
    assert rows.shape == (5, 14) and np.all(rows[:, 6] == 0) and np.allclose(rows[:, 7:], state[7:])
    assert np.abs(np.diff(np.r_[state[None], rows][:, :6], axis=0)).max() <= 0.05 + 1e-6
    robot = _KinematicRobot(state)
    _, meta, result = tk.servo(robot, 'left', target, max_steps=80)
    assert result['reached'] and result['steps'] == len(robot.calls) - 1  # One observe first.
    robot = _KinematicRobot(state)
    _, meta, diagnostics = tk.step_eef(robot, [{'left': target}] * 3)
    assert [c for c in robot.calls] == [('robodojo_observe', 0)] + [('robodojo_step', 1)] * 3


def test_native_ee_rows_and_execute_routing():
    import robodojo_toolkit as tk
    meta = {'states': [0.0] * 6 + [1.0] + [0.0] * 6 + [0.2],
            'eef_positions': [[0.0, 0.0, 1.0], [0.3, 0.0, 1.0]],
            'eef_quaternions_wxyz': [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]}
    row = tk.ee_row(meta, {'left': {'position': [0.1, 0.0, 1.0], 'quaternion_wxyz': [-1.0, 0.0, 0.0, 0.0],
                                    'gripper_closed': True}})
    # Left target (w >= 0 canonical, closed gripper); right holds its measured pose and command.
    assert row.tolist() == pytest.approx([0.1, 0, 1, 1, 0, 0, 0, 0, 0.3, 0, 1, 0, 1, 0, 0, 0.2])
    with pytest.raises(ValueError):
        tk.ee_row(meta, {'left': {'position': [0, 0, 1], 'quaternion_wxyz': [2, 0, 0, 0]}})

    class Recorder:
        def __init__(self):
            self.calls = []

        def call(self, tool, **arguments):
            import json
            self.calls.append((tool, len(arguments['actions'])))
            return {'content': [{'type': 'text', 'text': json.dumps(meta)}]}
    recorder = Recorder()
    tk.execute(recorder, tk.ee_rows(meta, [{}] * 60))
    tk.execute(recorder, np.zeros((3, 14)), chunk=2)
    assert recorder.calls == [('robodojo_step_ee', 50), ('robodojo_step_ee', 10), ('robodojo_step', 2), ('robodojo_step', 1)]
    with pytest.raises(ValueError):
        tk.execute(recorder, np.zeros((2, 15)))
