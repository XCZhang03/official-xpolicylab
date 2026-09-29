"""Planning accuracy smoke: toolkit cuRobo plans executed by the official environment.

Same bar as local_tests/unit/test_gpu_joint_replay.py: the observed link6 pose after
executing a plan must be within 1 mm and 0.005 rad of the goal, and every planned row
must advance the environment by exactly one step.
"""
import json

import numpy as np
from scipy.spatial.transform import Rotation

import robodojo_toolkit as toolkit

# Offsets from each arm's episode-start pose (m); mirrored in y for the right arm.
OFFSETS = [(0.0, 0.10, -0.05), (0.05, 0.15, 0.0), (0.0, 0.20, 0.05), (-0.04, 0.12, 0.02)]
POSITION_TOL_M, ROTATION_TOL_RAD = 0.001, 0.005


def observe(ctx):
    return json.loads(next(b['text'] for b in ctx.call('robodojo_observe')['content'] if b['type'] == 'text'))


def error(meta, index, goal):
    position = np.asarray(meta['eef_positions'][index]) - np.asarray(goal['position'])
    measured = Rotation.from_quat(np.asarray(meta['eef_quaternions_wxyz'][index])[[1, 2, 3, 0]])
    target = Rotation.from_quat(np.asarray(goal['quaternion_wxyz'])[[1, 2, 3, 0]])
    return float(np.linalg.norm(position)), float((target * measured.inv()).magnitude())


def main(ctx):
    planner = toolkit.shared_planner()
    meta = observe(ctx)
    start = {arm: {'position': meta['eef_positions'][i], 'quaternion_wxyz': meta['eef_quaternions_wxyz'][i]}
             for i, arm in enumerate(('left', 'right'))}
    fk_check = planner.kinematics.check(meta['states'], meta['eef_positions'], meta['eef_quaternions_wxyz'])
    report = {'fk_check_at_start': fk_check, 'moves': []}
    for index, arm in enumerate(('left', 'right')):
        sign = 1.0 if arm == 'left' else -1.0
        for offset in OFFSETS:
            goal = {'position': [start[arm]['position'][0] + sign * offset[0], start[arm]['position'][1] + offset[1],
                                 start[arm]['position'][2] + offset[2]], 'quaternion_wxyz': start[arm]['quaternion_wxyz']}
            plan = planner.plan(arm, meta['states'], goal)
            row = {'arm': arm, 'offset': offset, 'plan_status': plan['status']}
            if plan['status'] == 'Success':
                before = meta['step_id']
                rows = np.asarray(plan['actions']).tolist()
                for chunk in range(0, len(rows), toolkit.MAX_TRAJECTORY_ACTIONS):
                    reply = ctx.call('robodojo_step', actions=rows[chunk:chunk + toolkit.MAX_TRAJECTORY_ACTIONS])
                meta = json.loads(next(b['text'] for b in reply['content'] if b['type'] == 'text'))
                position_error, rotation_error = error(meta, index, goal)
                row.update(rows=len(rows), steps_advanced=meta['step_id'] - before,
                           position_error_m=position_error, rotation_error_rad=rotation_error,
                           passed=bool(position_error <= POSITION_TOL_M and rotation_error <= ROTATION_TOL_RAD
                                       and meta['step_id'] - before == len(rows)))
            else:
                row['reason'] = plan.get('reason')
            report['moves'].append(row)
            print('SMOKE', json.dumps(row), flush=True)
    executed = [m for m in report['moves'] if 'passed' in m]
    report['summary'] = {'executed': len(executed), 'passed': sum(m['passed'] for m in executed),
                         'max_position_error_m': max((m['position_error_m'] for m in executed), default=None),
                         'max_rotation_error_rad': max((m['rotation_error_rad'] for m in executed), default=None)}
    (ctx.output_dir / 'planning_smoke.json').write_text(json.dumps(report, indent=2))
    print('SMOKE_SUMMARY', json.dumps(report['summary']), 'FK', json.dumps(fk_check), flush=True)
