"""Native EEF action smoke: robodojo_step_ee rows solved by the environment's own IK.

For each arm and offset from its start pose, two ways of reaching the goal:
- ``jump``: the goal pose repeated for 15 steps (the IK solves the whole move at once);
- ``line``: 25 straight-line waypoints (position interpolated, orientation held), then
  5 settle steps at the goal.
Records the measured link6 error to the goal after each, and that every row advanced
the environment by exactly one step.
"""
import json

import numpy as np

import robodojo_toolkit as tk

OFFSETS = [(0.0, 0.10, -0.05), (0.05, 0.15, 0.0), (0.0, 0.20, 0.05)]
POSITION_TOL_M, ROTATION_TOL_RAD = 0.001, 0.005


def observe(ctx):
    return tk.parse_reply(ctx.call('robodojo_observe'), images=False)[0]


def run(ctx, meta, rows):
    before = meta['step_id']
    reply, after = tk.execute(ctx, rows)
    return after, after['step_id'] - before


def main(ctx):
    meta = observe(ctx)
    home = {arm: {'position': meta['eef_positions'][i], 'quaternion_wxyz': meta['eef_quaternions_wxyz'][i]}
            for i, arm in enumerate(('left', 'right'))}
    report = {'moves': []}
    for arm in ('left', 'right'):
        sign = 1.0 if arm == 'left' else -1.0
        for offset in OFFSETS:
            start = home[arm]
            goal = {'position': [start['position'][0] + sign * offset[0], start['position'][1] + offset[1],
                                 start['position'][2] + offset[2]], 'quaternion_wxyz': start['quaternion_wxyz']}
            for mode in ('jump', 'line'):
                meta = observe(ctx)
                if mode == 'jump':
                    rows = tk.ee_rows(meta, [{arm: goal}] * 15)
                else:
                    now = np.asarray(meta['eef_positions'][('left', 'right').index(arm)])
                    path = [now + (np.asarray(goal['position']) - now) * (k + 1) / 25 for k in range(25)]
                    rows = tk.ee_rows(meta, [{arm: {'position': p.tolist(), 'quaternion_wxyz': goal['quaternion_wxyz']}}
                                             for p in path] + [{arm: goal}] * 5)
                meta, advanced = run(ctx, meta, rows)
                position_error, rotation_error = tk.pose_error(meta, arm, goal)
                row = {'arm': arm, 'offset': offset, 'mode': mode, 'rows': len(rows), 'steps_advanced': advanced,
                       'position_error_m': position_error, 'rotation_error_rad': rotation_error,
                       'passed': bool(position_error <= POSITION_TOL_M and rotation_error <= ROTATION_TOL_RAD
                                      and advanced == len(rows))}
                report['moves'].append(row)
                print('EE_SMOKE', json.dumps(row), flush=True)
                # Return home with native EEF rows, as the next move starts from there.
                meta, _ = run(ctx, meta, tk.ee_rows(meta, [{arm: start}] * 15))
    moves = report['moves']
    report['summary'] = {'moves': len(moves), 'passed': sum(m['passed'] for m in moves),
                         'max_position_error_m': max(m['position_error_m'] for m in moves),
                         'max_rotation_error_rad': max(m['rotation_error_rad'] for m in moves)}
    (ctx.output_dir / 'ee_smoke.json').write_text(json.dumps(report, indent=2))
    print('EE_SMOKE_SUMMARY', json.dumps(report['summary']), flush=True)
