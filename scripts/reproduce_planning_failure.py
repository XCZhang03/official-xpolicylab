"""Read saved MCP poses/joints and reproduce cuRobo planning, without physics.

Operator diagnostic only; no simulator state restoration or evaluation retries.
Experimental tolerances apply to this process, never to production settings.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'RoboDojo')]


def summarize(result):
    if result is None:
        return None
    out = {}
    for key in ('success', 'feasible', 'position_error', 'rotation_error',
                'position_tolerance', 'orientation_tolerance'):
        value = getattr(result, key, None)
        if hasattr(value, 'detach'):
            value = value.detach().cpu().reshape(-1).tolist()
        out[key] = value
    for label in ('metrics', 'interpolated_metrics'):
        metrics = getattr(result, label, None)
        if metrics is not None:
            constraints = metrics.costs_and_constraints.constraints
            out[label] = {name: {'min': float(value.min()), 'max': float(value.max())}
                          for name, value in zip(constraints.names, constraints.values)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--position', type=float, default=.001)
    parser.add_argument('--rotation', type=float, default=.005)
    parser.add_argument('--ik-seeds', type=int, default=32)
    parser.add_argument('--traj-seeds', type=int, default=4)
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--waypoint-check', action='store_true',
        help='Try bounded two-leg alternatives from predicted endpoints, without executing physics')
    args = parser.parse_args()
    from env.planner_manager.curobo_planner import CuroboPlanner
    from curobo.batch_motion_planner import MotionPlannerCfg
    from services.robodojo.kinematics import ArmFK, transform
    from scipy.spatial.transform import Rotation
    import numpy as np
    import yaml

    class ExperimentalPlanner(CuroboPlanner):
        def _build_motion_planner_cfg(self, max_batch_size):
            self.use_graph_planner = args.graph
            cfg = MotionPlannerCfg.create(robot=self.robot_cfg, scene_model=self.scene_model,
                device_cfg=self.device_cfg, position_tolerance=args.position,
                orientation_tolerance=args.rotation, num_ik_seeds=args.ik_seeds,
                num_trajopt_seeds=args.traj_seeds, self_collision_check=True,
                use_cuda_graph=self.use_cuda_graph, max_batch_size=max_batch_size,
                multi_env=False, max_goalset=1)
            cfg.trajopt_solver_config.interpolation_dt = float(self.dt)
            if not args.graph:
                cfg.graph_planner_config = None
            return cfg

    names = ['joint'+str(i) for i in range(1,7)]
    planner = ExperimentalPlanner([0,0,0,1,0,0,0], names, names, dt=.004,
        yml_path=str(ROOT/'RoboDojo/third_party/curobo/robot/x5_v2/curobo.yml'),
        table_height=.74-.765)
    fk = ArmFK(ROOT/'RoboDojo/Assets/Robots/x5/X5A.urdf', names)
    robots = yaml.safe_load((ROOT/'RoboDojo/env_cfg/robot/dual_x5.yml').read_text())['robots']
    captured = []
    def collisions(q):
        kin = planner.motion_planner.ik_solver.kinematics
        spheres = kin._forward(q.reshape(-1,1,q.shape[-1])).get_link_spheres().detach().cpu().numpy()[0,0]
        cfg = kin.config.kinematics_config
        names_by_idx = {v:k for k,v in cfg.link_name_to_idx_map.items()}
        sphere_names = [names_by_idx[int(i)] for i in cfg.link_sphere_idx_map.reshape(-1)]
        sc = kin.get_self_collision_config()
        pairs = sc.collision_pairs.cpu().numpy()
        padding = sc.sphere_padding.cpu().numpy()
        pairs_out = []
        for i,j in pairs:
            if spheres[i,3] <= 0 or spheres[j,3] <= 0:
                continue
            overlap = float(spheres[i,3]+spheres[j,3]+padding[i]+padding[j]
                            -np.linalg.norm(spheres[i,:3]-spheres[j,:3]))
            if overlap > 0:
                pairs_out.append({'links': [sphere_names[i],sphere_names[j]], 'overlap_m': overlap})
        return sorted(pairs_out,key=lambda p:-p['overlap_m'])[:8]
    for label, solver in [('ik', planner.motion_planner.ik_solver),
                          ('trajopt', planner.motion_planner.trajopt_solver)]:
        rollout = solver.metrics_rollout
        compute = rollout.compute_metrics_from_action
        def metric_wrapper(*a, _compute=compute, _label=label, **kw):
            metrics = _compute(*a, **kw)
            constraints = metrics.costs_and_constraints.constraints
            captured.append({'stage': _label+'_constraints', 'constraints': {
                name: value.detach().cpu().reshape(value.shape[0], -1).max(dim=1).values.tolist()
                for name, value in zip(constraints.names, constraints.values)}})
            return metrics
        rollout.compute_metrics_from_action = metric_wrapper
        original = solver.solve_pose
        def wrapped(*a, _original=original, _label=label, **kw):
            result = _original(*a, **kw)
            details = {'stage': _label, 'result': summarize(result)}
            if _label == 'ik' and result.solution is not None:
                details['best_solution_collisions'] = collisions(result.solution.reshape(-1, result.solution.shape[-1])[:1])
            captured.append(details)
            return result
        solver.solve_pose = wrapped
    state = json.loads((args.session/'state.json').read_text())
    results = []
    for episode in state['results']:
        if episode.get('mode') != 'formal':
            continue
        if episode.get('formal_outcome') != 'error' and episode.get('returncode') in (None, 0):
            continue
        trace = args.session/'published'/Path(episode['artifacts']['mcp_trace']).relative_to('runtime/autonomous_controller')
        requests = {}
        for line in trace.read_text().splitlines():
            event = json.loads(line)
            if event['event'] == 'mcp_request':
                requests[event['call_id']] = event['arguments']
            returned = event.get('returned', {})
            if returned.get('motion_plan', {}).get('status') != 'Planning_Failed':
                continue
            request = requests[event['call_id']]
            arm = 0 if request['arm'] == 'left' else 1
            robot = robots[arm]
            root_pose = robot['default_root_pos'] + robot['default_root_rot']
            target = request['target']
            joints = returned['states'][7*arm:7*arm+6]
            captured.clear()
            start_collisions = collisions(planner._build_joint_state(joints).position)
            result = planner.plan_path(joints, target['position']+target['quaternion_wxyz'], root_pose)
            row = {'test': episode['formal_episode_index'], 'request': request,
                   'start_joints': joints, 'start_collisions': start_collisions,
                   'status': result['status'], 'attempts': list(captured)}
            if result['status'] == 'Success':
                matrix = transform(root_pose[:3], root_pose[3:]) @ fk.matrix(result['position'][-1])
                goal = transform(target['position'], target['quaternion_wxyz'])
                row['endpoint_error'] = {'position_m': float(np.linalg.norm(matrix[:3,3]-goal[:3,3])),
                    'rotation_rad': float(Rotation.from_matrix(goal[:3,:3] @ matrix[:3,:3].T).magnitude())}
            if args.waypoint_check:
                from services.robodojo.trajectory import MAX_TRAJECTORY_ACTIONS, policy_actions
                start_matrix = transform(root_pose[:3], root_pose[3:]) @ fk.matrix(joints)
                start_position = start_matrix[:3, 3]
                start_quaternion = Rotation.from_matrix(start_matrix[:3, :3]).as_quat()[[3,0,1,2]]
                goal_position = np.asarray(target['position'])
                midpoint = (start_position + goal_position) / 2
                candidates = [
                    ('lift_start', start_position + [0,0,.08], start_quaternion),
                    ('midpoint_lift_start_orientation', midpoint + [0,0,.08], start_quaternion),
                    ('midpoint_lift_goal_orientation', midpoint + [0,0,.08], target['quaternion_wxyz']),
                    ('above_goal', goal_position + [0,0,.08], target['quaternion_wxyz']),
                    ('midpoint_left', midpoint + [.08,0,0], start_quaternion),
                    ('midpoint_right', midpoint - [.08,0,0], start_quaternion),
                ]
                alternatives = []
                for name, position, quaternion in candidates:
                    waypoint = list(position) + list(quaternion)
                    current = list(joints)
                    trial = {'name': name, 'waypoint': waypoint, 'legs': []}
                    for pose in (waypoint, target['position'] + target['quaternion_wxyz']):
                        leg = planner.plan_path(current, pose, root_pose)
                        summary = {k: leg[k] for k in ('status', 'failure_stage', 'reason') if k in leg}
                        trial['legs'].append(summary)
                        if leg['status'] != 'Success':
                            break
                        initial = np.asarray(returned['states'], dtype=float).copy()
                        initial[7*arm:7*arm+6] = current
                        actions = policy_actions(leg['position'], initial, request['arm'],
                            initial[7*arm+6], planner_dt=leg['interpolation_dt'])
                        matrix = transform(root_pose[:3], root_pose[3:]) @ fk.matrix(leg['position'][-1])
                        goal = transform(pose[:3], pose[3:])
                        summary.update(action_count=len(actions), within_action_cap=len(actions) <= MAX_TRAJECTORY_ACTIONS,
                            position_error_m=float(np.linalg.norm(matrix[:3,3]-goal[:3,3])),
                            rotation_error_rad=float(Rotation.from_matrix(goal[:3,:3] @ matrix[:3,:3].T).magnitude()))
                        current = leg['position'][-1].tolist()
                    trial['both_legs_planned'] = len(trial['legs']) == 2 and all(x['status'] == 'Success' for x in trial['legs'])
                    alternatives.append(trial)
                    print(json.dumps({'test': row['test'], 'waypoint_trial': trial}), flush=True)
                    captured.clear()
                row['waypoint_trials'] = alternatives
                row['waypoint_scope'] = 'Planning only; predicted intermediate joints, no physics or real MCP execution'
            results.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from services.controller.storage import persist
    persist(args.output, {'settings': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                          'results': results})


if __name__ == '__main__':
    main()
