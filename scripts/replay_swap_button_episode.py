"""Operator-only unchanged-action replay of a saved Swap Blocks episode.

Source scripts/robodojo_env.sh, then run with --source <original sim directory>
and --output <new SSD directory>. No robot actions, layouts, reward predicates,
or button targets are edited. The installed simulator patch remains in effect.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.robodojo import rpc, server


def encode(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(type(value).__name__)


def replay(session, port, **kwargs):
    source = json.loads((args.source/'reset.json').read_text())
    assert source['metadata']['task'] == 'swap_blocks'
    files = sorted((args.source/source['episode_id']).glob('action_*.json'))
    assert files, 'No recorded actions'
    records = [json.loads(path.read_text()) for path in files]
    assert [r['step_id'] for r in records] == list(range(1, len(records)+1)), 'Incomplete action sequence'
    digest = hashlib.sha256(b''.join(path.read_bytes() for path in files)).hexdigest()
    session.reset(seed=source['metadata']['layout_id'], source='operator_fixed_action_replay',
                  policy_version='original_recorded_25hz_targets')
    manager = session.env.reward_manager
    parser = manager.func_parser
    layout = parser.layout_manager
    name = layout.get_instance_name(label='button0', env_idx=0)
    button = layout.get_scene_object(inst_name=name, env_idx=0)
    assert len(button.dof_names) == 1
    transitions = []
    original = parser.is_joint_position_ratio_change_from_above_to_below
    def watched(parameters):
        result = original(parameters)
        if result:
            transitions.append(int(session.env.take_action_cnt[0]))
        return result
    parser.is_joint_position_ratio_change_from_above_to_below = watched
    reset = json.loads((session.output/'reset.json').read_text())
    report = dict(source=str(args.source.resolve()), source_episode=source['episode_id'],
                  layout_seed=source['metadata']['layout_id'], eval_seed=source['metadata']['eval_seed'],
                  available_actions=len(records), source_action_files_sha256=digest,
                  initial_robot_state_matches=reset['initial_state_hash']==source['initial_state_hash'],
                  drive_reset_patch_sha256=hashlib.sha256((ROOT/'patches/robodojo-pr48-drive-reset.patch').read_bytes()).hexdigest())
    def snapshot():
        joint = button.get_joint_info(button.dof_names[0])
        return dict(step=session.step_id, button_ratio=(joint['position']-joint['lower'])/(joint['upper']-joint['lower']),
                    pending_checks=len(manager.check_list[0]), queries=manager.query_list,
                    counted_press_steps=list(transitions), success=session.success)
    with (args.output/'button-trace.jsonl').open('w', buffering=1) as trace:
        initial = snapshot()
        report['initial_button_ratio'] = initial['button_ratio']
        trace.write(json.dumps(initial,default=encode)+'\n')
        for record in records:
            session.chunk_step([record['executed_action']])
            trace.write(json.dumps(snapshot(),default=encode)+'\n')
            if session.step_id % 50 == 0:
                print('\nREPLAY_PROGRESS '+json.dumps(snapshot(),default=encode),flush=True)
            if session.terminated or session.truncated:
                break
    report.update(executed_actions=session.step_id, final=snapshot(),
                  task_complete=bool(session.success), terminated=bool(session.terminated),
                  truncated=bool(session.truncated))
    (args.output/'replay-report.json').write_text(json.dumps(report,indent=2,default=encode))
    print('\nREPLAY_COMPLETE '+json.dumps(report,default=encode),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    meta = json.loads((args.source/'reset.json').read_text())['metadata']
    sys.argv = ['swap-replay','--task','swap_blocks','--output',str(args.output/'sim'),
                '--eval-seed',str(meta['eval_seed']),'--camera-depth','--camera-calibration',
                '--no-enforce-step-limit']
    rpc.serve = replay
    server.main()
