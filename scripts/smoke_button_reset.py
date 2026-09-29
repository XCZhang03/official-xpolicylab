"""Operator-only PhysX regression; not an agent tool or task-success evaluation.

Run with scripts/robodojo_env.sh sourced and the project on PYTHONPATH:
python scripts/smoke_button_reset.py --task swap_blocks --output <SSD-dir>
Uses native task initialization/reset and drive actuation to depress the button,
then checks spring rebound and the native press-transition predicate. Robot
contact planning and the full block-swap task are deliberately not tested here.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.robodojo import rpc, server


def probe(session, port, **kwargs):
    from isaacsim.core.utils.types import ArticulationAction
    env = session.env
    rows = []
    # Production also creates a fresh simulator for every episode. Do not exercise
    # upstream in-process multi-layout reloads here; they are a different path.
    for seed in (PROBE_SEED,):
        env.reset(seed=[seed])
        parser = env.reward_manager.func_parser
        layout = parser.layout_manager
        name = layout.get_instance_name(label='button0', env_idx=0)
        button = layout.get_scene_object(inst_name=name, env_idx=0)
        assert len(button.dof_names) == 1, button.dof_names
        def ratio():
            info = button.get_joint_info(button.dof_names[0])
            return float((info['position']-info['lower'])/(info['upper']-info['lower']))
        def settle():
            for _ in range(250):
                env.sim_step(render=False)
        settle()
        row = {'seed':seed, 'reset_ratio':ratio(), 'cycles':[]}
        rows.append(row)
        (session.output/'button-reset-probe.json').write_text(json.dumps(rows,indent=2))
        assert row['reset_ratio'] > .95, row
        for _ in range(3):
            args = dict(env_idx=0,label='button0',tag='press',above_threshold=.95,below_threshold=.5)
            parser.is_joint_position_ratio_change_from_above_to_below(args)
            button.apply_action(ArticulationAction(joint_positions=button.lower_joint_positions.copy()))
            settle()
            pressed = ratio()
            counted = parser.is_joint_position_ratio_change_from_above_to_below(args)
            button.reset_drive_targets()
            settle()
            released = ratio()
            row['cycles'].append(dict(pressed_ratio=pressed,released_ratio=released,transition=float(counted)))
            (session.output/'button-reset-probe.json').write_text(json.dumps(rows,indent=2))
            assert pressed < .5 and released > .95 and counted == 1, row
    print('BUTTON_RESET_PASS '+json.dumps(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--layout-seed', type=int, default=7)
    probe_args, remaining = parser.parse_known_args()
    PROBE_SEED = probe_args.layout_seed
    sys.argv[1:] = remaining
    rpc.serve = probe
    server.main()
