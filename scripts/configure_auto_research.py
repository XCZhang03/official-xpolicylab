#!/usr/bin/env python3
"""Trusted operator: create a fresh auto-research configuration; no model/env launch."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from services.controller.config import Configuration, Limits, FORMAL_EPISODES, OFFICIAL_COLLECTIONS, EXPLORATION_COLLECTIONS, FORMAL_COLLECTION, exploration_layouts
from services.robodojo.timeouts import task_timeouts
from services.controller.batch import collection_layouts, validate_layouts
from services.controller.demonstrations import cache_demonstration
from services.robodojo.demonstrations import DEMONSTRATION_CONTEXTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--agent-cli', choices=('codex', 'claude'), default='codex', help='Agent CLI run in the container')
    parser.add_argument('--agent-provider', choices=('openrouter', 'claude-login', 'anthropic'), default='openrouter',
                        help='Claude only: model provider held by the host relay')
    parser.add_argument('--sim-gpu', required=True, help='GPU ordinal or UUID for the simulator')
    parser.add_argument('--research-gpu', required=True, help='Separate GPU for the agent container (cuRobo planning, rehearsal, formal)')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--exploration-collections', default=','.join(map(str, EXPLORATION_COLLECTIONS)),
                        help='Eval_Layout collections explored in order, each exhausted before the next')
    parser.add_argument('--exploration-envs', type=int, default=1, help='Concurrent exploration slots sharing the episode budget (1..8)')
    parser.add_argument('--demonstration-context', choices=DEMONSTRATION_CONTEXTS, default='terminal_state')
    parser.add_argument('--formal-seed', type=int, default=0)
    parser.add_argument('--formal-episodes', type=int, default=FORMAL_EPISODES['auto-research'])
    parser.add_argument('--formal-workers', type=int, default=4, help='Concurrent formal episodes')
    parser.add_argument('--formal-eval-seed', type=int, default=FORMAL_COLLECTION,
                        help='Saved-layout collection of the harness formal batch (default: held-out collection)')
    parser.add_argument('--exploration-start-seed', type=int, default=0, help='First layout index in each collection')
    parser.add_argument('--image', default='robodojo-official:dev', help='Installed image only; never pulls')
    parser.add_argument('--workspace-gib', type=int, default=20, help='Hard cap shared by workspace and Codex state')
    args = parser.parse_args()
    args.formal_seconds = task_timeouts(args.task)['episode_seconds']
    image_name = args.image
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    mapping = {}
    for line in devices.splitlines():
        index, identifier = map(str.strip, line.split(','))
        mapping[index] = mapping[identifier] = identifier
    spec = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image_name], text=True))[0]
    image = spec['Id']
    parent = (PROJECT/'runtime').resolve()/'auto-research'
    root = parent/(time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+'_'+uuid.uuid4().hex[:10])
    # Recordings keep every 25 Hz frame (about 1.1 MB per step for three RGB cameras),
    # and the official adapter runs each rehearsal/formal episode to its native end.
    data_bytes = max(2*1024**3, task_timeouts(args.task)['native_step_limit']*3*512*1024 + 512*1024**2)
    def limits(seconds, memory=8192):
        return Limits(seconds, memory, 4, 256, 1024, data_bytes)
    collections = [int(c) for c in args.exploration_collections.split(',')]
    if not collections or len(set(collections)) != len(collections):
        parser.error('--exploration-collections must list distinct collections')
    counts = {c: collection_layouts(args.task, c) for c in collections}
    # Every layout of the first collection, then the next: collections differ in size
    # (45 per collection for *_random tasks), so any budget up to their total is valid.
    layouts = exploration_layouts(collections, counts, args.exploration_start_seed, args.episodes)
    if len(layouts) < args.episodes:
        parser.error(f'Only {len(layouts)} saved layouts are available for {args.task}')
    config = Configuration(root, image, args.task, tuple(layouts),
        args.formal_seed, 0, mapping[args.sim_gpu],
        mapping[args.research_gpu], mapping[args.research_gpu],
        limits(args.formal_seconds), limits(args.formal_seconds), limits(args.formal_seconds, 16384),
        workspace_mb=args.workspace_gib*1024,
        demonstration_context=args.demonstration_context,
        formal_episodes=args.formal_episodes, formal_eval_seed=args.formal_eval_seed,
        formal_workers=args.formal_workers, observation_profile='official',
        exploration_envs=args.exploration_envs, agent_cli=args.agent_cli, agent_provider=args.agent_provider)
    if config.formal_collection in collections:
        parser.error(f'Formal collection {config.formal_collection} is also explored; choose a held-out collection')
    try:
        validate_layouts(config)
    except ValueError as error:
        parser.error(f'{error}. Build the held-out collection under Eval_Layout/RoboDojo/arx_x5/{config.formal_collection}/ first.')
    # Episodes run on the pristine official RoboDojo; stage it now, before any launch.
    from services.robodojo.source import official_stage
    print(f'Official RoboDojo stage: {official_stage(PROJECT)}', file=sys.stderr)
    if config.formal_collection in OFFICIAL_COLLECTIONS:
        print(f'Note: formal collection {config.formal_collection} is an official test collection, '
              'not an operator-built held-out one.', file=sys.stderr)
    # Fetch only official visual media outside the offline container, before any
    # model/episode starts. Deployment converts the selected format to local images.
    cache_demonstration(PROJECT/'runtime/reference-demos/website', args.task, args.demonstration_context)
    root.mkdir(parents=True, mode=0o700)
    path = root/'operator-input.json'
    with path.open('x') as stream:
        json.dump(asdict(config), stream, default=str, indent=2)
    path.chmod(0o600)
    print(path)
    print(f'bash scripts/start_auto_research_agent.sh --config {path}', file=sys.stderr)


if __name__ == '__main__':
    main()
