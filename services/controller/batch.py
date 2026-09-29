"""Operator-only post-hoc formal batch of an already rehearsed, frozen bundle.

Creates a new archive; never reopens or overwrites the source formal attempt.
No Codex process, code edits, API credentials, or episode retries are involved.
"""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import signal
import time

from .backend import NativeBackend
from .config import Configuration, FORMAL_COLLECTION, layout_of
from .storage import persist, verify
from .supervisor import Supervisor, qualifying_rehearsal


LAYOUTS = Path(__file__).resolve().parents[2]/'RoboDojo/Assets/Eval_Layout/RoboDojo/arx_x5'


def collection_layouts(task, collection):
    """Number of saved layouts, as the native SeedManager enumerates them (task_<n>.json)."""
    return sum(1 for p in (LAYOUTS/str(collection)).glob(task+'_*.json')
               if p.stem[len(task)+1:].isdigit())


def validate_layouts(config):
    count = collection_layouts(config.task, config.formal_collection)
    if config.formal_seed+config.formal_episodes > count:
        raise ValueError(f'Formal collection {config.formal_collection} has {count} layouts for {config.task}; requested range is unavailable')
    for key in config.exploration_seeds:
        collection, index = layout_of(key)
        if index >= collection_layouts(config.task, collection):
            raise ValueError(f'Exploration layout {index} is missing from collection {collection}')


def import_rehearsed_bundle(supervisor, source, bundle_id):
    """Import evidence only after validating the source manifest and runtime."""
    state = json.loads((source/'state.json').read_text())
    manifest = state['bundles'][bundle_id]
    source_bundle = (source/'bundles'/manifest['directory']).resolve()
    if not source_bundle.is_relative_to((source/'bundles').resolve()):
        raise ValueError('Invalid source bundle directory')
    verify(source_bundle, manifest)
    # Same evidence rule as Supervisor._has_successful_rehearsal, so a batch
    # session's own imported evidence can seed its parallel formal workers.
    evidence = next((r for r in state['results'] + state.get('imported_rehearsals', [])
                     if qualifying_rehearsal(r, bundle_id, manifest)), None)
    if evidence is None:
        raise ValueError('Source bundle has no qualifying successful rehearsal')
    original = Configuration.from_dict(state['configuration'])
    if original.task != supervisor.config.task or original.image != supervisor.config.image:
        raise ValueError('Source task/image must match the batch')
    imported, copied = supervisor.register(source_bundle, workspace_path=manifest['workspace_path'])
    if copied['sha256'] != manifest['sha256']:
        raise ValueError('Imported bundle fingerprint differs from rehearsal')
    supervisor.state['imported_rehearsals'] = [{
        'mode': 'rehearsal', 'bundle': imported, 'sha256': copied['sha256'],
        'task_complete': True, 'returncode': evidence['returncode'], 'reason': evidence['reason'],
        'source_session': str(source), 'source_bundle': bundle_id,
        'source_episode_id': evidence.get('episode_id')}]
    supervisor._save()
    return imported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-session', type=Path, required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New SSD-backed auto-research session directory')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--eval-seed', type=int, default=FORMAL_COLLECTION)
    parser.add_argument('--first-layout', type=int, default=0)
    parser.add_argument('--formal-wall-seconds', type=int,
                        help='Override the per-episode wall limit in this new evaluation only')
    args = parser.parse_args()
    source, output = args.source_session.resolve(), args.output.resolve()
    original = Configuration.from_dict(json.loads((source/'configuration.json').read_text()))
    config = replace(original, root=output, formal_episodes=args.episodes,
                     formal_eval_seed=args.eval_seed, formal_seed=args.first_layout,
                     demonstration_context='none', formal_workers=1)
    if args.formal_wall_seconds is not None:
        config = replace(config, formal=replace(config.formal, wall_seconds=args.formal_wall_seconds))
    validate_layouts(config)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    persist(output/'operator-input.json', json.loads(json.dumps(asdict(config), default=str)))
    lifecycle = {'kind': 'formal_batch', 'pid': os.getpid(), 'status': 'running',
                 'started_at': time.time(), 'source_session': str(source), 'source_bundle': args.bundle}
    persist(output/'agent-lifecycle.json', lifecycle)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    supervisor = None
    try:
        supervisor = Supervisor(config, lambda: NativeBackend(config))
        bundle = import_rehearsed_bundle(supervisor, source, args.bundle)
        print(json.dumps({'status': 'starting', 'output': str(output), 'bundle': bundle,
                          'episodes': config.formal_episodes, 'eval_seed': config.formal_collection,
                          'first_layout': config.formal_seed}), flush=True)
        result = supervisor.run(bundle, formal=True)
        persist(output/'formal-report.json', result)
        lifecycle.update(status='exited', returncode=0)
        print(json.dumps(result), flush=True)
    except KeyboardInterrupt:
        lifecycle['status'] = 'stopped'
        raise SystemExit(130)
    except Exception as exc:
        lifecycle.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        try:
            if supervisor:
                supervisor.close()
        finally:
            lifecycle['finished_at'] = time.time()
            persist(output/'agent-lifecycle.json', lifecycle)


if __name__ == '__main__':
    main()
