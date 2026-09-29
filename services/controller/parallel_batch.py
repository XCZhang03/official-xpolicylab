"""Operator-only parallel batch or migration; never retries a started layout.

Workers use the existing single-episode evaluator in independent processes. This
post-hoc evaluator has no API credentials; it is not the agent submission route.
Source results remain untouched. A durable, exclusive claim prevents re-migration.
"""
import argparse
from collections import Counter
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import warnings

from .config import Configuration, FORMAL_COLLECTION
from .storage import persist, verify
from .batch import validate_layouts
from .supervisor import formal_outcome as outcome, qualifying_rehearsal, note_worker_exit
from .progress_report import write_report
from services.robodojo.scoring import operator_formal_report


def migration_plan(state, source):
    batch = state['formal_batch']
    if batch['status'] != 'interrupted' or state.get('active'):
        raise ValueError('Source must be stopped and fully finalized before migration')
    c = Configuration.from_dict(state['configuration'])
    expected = set(range(c.formal_seed, c.formal_seed + c.formal_episodes))
    rows, seen = [], set()
    for result in state['results']:
        if result.get('mode') != 'formal':
            continue
        layout = result['layout_id']
        if (layout not in expected or layout in seen or result['eval_seed'] != c.formal_collection
                or result['sha256'] != batch['sha256']):
            raise ValueError('Source layout identity is inconsistent')
        seen.add(layout)
        # An interrupted _run_episode can finalize evidence without returning to
        # the batch loop. Never promote that partial evidence to a completed test.
        label = result.get('formal_outcome', 'interrupted')
        if label not in {'success', 'unsuccessful', 'error', 'interrupted'}:
            raise ValueError('Invalid source outcome')
        rows.append({**result, 'formal_outcome': label, 'source_session': str(source)})
    if sum(r['formal_outcome'] != 'interrupted' for r in rows) != batch['completed_episodes']:
        raise ValueError('Source completion count is inconsistent')
    return c, rows, sorted(expected - seen)


def report(config, rows, pending, active, *, status, source, started):
    counts = Counter(r['formal_outcome'] for r in rows)
    result = {'status': status, 'source_session': str(source), 'started': started,
        'updated': time.time(), 'episode_count': config.formal_episodes,
        'completed_episodes': len(rows) - counts['interrupted'], 'resolved_episodes': len(rows),
        'success_count': counts['success'], 'unsuccessful_count': counts['unsuccessful'],
        'error_count': counts['error'], 'interrupted_count': counts['interrupted'],
        'infrastructure_error_count': sum(bool(r.get('infrastructure_errors')) for r in rows
                                         if r['formal_outcome'] != 'interrupted'),
        # Conservative fixed denominator; interruptions are explicitly disclosed.
        'success_rate': counts['success']/config.formal_episodes if status == 'completed' else None,
        'eval_seed': config.formal_collection, 'pending_layouts': list(pending),
        'active': active, 'episodes': sorted(rows, key=lambda r: r['layout_id'])}
    return operator_formal_report(config.root, result)


def worker_result(directory, config, layout, fingerprint, returncode):
    finalized = False
    try:
        state = json.loads((directory/'state.json').read_text())
        results = [r for r in state['results'] if r.get('mode') == 'formal']
        if len(results) != 1:
            raise ValueError('Worker must produce exactly one formal record')
        row = results[0]
        if (row['layout_id'] != layout or row['eval_seed'] != config.formal_collection
                or row['sha256'] != fingerprint):
            raise ValueError('Worker result identity mismatch')
        finalized = True
        label = outcome(row)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        row, label = {'reason': 'worker_error', 'error_type': type(exc).__name__}, 'error'
    note_worker_exit(row, returncode, finalized=finalized)
    return {**row, 'layout_id': layout, 'eval_seed': config.formal_collection,
        'formal_episode_index': layout-config.formal_seed+1, 'sha256': fingerprint,
        'formal_outcome': label, 'worker_returncode': returncode, 'source_session': str(directory)}


def run(source, output, workers, *, bundle_id=None, episodes=50, eval_seed=FORMAL_COLLECTION,
        formal_wall_seconds=None):
    if not 1 <= workers <= 8:
        raise ValueError('workers must be 1..8')
    if formal_wall_seconds is not None and bundle_id is None:
        raise ValueError('Changing the wall limit requires a fresh batch, not migration')
    from services.storage_root import require_artifact_path
    require_artifact_path(output, 'Evaluation artifacts')
    # Hold the existing owner lock throughout; never instantiate its Supervisor.
    with (source/'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((source/'state.json').read_text())
        if bundle_id is None:
            config, rows, pending = migration_plan(state, source)
            manifest = state['bundles'][state['formal_batch']['bundle']]
            lifecycle = json.loads((source/'agent-lifecycle.json').read_text())
            original, original_bundle = Path(lifecycle['source_session']), lifecycle['source_bundle']
        else:
            config = replace(Configuration.from_dict(state['configuration']), root=output,
                formal_seed=0, formal_eval_seed=eval_seed, formal_episodes=episodes)
            manifest = state['bundles'][bundle_id]
            original, original_bundle = source, bundle_id
            rows, pending = [], list(range(episodes))
        if formal_wall_seconds is not None:
            config = replace(config, formal=replace(config.formal, wall_seconds=formal_wall_seconds))
        validate_layouts(config)
        bundle = (source/'bundles'/manifest['directory']).resolve()
        if not bundle.is_relative_to((source/'bundles').resolve()):
            raise ValueError('Invalid source bundle path')
        verify(bundle, manifest)
        # Original successful rehearsal is verified again by every worker.
        evidence = json.loads((original/'state.json').read_text())
        if evidence['bundles'][original_bundle]['sha256'] != manifest['sha256']:
            raise ValueError('Original bundle differs from the frozen evaluation bundle')
        if not any(qualifying_rehearsal(r, original_bundle, evidence['bundles'][original_bundle])
                   for r in evidence['results']):
            raise ValueError('Original bundle has no successful rehearsal')
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        if bundle_id is None:
            with (source/'parallel-migration.json').open('x') as claim:
                json.dump({'output': str(output), 'pid': os.getpid(), 'created': time.time()}, claim)
                claim.flush()
                os.fsync(claim.fileno())
        persist(output/'source-state.json', state)
        project = Path(__file__).resolve().parents[2]
        planner_files = ['RoboDojo/env/planner_manager/curobo_planner.py',
                         'services/robodojo/session.py', 'services/robodojo/trajectory.py',
                         'services/robodojo/protocol.py']
        def fingerprints():
            return {p: hashlib.sha256((project/p).read_bytes()).hexdigest() for p in planner_files}
        infrastructure = fingerprints()
        persist(output/'evaluation-plan.json', {'source_session': str(source),
            'kind': 'migration' if bundle_id is None else 'fresh_batch',
            'sha256': manifest['sha256'], 'workers': workers, 'pending_layouts': pending,
            'preserved_layouts': [r['layout_id'] for r in rows], 'retry_allowed': False,
            'infrastructure_sha256': infrastructure,
            'sim_gpu': config.sim_gpu, 'controller_gpu': config.controller_gpu,
            'formal_wall_seconds': config.formal.wall_seconds})
        active, processes = {}, {}
        started, status = time.time(), 'running'
        lifecycle = {'kind': 'parallel_formal_batch', 'pid': os.getpid(), 'started_at': started,
                     'workers': workers, 'source_session': str(source)}
        def save():
            progress = report(config, rows, pending, active,
                status=status, source=source, started=started)
            persist(output/'formal-report.json', progress)
            persist(output/'agent-lifecycle.json', {**lifecycle, 'status': status})
            try:
                write_report(output, progress)
            except OSError as exc:
                warnings.warn(f'Could not update readable progress report: {exc}')
        def stop(*_):
            raise KeyboardInterrupt
        old = signal.signal(signal.SIGTERM, stop)
        try:
            save()
            while pending or processes:
                while pending and len(processes) < workers:
                    if fingerprints() != infrastructure:
                        raise RuntimeError('Planner infrastructure changed during the batch')
                    layout = pending.pop(0)
                    directory = output/f'layout-{layout:03d}'
                    active[str(layout)] = {'layout_id': layout, 'directory': str(directory),
                        'started': time.time(), 'status': 'reserved'}
                    save()  # Reserve before launch. A crash never re-enqueues it.
                    command = [sys.executable, '-m', 'services.controller.batch',
                        '--source-session', str(original), '--bundle', original_bundle,
                        '--output', str(directory), '--episodes', '1',
                        '--eval-seed', str(config.formal_collection), '--first-layout', str(layout),
                        '--formal-wall-seconds', str(config.formal.wall_seconds)]
                    try:
                        with (output/f'layout-{layout:03d}.log').open('xb') as log:
                            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
                    except OSError:
                        rows.append(worker_result(directory, config, layout, manifest['sha256'], -1))
                        del active[str(layout)]
                    else:
                        processes[layout] = (process, directory)
                        active[str(layout)].update(pid=process.pid, status='running')
                    save()
                for layout, (process, directory) in list(processes.items()):
                    code = process.poll()
                    if code is not None:
                        rows.append(worker_result(directory, config, layout, manifest['sha256'], code))
                        del active[str(layout)], processes[layout]
                        save()
                if processes:
                    time.sleep(2)
            status = 'completed'
        except BaseException:
            status = 'interrupted'
            raise
        finally:
            # Terminate supervisors, not their groups: their finally blocks own
            # container/simulator cleanup. Do not forcibly kill unrelated jobs.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            for process, _ in processes.values():
                if process.poll() is None:
                    process.terminate()
            for layout, (process, directory) in processes.items():
                try:
                    code = process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    active[str(layout)]['status'] = 'cleanup_pending'
                    continue
                row = worker_result(directory, config, layout, manifest['sha256'], code)
                row['formal_outcome'] = 'interrupted'
                rows.append(row)
                del active[str(layout)]
            save()
            signal.signal(signal.SIGTERM, old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-batch', type=Path)
    source.add_argument('--source-session', type=Path)
    parser.add_argument('--bundle', help='Required with --source-session for a fresh operator batch')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--eval-seed', type=int, default=FORMAL_COLLECTION)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--formal-wall-seconds', type=int,
                        help='Per-episode wall limit override; fresh batches only')
    args = parser.parse_args()
    if bool(args.source_session) != bool(args.bundle):
        parser.error('--bundle is required only with --source-session')
    run((args.source_batch or args.source_session).resolve(), args.output.resolve(), args.workers,
        bundle_id=args.bundle, episodes=args.episodes, eval_seed=args.eval_seed,
        formal_wall_seconds=args.formal_wall_seconds)


if __name__ == '__main__':
    main()
