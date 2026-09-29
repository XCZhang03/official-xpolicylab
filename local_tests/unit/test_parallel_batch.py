"""Independent operator workers preserve identities and never retry failures."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from services.controller.config import FORMAL_COLLECTION
from services.controller.parallel_batch import migration_plan, outcome, report, worker_result
from test_formal_batch import config


def state(tmp_path):
    c = config(tmp_path)
    object.__setattr__(c, 'root', Path('/mnt/ssd8/parallel-test'))
    configuration = json.loads(json.dumps(asdict(c), default=str))
    rows = [{'mode': 'formal', 'layout_id': i, 'eval_seed': 1, 'sha256': 'frozen',
             'formal_outcome': 'success' if i == 0 else 'error'} for i in range(2)]
    # A finalized partial episode has no batch-loop classification.
    rows.append({'mode': 'formal', 'layout_id': 2, 'eval_seed': 1, 'sha256': 'frozen'})
    return {'configuration': configuration, 'formal_batch': {'status': 'interrupted',
        'sha256': 'frozen', 'completed_episodes': 2}, 'active': None, 'results': rows}


def test_migration_never_requeues_started_or_interrupted_layout(tmp_path):
    s = state(tmp_path)
    c, rows, pending = migration_plan(s, tmp_path)
    assert pending == list(range(3,50))
    assert [r['formal_outcome'] for r in rows] == ['success','error','interrupted']
    assert 'formal_outcome' not in s['results'][2]  # Source is unchanged.
    r = report(c, rows, pending, {}, status='running', source=tmp_path, started=0)
    assert r['success_rate'] is None and r['completed_episodes'] == 2
    assert r['resolved_episodes'] == 3 and r['interrupted_count'] == 1
    for i in pending:
        rows.append({'layout_id': i, 'formal_outcome': 'success'})
    r = report(c, rows, [], {}, status='completed', source=tmp_path, started=0)
    assert r['success_rate'] == 48/50


@pytest.mark.parametrize('damage', ['active', 'running', 'duplicate', 'hash', 'collection', 'count'])
def test_invalid_migrations_fail_closed(tmp_path, damage):
    s = state(tmp_path)
    if damage == 'active': s['active'] = {'layout_id': 3}
    if damage == 'running': s['formal_batch']['status'] = 'running'
    if damage == 'duplicate': s['results'].append(s['results'][0])
    if damage == 'hash': s['results'][0]['sha256'] = 'changed'
    if damage == 'collection': s['results'][0]['eval_seed'] = 0
    if damage == 'count': s['formal_batch']['completed_episodes'] = 3
    with pytest.raises(ValueError): migration_plan(s, tmp_path)


def test_worker_identity_exit_and_recording_errors(tmp_path):
    c, _, _ = migration_plan(state(tmp_path), tmp_path)
    row = {'mode': 'formal', 'layout_id': 7, 'eval_seed': 1, 'sha256': 'frozen',
           'task_complete': True, 'reason': 'episode_ended', 'returncode': 0}
    (tmp_path/'state.json').write_text(json.dumps({'results':[row]}))
    assert worker_result(tmp_path, c, 7, 'frozen', 0)['formal_outcome'] == 'success'
    crashed = worker_result(tmp_path, c, 7, 'frozen', 1)
    assert crashed['formal_outcome'] == 'success' and crashed['worker_returncode'] == 1
    assert crashed['infrastructure_errors'][0]['episode_record_finalized'] is True
    assert worker_result(tmp_path, c, 8, 'frozen', 0)['formal_outcome'] == 'error'
    assert worker_result(tmp_path, c, 7, 'changed', 0)['formal_outcome'] == 'error'
    row['recording_errors'] = ['disk_full']
    assert outcome(row) == 'error'


def test_missing_worker_result_is_not_retried_or_counted_as_success(tmp_path):
    c, _, _ = migration_plan(state(tmp_path), tmp_path)
    row = worker_result(tmp_path, c, 3, 'frozen', 1)
    assert row['formal_outcome'] == 'error' and row['reason'] == 'worker_error'
    assert row['infrastructure_errors'][0]['episode_record_finalized'] is False
    assert row['formal_episode_index'] == 4


@pytest.mark.parametrize('wall_seconds', [None, 1200])
def test_fresh_dispatch_runs_fifty_once_with_bounded_concurrency(tmp_path, monkeypatch, wall_seconds):
    from services.controller import parallel_batch as module
    source, output = tmp_path/'source', tmp_path/'batch'
    source.mkdir()
    s = state(tmp_path)
    s['bundles'] = {'bundle': {'directory': 'bundle', 'sha256': 'frozen'}}
    (source/'state.json').write_text(json.dumps(s))
    relative = Path.is_relative_to
    monkeypatch.setattr(Path, 'is_relative_to', lambda p, base: True if str(base) == '/mnt/ssd8' else relative(p, base))
    monkeypatch.setattr(module, 'verify', lambda *a: None)
    monkeypatch.setattr(module, 'validate_layouts', lambda *a: None)
    monkeypatch.setattr(module, 'qualifying_rehearsal', lambda *a: True)
    monkeypatch.setattr(module.time, 'sleep', lambda *a: None)
    launched, live = [], set()
    peaks = []
    class Process:
        def __init__(self, command, **kwargs):
            expected_wall = wall_seconds or s['configuration']['formal']['wall_seconds']
            assert int(command[command.index('--formal-wall-seconds')+1]) == expected_wall
            self.layout = int(command[command.index('--first-layout')+1])
            self.directory = Path(command[command.index('--output')+1])
            self.directory.mkdir()
            self.pid, self.polls = 123+self.layout, 0
            launched.append(self.layout)
            live.add(self.layout)
            peaks.append(len(live))
        def poll(self):
            self.polls += 1
            if self.polls < 3: return None
            live.discard(self.layout)
            row = {'mode': 'formal', 'layout_id': self.layout, 'eval_seed': FORMAL_COLLECTION, 'sha256': 'frozen',
                   'task_complete': self.layout % 2 == 0, 'reason': 'exit', 'returncode': 0}
            (self.directory/'state.json').write_text(json.dumps({'results': [row]}))
            return 0
    monkeypatch.setattr(module.subprocess, 'Popen', Process)
    module.run(source, output, 4, bundle_id='bundle', formal_wall_seconds=wall_seconds)
    plan = json.loads((output/'evaluation-plan.json').read_text())
    assert plan['formal_wall_seconds'] == (wall_seconds or s['configuration']['formal']['wall_seconds'])
    final = json.loads((output/'formal-report.json').read_text())
    assert launched == list(range(50)) and max(peaks) == 4 and not live
    assert final['status'] == 'completed' and final['completed_episodes'] == 50
    assert final['success_rate'] == .5 and len(final['episodes']) == 50
    assert not final['active'] and not final['pending_layouts']
    assert json.loads((source/'state.json').read_text()) == s
    with pytest.raises(FileExistsError): module.run(source, output, 4, bundle_id='bundle')


def test_migration_cannot_change_time_limit(tmp_path):
    from services.controller.parallel_batch import run
    with pytest.raises(ValueError, match='fresh batch'):
        run(tmp_path/'source', tmp_path/'output', 2, formal_wall_seconds=1200)
