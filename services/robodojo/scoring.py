"""Operator-only native partial-score reporting; never sent through robot MCP.

Use RoboDojo's run_eval rule: success=1, otherwise native process score/100.
Do not maximize historical scores or reconstruct task predicates from images.
Legacy early exits can be read from their final trusted operator snapshot.
"""
import json
import math
from pathlib import Path


def _read(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and 0 <= value <= 1 else None


def episode_score(sim, episode_id=None):
    """Read a finalized outcome, or last native snapshot after an early exit."""
    sim = Path(sim)
    state = _read(sim/'operator_state.json')
    if episode_id and state.get('episode_id') != episode_id:
        return {'score': None, 'source': 'identity_mismatch'}
    outcome = _read(sim/'evaluation_outcome.json')
    reward = state.get('reward') or {}
    if outcome.get('status') == 'invalid_native_layout' or reward.get('unstable_native_layout'):
        return {'score': None, 'source': 'invalid_native_layout'}
    score = _score(outcome.get('native_score'))
    source = 'native_outcome'
    if score is None:
        source = 'last_native_snapshot'
        if reward.get('native_success') is True:
            score = 1.0
        else:
            percent = reward.get('native_score_percent')
            score = _score(percent / 100) if type(percent) in (int, float) else None
    return {'score': score, 'source': source if score is not None else 'unavailable',
            'step_id': state.get('step_id'), 'episode_id': state.get('episode_id'),
            'native_complete': outcome.get('complete', False)}


def formal_scores(root, batch, results=None):
    """Aggregate a fixed formal schedule without leaking scores to the agent.

Missing/invalid evidence contributes zero, is explicitly counted, and never
silently reduces the denominator. While running, only the recorded mean is
available. Reading old archives does not modify their episode records.
"""
    root = Path(root)
    rows = batch.get('episodes', []) if results is None else results
    count = batch.get('episode_count', 0)
    if not count:
        return None
    by_id, seen_paths = {}, set()
    # Operator-only migration batches retain their original worker locations.
    roots = {root, *(Path(r['source_session']) for r in rows if r.get('source_session'))}
    for directory in roots:
        for pattern in ('native/results/*/sim/operator_state.json',
                        'formal-workers/episode-*/native/results/*/sim/operator_state.json',
                        'layout-*/native/results/*/sim/operator_state.json',
                        'formal/episode-*/results/robodojo_mcp/*/sim/operator_state.json'):
            for path in directory.glob(pattern):
                if path.resolve() in seen_paths:
                    continue
                seen_paths.add(path.resolve())
                state = _read(path)
                if state.get('episode_mode') == 'formal' and state.get('episode_id'):
                    by_id.setdefault(state['episode_id'], []).append(path.parent)
    entries = []
    for row in rows:
        eid = row.get('episode_id')
        paths = by_id.get(eid, [])
        value = episode_score(paths[0], eid) if len(paths) == 1 else {'score': None, 'source': 'unavailable'}
        # Direct-control results also retain the trusted native outcome. Useful
        # when a simulator never produced a final operator snapshot.
        if value['score'] is None and not paths:
            outcome = row.get('native_outcome') or {}
            score = _score(outcome.get('native_score'))
            if outcome.get('status') != 'invalid_native_layout' and score is not None:
                value = {'score': score, 'source': 'native_outcome'}
        entries.append({'episode_index': row.get('formal_episode_index', row.get('index', 0) + 1),
                        'episode_id': eid, **value})
    values = [r['score'] for r in entries if r['score'] is not None]
    finished = batch.get('status') == 'completed'
    average = sum(values)/count if finished else None
    return {'schema': 'native_formal_scores_v1', 'episode_count': count,
            'recorded_score_count': len(values), 'missing_score_count': count-len(values),
            'score_complete': len(values) == count,
            'average_score': average,
            'average_score_percent': average*100 if average is not None else None,
            'recorded_average_score_percent': sum(values)/len(values)*100 if values else None,
            'missing_score_policy': 'zero_in_fixed_denominator', 'episodes': entries}


def operator_formal_report(root, batch, results=None):
    """Decorate only host/operator responses, never the published agent report."""
    if not batch:
        return batch
    return {**batch, 'score_summary': formal_scores(root, batch, results)}
