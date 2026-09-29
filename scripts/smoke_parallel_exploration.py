"""Operator-only two-slot MCP/native smoke test; no Codex or Gemini API calls."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from services.exploration.pool import ExplorationPool
from services.robodojo.timeouts import task_timeouts
from services.robodojo.mcp_server import _write_private_json


def value(reply):
    assert not reply.get('isError'), reply
    return json.loads(reply['content'][0]['text'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True, help='Explicit simulator GPU UUID')
    parser.add_argument('--task', default='make_kong')
    args = parser.parse_args()
    if not args.gpu.startswith('GPU-'):
        parser.error('Use a GPU UUID')
    parent = (PROJECT/'runtime').resolve()/'diagnostics'
    parent.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='parallel-exploration-', dir=parent))
    config = {'task': args.task, 'sim_gpu': args.gpu, 'eval_seed': 0,
        'episode_seconds': task_timeouts(args.task)['episode_seconds'],
        'published_root': str(root/'published'), 'artifact_bytes': 512*1024**2}
    pool = ExplorationPool(root/'pool', count=2, seeds=[0, 1], worker_config=config)
    report = {'root': str(root), 'task': args.task, 'gpu': args.gpu}
    print(json.dumps(report), flush=True)
    try:
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as executor:
            initial = list(executor.map(pool.start, [0, 1]))
            report['parallel_start_seconds'] = time.monotonic() - started
            from PIL import Image
            for reply in initial:
                for block in reply['content']:
                    if block['type'] == 'image':
                        with Image.open(io.BytesIO(base64.b64decode(block['data']))) as image:
                            image.verify()
            observations = [value(r) for r in initial]
            assert observations[0]['episode_id'] != observations[1]['episode_id']
            import numpy as np
            actions = [np.asarray(v['states']).reshape(14).tolist() for v in observations]
            replies = list(executor.map(lambda i: pool.call(i, 'robodojo_step', {'actions': [actions[i]] * 2}), [0, 1]))
            report['steps'] = [value(r)['step_id'] for r in replies]
            assert report['steps'] == [2, 2], report
            pool.call(0, 'robodojo_step', {'actions': [actions[0]]})
            assert value(pool.call(1, 'robodojo_status', {}))['step_id'] == 2
            for i in [0, 1]:
                try:
                    pool.start(i)
                except RuntimeError as exc:
                    assert 'budget exhausted' in str(exc)
                else:
                    raise AssertionError('Budget overrun')
            report['live_budget'] = pool.status()
            report['image_counts'] = [sum(b['type'] == 'image' for b in r['content']) for r in replies]
            assert all(report['image_counts'])
            assert all(value(r).get('frame_sequence', {}).get('manifest_path') for r in replies)
            list(executor.map(pool.finish, [0, 1]))
            assert pool.status()['exploration_complete']
            report['passed'] = True
    except BaseException as exc:
        report.update(passed=False, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            pool.close()
        finally:
            report['final_budget'] = pool.status()
            _write_private_json(root/'report.json', report)
            print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
