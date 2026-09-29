"""Auto-research sessions: trusted launch controls and read-only monitoring.

Never instantiate the robot supervisor here: it owns an exclusive lock and live
environment state. Dashboard reads are snapshots, not another robot connection.
"""
from datetime import datetime, timezone
import ctypes
import io
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import threading
import time

import numpy as np
from PIL import Image

from services.codex_usage import CodexUsageReader
from services.controller.storage import persist
from services.controller.recording import read_regular
from services.controller.supervisor import qualifying_rehearsal
from services.robodojo.timeouts import task_timeouts
from services.robodojo.scoring import operator_formal_report
from services.robodojo.demonstrations import DEMONSTRATION_CONTEXTS
from services.controller.config import EXPLORATION_COLLECTIONS, FORMAL_COLLECTION
from .observations import process_matches

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')

RUN_ID = re.compile(r"^\d{8}T\d{6}Z_[a-f0-9]{10}$")
# Exactly the options of scripts/configure_auto_research.py.
DEFAULTS = dict(task='make_kong', sim_gpu='0', research_gpu='1', episodes=5,
                formal_seed=0, formal_episodes=50, formal_eval_seed=FORMAL_COLLECTION, formal_workers=4,
                exploration_start_seed=0, exploration_collections=','.join(map(str, EXPLORATION_COLLECTIONS)), exploration_envs=1, workspace_gib=20,
                demonstration_context='terminal_state', image='robodojo-official:dev',
                agent_cli='codex', agent_provider='openrouter')
BOUNDS = dict(episodes=(2, 1000), formal_seed=(0, 1000000), formal_episodes=(1, 100), formal_workers=(1, 8),
              exploration_envs=(1, 8), formal_eval_seed=(0, 1000000), exploration_start_seed=(0, 1000000),
              workspace_gib=(1, 1024))
IMAGE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_./:@-]{0,254}')


def read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def relay_usage(path):
    """Token totals from the Claude Messages relay's JSONL, in the Codex usage shape."""
    totals = dict.fromkeys(('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens',
                            'output_tokens', 'total_tokens'), 0)
    calls, cost = 0, 0.0
    try:
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                calls += 1
                totals['input_tokens'] += row.get('input_tokens', 0) + row.get('cache_read_input_tokens', 0) \
                    + row.get('cache_creation_input_tokens', 0)
                totals['cached_input_tokens'] += row.get('cache_read_input_tokens', 0)
                totals['cache_write_input_tokens'] += row.get('cache_creation_input_tokens', 0)
                totals['output_tokens'] += row.get('output_tokens', 0)
                cost += row.get('cost_usd', 0) or 0
    except OSError:
        return None
    totals['total_tokens'] = totals['input_tokens'] + totals['output_tokens']
    return {'token_usage': totals, 'total_session_usage': totals, 'model_calls': calls,
            'reported_cost_usd': round(cost, 6)}


def tail(path, maximum=65536):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-maximum))
            return stream.read(maximum).decode('utf-8', errors='replace')
    except OSError:
        return ''


class Sessions:
    def __init__(self, project, tasks, started_at=None):
        self.project = Path(project).resolve()
        self.root = (self.project/'runtime').resolve()/'auto-research'
        self.profile = (self.project/'runtime/operator/auto-research.json').resolve()
        self.tasks = tasks
        self.started_at = started_at
        self.lock = threading.RLock()
        self.usage_readers = {}
        self.observed_active = set()

    def directory(self, run_id, *, history=False):
        """A session directory. Sessions older than this server are review-only (history=True)."""
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise ValueError('Invalid auto-research session ID')
        path = self.root/run_id
        if not path.is_dir() or path.is_symlink() or path.resolve().parent != self.root.resolve():
            raise ValueError('Unknown auto-research session')
        created = datetime.strptime(run_id[:16], '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).timestamp()
        if not history and self.started_at is not None and created < self.started_at:
            if self.alive(path, read_json(path/'agent-lifecycle.json')):
                self.observed_active.add(run_id)
            elif run_id not in self.observed_active:
                raise ValueError('Session predates this dashboard server')
        return path

    def settings(self, supplied):
        supplied = dict(supplied)
        if set(supplied)-set(DEFAULTS):
            raise ValueError('Unknown auto-research setup field')
        values = {**DEFAULTS, **supplied}
        from services.mcp_contract import Contract
        Contract('auto-research', observation_profile='official', environments=values['exploration_envs'])
        if values['task'] not in self.tasks():
            raise ValueError('Select an installed task')
        if values['demonstration_context'] not in DEMONSTRATION_CONTEXTS:
            raise ValueError('Invalid demonstration_context')
        if values['agent_cli'] not in ('codex', 'claude'):
            raise ValueError('agent_cli must be codex or claude')
        if values['agent_provider'] not in ('openrouter', 'claude-login', 'anthropic'):
            raise ValueError('Invalid agent_provider')
        for name, (low, high) in BOUNDS.items():
            if type(values[name]) is not int or not low <= values[name] <= high:
                raise ValueError(f'{name} must be an integer in {low}..{high}')
        for name in ('sim_gpu', 'research_gpu'):
            if not isinstance(values[name], str) or not re.fullmatch(r'\d+|GPU-[a-fA-F0-9-]+', values[name]):
                raise ValueError(f'Invalid {name}')
        if values['sim_gpu'] == values['research_gpu']:
            raise ValueError('Simulator and research GPUs must differ')
        if not isinstance(values['image'], str) or not IMAGE.fullmatch(values['image']):
            raise ValueError('Invalid image reference')
        if not isinstance(values['exploration_collections'], str) or not re.fullmatch(r'\d+(,\d+)*', values['exploration_collections']):
            raise ValueError('exploration_collections must be comma-separated collection numbers')
        return values

    def prepare(self, supplied):
        values = self.settings(supplied)
        if values['agent_cli'] == 'claude' and values['agent_provider'] == 'claude-login':
            # Surface a missing or rejected experiment token before a session exists.
            from harness.claude_cli import experiment_login
            try:
                experiment_login.check()
            except (RuntimeError, OSError) as exc:
                raise ValueError(str(exc)) from None
        command = [str(self.project/'runtime/envs/robodojo/bin/python'),
                   str(self.project/'scripts/configure_auto_research.py')]
        for key, value in values.items():
            command += ['--'+key.replace('_', '-'), str(value)]
        with self.lock:
            result = subprocess.run(command, cwd=self.project, capture_output=True, text=True, timeout=180)
            if result.returncode:
                raise ValueError('Configuration/preflight failed: '+result.stderr[-2000:])
            path = Path(result.stdout.strip())
            directory = self.directory(path.parent.name)
            if path != directory/'operator-input.json' or not path.is_file():
                raise ValueError('Configuration command returned an unexpected path')
            self.profile.parent.mkdir(parents=True, exist_ok=True)
            persist(self.profile, values)
            return self.detail(directory.name)

    def command(self, directory):
        return 'cd '+shlex.quote(str(self.project))+' && '+shlex.join([
            'bash', 'scripts/start_auto_research_agent.sh', '--config', str(directory/'operator-input.json')])

    def launch(self, run_id):
        with self.lock:
            directory = self.directory(run_id)
            if any((directory/name).exists() for name in ('launch-request.json', 'agent-lifecycle.json', 'agent-workspace', 'state.json')):
                raise ValueError('Session was already launched; prepare a fresh configuration')
            if not (directory/'operator-input.json').is_file():
                raise ValueError('No prepared configuration')
            persist(directory/'launch-request.json', {'requested_at': time.time()})
            with (directory/'agent.log').open('ab') as log:
                process = subprocess.Popen(['bash', str(self.project/'scripts/start_auto_research_agent.sh'),
                    '--config', str(directory/'operator-input.json'), '--non-interactive'],
                    cwd=self.project, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            threading.Thread(target=process.wait, daemon=True).start()  # Reap without owning its lifetime.
            return {'run_id': run_id, 'pid': process.pid, 'status': 'launching'}

    @staticmethod
    def alive(directory, lifecycle):
        pid = lifecycle.get('pid')
        if type(pid) is not int or pid <= 1 or lifecycle.get('finished_at'):
            return False
        try:
            args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            if lifecycle.get('kind') == 'formal_batch':
                return b'services.controller.batch' in args and str(directory).encode() in args
            return b'harness.codex_cli.auto_research' in args and str(directory/'operator-input.json').encode() in args
        except OSError:
            return False

    def stop(self, run_id):
        directory = self.directory(run_id)
        lifecycle = read_json(directory/'agent-lifecycle.json')
        if not self.alive(directory, lifecycle):
            raise ValueError('No matching live launcher; nothing was signaled')
        # Pin process identity before rechecking; never signal a recycled PID.
        # Some bundled Python builds omit os.pidfd_open/signal.pidfd_send_signal
        # despite host libc/kernel support. Use libc without falling back to a
        # racy numeric-PID kill.
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            open_pid, send = libc.pidfd_open, libc.pidfd_send_signal
        except AttributeError as exc:
            raise ValueError('Host lacks PID-safe stop support; stop the launch terminal instead') from exc
        open_pid.argtypes, open_pid.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
        send.argtypes, send.restype = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint], ctypes.c_int
        fd = open_pid(lifecycle['pid'], 0)
        if fd < 0:
            raise OSError(ctypes.get_errno(), 'Cannot pin launcher identity')
        try:
            if not self.alive(directory, lifecycle):
                raise ValueError('Launcher exited; nothing was signaled')
            if send(fd, signal.SIGTERM, None, 0) < 0:
                raise OSError(ctypes.get_errno(), 'Cannot signal pinned launcher')
        finally:
            os.close(fd)
        return {'run_id': run_id, 'status': 'stop requested; waiting for cleanup'}

    def summary(self, directory):
        config = read_json(directory/'operator-input.json')
        state = read_json(directory/'state.json')
        lifecycle = read_json(directory/'agent-lifecycle.json')
        alive = self.alive(directory, lifecycle)
        status = lifecycle.get('status', 'launching' if (directory/'launch-request.json').exists() else 'prepared')
        if status in ('preparing', 'running') and not alive:
            status = 'interrupted'
        active = state.get('active') or {}
        results = state.get('results', [])
        rehearsed = [bundle for bundle, manifest in state.get('bundles', {}).items()
                     if any(qualifying_rehearsal(result, bundle, manifest) for result in results+state.get('imported_rehearsals', []))]
        formal = next((r for r in reversed(results) if r.get('mode') == 'formal'), None)
        batch = operator_formal_report(directory, state.get('formal_batch'),
                                      [r for r in results if r.get('mode') == 'formal'])
        if batch:
            batch_status = batch['status'] if batch['status'] != 'running' or alive else 'interrupted'
            stage = f"Formal batch {batch_status}: {batch['completed_episodes']}/{batch['episode_count']} completed, {batch['success_count']} successes"
            if batch.get('success_rate') is not None:
                stage += f" ({100*batch['success_rate']:.1f}%)"
        elif formal:
            stage = 'Formal complete' if formal.get('task_complete') is True else 'Formal ended without confirmed success'
        elif state.get('formal_reserved'):
            stage = 'Formal running' if alive and active.get('mode') == 'formal' else 'Formal reserved; no retry'
        elif active.get('mode') == 'rehearsal':
            stage = 'Rehearsal running' if alive else 'Rehearsal interrupted'
        elif rehearsed:
            stage = 'Successful rehearsal recorded'
        elif state.get('interactive_success'):
            stage = 'Autonomous development / rehearsal'
        else:
            stage = 'Manual robot control' if state else 'Not started'
        return dict(run_id=directory.name, task=config.get('task'), status=status, alive=alive, stage=stage,
            demonstration_context=config.get('demonstration_context', 'none'),
            sim_gpu=config.get('sim_gpu'), research_gpu=config.get('training_gpu'),
            exploration_used=state.get('exploration_started', 0), exploration_limit=len(config.get('exploration_seeds', [])),
            exploration_pool=state.get('exploration_pool'),
            manual_success=bool(state.get('interactive_success')), formal_reserved=bool(state.get('formal_reserved')),
            active=active, formal_batch=batch, rehearsed_bundles=rehearsed,
            gemini=self.gemini(state), agent=self.agent(directory, config),
            launchable=status == 'prepared' and not (directory/'agent-workspace').exists())

    @staticmethod
    def agent(directory, config):
        """Agent CLI and provider; for claude-login, the dedicated account it bills."""
        cli, provider = config.get('agent_cli', 'codex'), config.get('agent_provider', 'openrouter')
        result = {'cli': cli, 'provider': provider if cli == 'claude' else None, 'account': None}
        if cli == 'claude' and provider == 'claude-login':
            relay = read_json(directory/'model-relay.json')  # Written at launch; identity only.
            if relay:
                result['account'] = (relay.get('account') or {}).get('email')
            else:
                from harness.claude_cli import experiment_login
                result['account'] = experiment_login.identity().get('email')
                result['pending'] = True  # Resolved again when the session launches.
        return result

    def detail(self, run_id, *, history=False):
        directory = self.directory(run_id, history=history)
        state = read_json(directory/'state.json')
        usage = relay_usage(directory/'model-usage.jsonl')  # Claude sessions: the host relay's log.
        for rollout in sorted((directory/'codex-home/sessions').rglob('*.jsonl')):
            try:
                reader = self.usage_readers.setdefault(str(rollout), CodexUsageReader())
                usage = reader.read(rollout)
            except OSError:
                continue
        return {**self.summary(directory), 'command': self.command(directory),
                'release': self.release(directory, state),
                'live_observation': self.live_observation(directory),
                'configuration': read_json(directory/'operator-input.json'), 'usage': usage,
                'results': state.get('results', [])[-100:], 'bundles': state.get('bundles', {}),
                'workspace': str(directory/'agent-workspace'), 'artifacts': str(directory/'published'),
                'agent_log': tail(directory/'agent.log'), 'frontend_log': tail(directory/'frontend.log')}

    @staticmethod
    def latest_native(directory):
        """Newest simulator state; during a concurrent formal batch, the lowest live episode.

        Preferring a stable live worker keeps the view from jumping between episodes.
        """
        candidates = []
        paths = [*(directory/'native/results').glob('*/sim/operator_state.json'),
                 *directory.glob('exploration/episode-*/native/results/*/sim/operator_state.json'),
                 *directory.glob('formal-workers/episode-*/native/results/*/sim/operator_state.json')]
        for path in paths:
            try:
                state = json.loads(read_regular(path, 65536))
                if not isinstance(state, dict):
                    continue
                candidates.append((path.stat().st_mtime_ns, str(path), path.parent, state))
            except (OSError, ValueError):
                continue
        if not candidates:
            return None, None
        live = sorted((c for c in candidates if 'formal-workers' in c[1]
                       and process_matches(c[3].get('simulator_pid'), c[2])), key=lambda c: c[1])
        _, _, sim, state = live[0] if live else max(candidates, key=lambda entry: entry[:2])
        return sim, state

    @staticmethod
    def observation_path(sim, state):
        relative = Path(state.get('observation_file') or '')
        if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.npz':
            raise ValueError('Invalid native observation path')
        return sim/relative

    def live_observation(self, directory):
        sim, state = self.latest_native(directory)
        if state is None:
            return None
        cameras = [camera for camera in CAMERAS if camera in state.get('cameras', [])]
        try:
            available = self.observation_path(sim, state).is_file()
        except ValueError:
            available = False
        return {'native_run': sim.parent.name, 'episode_id': state.get('episode_id'),
                'episode_mode': state.get('episode_mode'), 'step_id': state.get('step_id'),
                'episode_step_limit': state.get('episode_step_limit'),
                'active': process_matches(state.get('simulator_pid'), sim),
                'updated_at_unix_s': state.get('updated_at_unix_s'),
                'available': available, 'cameras': cameras if available else [],
                'reward': state.get('reward', {}), 'reason': state.get('reason')}

    def frame(self, run_id, camera, native_run, step_id, *, history=False):
        directory = self.directory(run_id, history=history)
        if camera not in CAMERAS:
            raise ValueError('Unknown camera')
        sim, state = self.latest_native(directory)
        if state is None:
            raise FileNotFoundError('No native observation yet')
        # A reset/step between metadata and image fetch must not mix episodes/frames.
        if sim.parent.name != native_run or str(state.get('step_id')) != str(step_id):
            raise FileNotFoundError('Observation changed; refresh the live view')
        raw = read_regular(self.observation_path(sim, state), 64*1024*1024)
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if camera not in archive:
                raise FileNotFoundError('Camera is unavailable')
            rgb = archive[camera]
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
                raise ValueError('Invalid RGB observation')
            stream = io.BytesIO()
            Image.fromarray(rgb[..., :3]).save(stream, format='JPEG', quality=88, optimize=True)
        return stream.getvalue()

    def artifact(self, run_id, logical_path, *, history=False):
        directory = self.directory(run_id, history=history)
        relative = Path(logical_path)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[:3] != ('runtime', 'autonomous_controller', 'runs'):
            raise ValueError('Only published run artifacts are available')
        path = directory/'published'/Path(*relative.parts[2:])
        if path.suffix not in {'.png', '.json', '.jsonl', '.log', '.txt'}:
            raise ValueError('Select a PNG image or text artifact')
        # Reject links/special files, and bound responses before allocation.
        data = read_regular(path, 4*1024*1024)
        return data, 'image/png' if path.suffix == '.png' else 'text/plain; charset=utf-8'

    @staticmethod
    def gemini(state):
        """Session Gemini ledger: spend against the $10 cap, and calls per phase."""
        from services.controller.billing import NANODOLLARS, SESSION_LIMIT
        spend = state.get('gemini_spend')
        if not isinstance(spend, dict):
            return None
        usage = state.get('usage', {})
        reported, reserved = spend.get('reported', 0) / NANODOLLARS, spend.get('reserved', 0) / NANODOLLARS
        return {'limit_usd': SESSION_LIMIT / NANODOLLARS, 'reported_usd': reported, 'reserved_usd': reserved,
                'remaining_usd': 0 if spend.get('blocked') else max(0, SESSION_LIMIT / NANODOLLARS - reported - reserved),
                'blocked': bool(spend.get('blocked')),
                'calls': {phase: (usage.get(phase) or {}).get('calls', 0) for phase in ('development', 'formal')}}

    @staticmethod
    def _release_bundle(directory, state):
        """(bundle ID, frozen bundle path, submitted) for the submitted or newest rehearsed bundle."""
        batch = state.get('formal_batch') or {}
        bundles = state.get('bundles', {})
        results = state.get('results', [])+state.get('imported_rehearsals', [])
        ident = batch.get('bundle') or next((b for b in reversed(list(bundles))
            if any(qualifying_rehearsal(r, b, bundles[b]) for r in results)), None)
        if ident not in bundles:
            return None
        return ident, directory/'bundles'/bundles[ident]['directory'], batch.get('bundle') == ident

    def _counterpart(self, task, exclude):
        """Newest other session's release bundle for the std/random variant of ``task``."""
        other = task[:-len('_random')] if task.endswith('_random') else task + '_random'
        if other not in self.tasks():
            return None
        for directory in sorted(self.root.glob('*'), key=lambda p: p.name, reverse=True):
            if directory.name == exclude or directory.is_symlink() or not RUN_ID.fullmatch(directory.name):
                continue
            if read_json(directory/'operator-input.json').get('task') != other:
                continue
            found = self._release_bundle(directory, read_json(directory/'state.json'))
            if found:
                return {'task': other, 'run_id': directory.name, 'bundle': found[0], 'path': str(found[1]),
                        'submitted': found[2]}
        return {'task': other, 'run_id': None}

    def release(self, directory, state):
        """Commands that turn the submitted (or newest rehearsed) bundle into an official checkpoint."""
        found = self._release_bundle(directory, state)
        if not found:
            return None
        ident, bundle, submitted = found
        manifest = state['bundles'][ident]
        task = read_json(directory/'operator-input.json').get('task')
        tools = self.project/'official/xpolicylab'
        releases = self.project/'runtime/official-xpolicylab/releases'
        build = ['bash', str(tools/'build_release.sh')]
        result = {'bundle': ident, 'sha256': manifest['sha256'], 'submitted': submitted, 'path': str(bundle),
                  # sim: the real simulator for one official episode (debug is a fake test env).
                  'official_check': shlex.join(['bash', str(tools/'run_eval.sh'), 'sim', task, str(bundle), '0', '1']),
                  'build_checkpoint': shlex.join(build + [str(releases/f'{directory.name}-{ident[:8]}'), f'{task}={bundle}'])}
        counterpart = self._counterpart(task, directory.name)
        if counterpart:
            # Official sweeps run std and _random variants as separate tasks with their own bundle.
            result['counterpart'] = counterpart
            if counterpart.get('path'):
                result['build_combined'] = shlex.join(build + [
                    str(releases/f'{directory.name}-{ident[:8]}+{counterpart["run_id"]}-{counterpart["bundle"][:8]}'),
                    f'{task}={bundle}', f'{counterpart["task"]}={counterpart["path"]}'])
        return result

    def episodes(self, run_id, *, history=False):
        from .review import episodes
        return episodes(self.directory(run_id, history=history))

    def video(self, run_id, episode, *, history=False):
        from .review import video_path
        return video_path(self.directory(run_id, history=history), episode)

    def status(self, run_id=None, *, history=False):
        sessions = []
        for directory in sorted(self.root.glob('*'), key=lambda p: p.name, reverse=True):
            try:
                self.directory(directory.name, history=history)
            except ValueError:
                continue
            if (directory/'operator-input.json').is_file():
                sessions.append(self.summary(directory))
        selected = run_id or (sessions[0]['run_id'] if sessions else None)
        profile = {**DEFAULTS, **{k: v for k, v in read_json(self.profile).items() if k in DEFAULTS}}
        if selected:
            detail = self.detail(selected, history=history)
            # Launch/stop never act on sessions older than this server.
            detail['launchable'] = detail['launchable'] and not history
        return {'sessions': sessions, 'selected': detail if selected else None,
                'profile': profile, 'task_timeouts': {t: task_timeouts(t) for t in self.tasks()},
                'tasks': self.tasks(), 'demonstration_contexts': list(DEMONSTRATION_CONTEXTS)}
