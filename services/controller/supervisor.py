"""Durable trial/episode owner. Frontends must never instantiate a second owner."""
from __future__ import annotations

from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import uuid

from .gateway import Gateway, MUTATING, complete, sanitized
from .timing import Timings
from .errors import public_failure
from .sandbox import DockerSandbox, ExecutionDeadline, deadline
from .storage import persist, snapshot, verify, project_directory
from .recording import RunRecording
from .billing import NANODOLLARS, SESSION_LIMIT, nanodollars

# Beyond a formal episode's own wall limit: startup, evaluation and cleanup.
WORKER_GRACE_SECONDS = 10 * 60
# After one SIGTERM, time for a worker's own container/simulator cleanup before SIGKILL.
WORKER_KILL_SECONDS = 3 * 60


def qualifying_rehearsal(result, bundle_id, manifest):
    """Shared evidence rule for admission and the operator's read-only display."""
    return (result.get('mode') == 'rehearsal'
            and result.get('bundle') == bundle_id
            and result.get('sha256') == manifest['sha256']
            and result.get('task_complete') is True
            and not any(result.get(key) for key in ('control_uncertain', 'evaluation_error',
                'error_type', 'output_error', 'recording_errors'))
            and result.get('reason') in {'episode_ended', 'exit'}
            and (result.get('returncode') == 0 or result.get('finalization_timeout') is True))


def formal_outcome(result):
    """Task outcome from episode evidence, not a later infrastructure failure."""
    healthy = (not any(result.get(k) for k in ('control_uncertain', 'evaluation_error',
        'error_type', 'output_error', 'recording_errors'))
        and result.get('reason') in {'episode_ended', 'exit'}
        and (result.get('returncode') == 0 or result.get('finalization_timeout') is True))
    if healthy and result.get('task_complete') is True:
        return 'success'
    if healthy and result.get('task_complete') is False:
        return 'unsuccessful'
    return 'error'


def note_worker_exit(result, returncode, *, finalized):
    """Keep post-evaluation process health separate from the trusted task result."""
    result['worker_returncode'] = returncode
    if returncode not in (None, 0):
        result.setdefault('infrastructure_errors', []).append({
            'stage': 'worker_exit', 'type': 'WorkerExit', 'returncode': returncode,
            'episode_record_finalized': finalized,
            'reason': ('Worker exited abnormally after saving its episode result.' if finalized
                       else 'Worker exited abnormally without a finalized episode result.')})


def remove_container(name):
    """Force-remove an owned controller container; never a name we did not create."""
    import re
    if not re.fullmatch(r"robodojo-controller-[a-f0-9]{32}", name or ""):
        raise ValueError("Invalid persisted container identity")
    result = subprocess.run(["docker", "rm", "--force", name], capture_output=True, timeout=30)
    if result.returncode and b"No such container" not in result.stderr:
        raise RuntimeError("Cannot clean interrupted container; trial remains closed")


def _stop_orphaned_workers(root, pids):
    """Signal only live processes that are provably this session's formal workers."""
    for pid in pids:
        try:
            args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if b'services.controller.formal_worker' in args and str(root).encode() in args:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass


class Supervisor:
    def __init__(self, config, backend_factory, gemini=None, *, runner_factory=DockerSandbox,
                 worker_factory=None, published_root=None):
        self.config, self.backend_factory, self.gemini = config, backend_factory, gemini
        self.runner_factory = runner_factory
        # Parallel formal episodes run as separate worker processes (formal_worker.py).
        self.worker_factory = worker_factory
        # Workers publish into their parent session's agent-visible run tree.
        self.published_root = published_root or config.root / "published"
        root = config.root
        root.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(root, 0o700)
        self._lock = (root / "owner.lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self._lock.close()
            raise RuntimeError("Trial already has a supervisor")
        self.backend = self.gateway = None
        self._interactive_token = None
        self._recording = None
        self.exploration_pool = None
        self.phase = "development"
        # Broker threads charge Gemini while the batch loop records results.
        self._state_lock = threading.RLock()
        configuration = json.loads(json.dumps(asdict(config), default=str))
        self.state_path = root / "state.json"
        try:
            if self.state_path.exists():
                self.state = json.loads(self.state_path.read_text())
                # Legacy cumulative rehearsal-time accounting no longer limits admission.
                self.state.pop('development_reserved_seconds', None)
                self.state['configuration'].pop('training_seconds', None)
                self.state['configuration'].setdefault('demonstration_context', 'none')
                self.state['configuration'].setdefault('formal_episodes', 1)
                self.state['configuration'].setdefault('formal_eval_seed', None)
                self.state['configuration'].setdefault('formal_workers', 1)
                self.state['configuration'].setdefault('exploration_envs', 1)
                for key in ('mode', 'observation_profile'):
                    self.state['configuration'].setdefault(key, getattr(type(config), key))
                for key in ('student_model', 'student_reasoning_effort', 'student_max_model_calls', 'teacher_source_session'):
                    self.state['configuration'].pop(key, None)
                for phase in ('development', 'formal', 'training'):
                    self.state['configuration'][phase].pop('gemini_calls', None)
                    self.state['configuration'][phase].pop('gemini_tokens', None)
                if self.state["configuration"] != configuration:
                    raise ValueError("A trial's saved configuration cannot change")
                if 'gemini_spend' not in self.state:
                    usage = self.state.setdefault('usage', {p: {"calls": 0, "tokens": 0, "reported_tokens": 0, "cost_usd": 0.0}
                                                            for p in ("development", "formal")})
                    # Old aggregate usage did not record whether every cost was
                    # reported. Never grant a fresh allowance over unknown spend.
                    used = any(u['calls'] for u in usage.values())
                    self.state['gemini_spend'] = {'reported': sum(nanodollars(u['cost_usd']) for u in usage.values()),
                                                  'reserved': 0, 'blocked': used}
                    self.state['schema'] = max(2, int(self.state.get('schema', 1)))
                    self._save()
                if self.state.get("active"):
                    active = self.state["active"]
                    if active.get("container"):
                        remove_container(active["container"])
                    self.state["results"].append({**active, "reason": "supervisor_interrupted", "task_complete": None})
                    self.state["active"] = None
                    self._save()
                if (self.state.get('formal_batch') or {}).get('status') == 'running':
                    self._recover_parallel_formal()
                    self.state['formal_batch']['status'] = 'interrupted'
                    self.state['formal_batch']['finished'] = time.time()
                    self._save()
            else:
                self.state = {"schema": 3, "configuration": configuration, "exploration_started": 0,
                    "interactive_success": None, "formal_reserved": False, "active": None,
                    "bundles": {}, "results": [], "audit_bytes": 0,
                    "gemini_spend": {"reported": 0, "reserved": 0, "blocked": False},
                    "usage": {p: {"calls": 0, "tokens": 0, "reported_tokens": 0, "cost_usd": 0.0}
                              for p in ("development", "formal")}}
                self._save()
        except BaseException:
            self._lock.close()
            raise

    def _save(self):
        with self._state_lock:
            persist(self.state_path, self.state)

    def audit(self, event):
        raw = (json.dumps({"time": time.time(), **event}, allow_nan=False) + "\n").encode()
        # Recording quota is trial-wide; failure closes control rather than
        # silently continuing without a trustworthy interaction record.
        maximum = self.config.development.artifact_bytes + self.config.formal.artifact_bytes*self.config.formal_episodes
        if self.state["audit_bytes"] + len(raw) > maximum:
            if self.gateway:
                self.gateway.terminal = True
            raise RuntimeError("Trial recording budget exhausted")
        self.state["audit_bytes"] += len(raw)
        self._save()
        with (self.config.root / "mcp.jsonl").open("ab") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if self._recording is not None:
            try:
                self._recording.event(event)
            except Exception as exc:
                self._recording.errors.append(f"trace: {type(exc).__name__}")
                raise

    def _begin_recording(self, mode, ident, maximum):
        previous = self._recording
        self._recording = RunRecording(self.published_root, f"{mode}-{ident}", maximum,
                                       observation_profile=self.config.observation_profile)
        # A standalone read-only file-view workspace. Frontends may mount ONLY
        # published/ at runtime/autonomous_controller; never grant access to config.root.
        view = self.config.root / "agent-view" / "runtime"
        view.mkdir(parents=True, exist_ok=True)
        link = view / "autonomous_controller"
        target = self.published_root
        if link.is_symlink():
            if link.resolve() != target:
                raise RuntimeError("Unexpected artifact-view symlink")
        elif link.exists():
            raise RuntimeError("Artifact-view path already occupied")
        else:
            link.symlink_to(target, target_is_directory=True)
        return previous

    def _publish_sequence(self, value):
        if self._recording is None or not value.get("frame_sequence"):
            return None
        try:
            return self._recording.sequence(value, self.backend.frame_workspace)
        except Exception as exc:
            self._recording.errors.append(f"frame_sequence: {type(exc).__name__}")
            raise

    def _boundary(self, label, token, *, propagate_timeout=False):
        """Record a read-only boundary frame; never place its content in summary."""
        if self._recording is None or token is None or self.gateway is None:
            return
        try:
            with deadline(30):
                response = self.gateway.handle(token, {"jsonrpc": "2.0", "id": f"supervisor-{label}",
                    "method": "tools/call", "params": {"name": "robodojo_observe", "arguments": {}}})
            if "error" in response:
                raise RuntimeError("Boundary observation failed")
        except (Exception, ExecutionDeadline) as exc:
            if propagate_timeout and isinstance(exc, ExecutionDeadline):
                raise
            # Preserve the original execution result even if terminal observation
            # is unavailable after an infrastructure error.
            self._recording.errors.append(f"{label}_observation: {type(exc).__name__}")
            try:
                self._recording.trace({"event": "boundary_unavailable", "boundary": label, "error": type(exc).__name__})
            except Exception:
                pass

    def _end_recording(self, previous, bundle, output, result):
        try:
            try:
                result["api_budget"] = self.budget()
            except Exception:
                result["api_budget"] = None  # A broken broker must not discard run evidence.
            result.update(self._recording.finish(bundle, output, result))
        except Exception as exc:
            result["recording_errors"] = [type(exc).__name__]
            if self._recording:
                result["artifacts"] = self._recording.paths
        finally:
            self._recording = previous

    def budget(self, phase=None):
        with self._state_lock:
            return self._budget(phase)

    def _budget(self, phase=None):
        phase = phase or self.phase
        usage = self.state["usage"][phase]
        spend = self.state['gemini_spend']
        return {"phase": phase, "calls_used": usage["calls"],
                "tokens_charged": usage["tokens"], "tokens_reported": usage["reported_tokens"],
                "unresolved_reserved_tokens": usage["tokens"] - usage["reported_tokens"],
                "reported_cost_usd": usage["cost_usd"],
                'session_cost_limit_usd': SESSION_LIMIT / NANODOLLARS,
                'session_cost_reported_usd': spend['reported'] / NANODOLLARS,
                'session_cost_reserved_usd': spend['reserved'] / NANODOLLARS,
                'session_cost_remaining_usd': (0 if spend['blocked'] else
                    max(0, SESSION_LIMIT-spend['reported']-spend['reserved'])) / NANODOLLARS,
                'session_cost_blocked': spend['blocked']}

    def charge(self, tokens, cost_reservation):
        with self._state_lock:
            if type(cost_reservation) is not int or cost_reservation < 1:
                raise ValueError('Invalid dollar reservation')
            spend = self.state['gemini_spend']
            if spend['blocked'] or spend['reported']+spend['reserved']+cost_reservation > SESSION_LIMIT:
                raise RuntimeError('Gemini session $10 budget exhausted')
            usage = self.state["usage"][self.phase]
            usage["calls"] += 1
            usage["tokens"] += tokens
            spend['reserved'] += cost_reservation
            self._save()

    def refund(self, tokens, reported, cost_reservation):
        with self._state_lock:
            usage = self.state["usage"][self.phase]
            usage["tokens"] -= tokens
            usage["reported_tokens"] += reported["total_tokens"]
            spend = self.state['gemini_spend']
            try:
                cost = nanodollars(reported.get('cost'))
            except ValueError:
                # Successful text without trustworthy billing is not a free call.
                self._save()
                return
            spend['reserved'] -= cost_reservation
            spend['reported'] += cost
            usage['cost_usd'] += cost / NANODOLLARS
            if cost > cost_reservation:
                spend['blocked'] = True  # Provider violated the enforced price/token bound.
            self._save()
            if spend['blocked']:
                raise RuntimeError('Provider cost exceeded reservation; Gemini session closed')

    def _success(self):
        active = self.state["active"] or {}
        if active.get("mode") == "interactive":
            self.state["interactive_success"] = {"episode": active["episode"], "at": time.time()}
            self._save()

    def _available(self):
        if self.state["formal_reserved"]:
            raise RuntimeError("The formal attempt has already been reserved; no retry")
        if self.state["active"]:
            raise RuntimeError("Another execution is active")

    def _has_successful_rehearsal(self, bundle_id, manifest):
        return any(qualifying_rehearsal(result, bundle_id, manifest)
                   for result in self.state['results'] + self.state.get('imported_rehearsals', []))

    def register(self, source, *, workspace_path='code/controller'):
        if self.state['formal_reserved'] or (self.state.get('active') or {}).get('container'):
            raise RuntimeError('Cannot register during execution or after formal')
        ident = uuid.uuid4().hex
        parent = self.config.root / "bundles"
        parent.mkdir(mode=0o700, exist_ok=True)
        manifest = snapshot(Path(source), parent / ident, image=self.config.image,
                            max_bytes=self.config.formal.artifact_bytes, entrypoint="controller.py",
                            workspace_path=workspace_path)
        if "controller.py" not in manifest["files"]:
            raise ValueError("Bundle requires controller.py")
        manifest["directory"] = ident
        self.state["bundles"][ident] = manifest
        self._save()
        return ident, manifest

    def _connect(self):
        self.backend = self.backend_factory()
        from services.mcp_contract import Contract
        self.gateway = Gateway(self.backend, self.gemini, audit=self.audit,
            charge=self.charge, refund=self.refund, budget=self.budget, on_success=self._success,
            publish_sequence=self._publish_sequence, contract=Contract.from_config(self.config, isolated=True))
        return self.gateway.acquire()

    def start_interactive(self):
        """Start exploration shared by direct agent calls and development Python."""
        self._available()
        index = self.state["exploration_started"]
        if index >= len(self.config.exploration_seeds):
            raise RuntimeError("Exploration episodes exhausted")
        self.phase = "development"
        self.state["exploration_started"] += 1
        self.state["active"] = {"mode": "interactive", "episode": index + 1}
        self._save()  # Reserve before environment creation.
        self._begin_recording("interactive", uuid.uuid4().hex, self.config.development.artifact_bytes)
        try:
            token = self._connect()
            self._interactive_token = token
            with deadline(240):
                value, _ = self.backend.start(formal=False, seed=self.config.exploration_seeds[index])
            self.audit({"event": "episode_start", "mode": "interactive", "result": sanitized(value)})
            self._boundary("start", token)
            return self.gateway, token
        except BaseException:
            self.finish_interactive(evaluate=False)
            raise

    def finish_interactive(self, *, evaluate=True):
        if not self.state.get("active") or self.state["active"]["mode"] != "interactive":
            raise RuntimeError("No interactive episode")
        result = dict(self.state["active"])
        if self.gateway and self.gateway.control_uncertain:
            # A partial failed action can alter state without advancing step_id.
            # Do not reuse/issue an evaluation of that uncertain scene on cleanup.
            evaluate = False
            result['control_uncertain'] = True
        try:
            if evaluate and self.backend:
                with deadline(60):
                    evaluation = self.backend.evaluate()
                result["evaluation"] = evaluation
                if complete(evaluation):
                    self._success()
        finally:
            try:
                try:
                    if evaluate and self._recording:
                        self._boundary("end", self._interactive_token)
                finally:
                    self._disconnect()
            finally:
                if self._recording:
                    result['api_budget'] = self.budget()
                    result.update(self._recording.finish_interactive(result))
                    self._recording = None
                self.state["active"] = None
                self.state["results"].append(result)
                self._save()
        return result

    def _disconnect(self):
        if self.gateway:
            self.gateway.revoke()
        try:
            if self.backend:
                self.backend.close()
        finally:
            self.gateway = self.backend = None
            self._interactive_token = None

    def run(self, bundle_id, *, formal):
        self._available()
        manifest = self.state["bundles"][bundle_id]
        bundle = self.config.root / "bundles" / manifest["directory"]
        verify(bundle, manifest)
        project_directory(manifest['workspace_path'])
        if formal and not self._has_successful_rehearsal(bundle_id, manifest):
            raise RuntimeError("A successful rehearsal of this exact bundle is required before formal submission")
        if not formal or self.config.formal_episodes == 1:
            return self._run_episode(bundle_id, formal=formal)
        # One submission reserves the entire immutable test schedule. Neither the
        # agent nor a restarted supervisor may repeat or resume a reserved batch.
        self.runner_factory(self.config.image, self.config.formal, gpu=self.config.controller_gpu).preflight()
        self.state['formal_reserved'] = True
        batch = {'bundle': bundle_id, 'sha256': manifest['sha256'], 'status': 'running',
                 'episode_count': self.config.formal_episodes, 'completed_episodes': 0,
                 'success_count': 0, 'unsuccessful_count': 0, 'error_count': 0,
                 'infrastructure_error_count': 0,
                 'success_rate': None, 'started': time.time(),
                 'report_path': 'runtime/autonomous_controller/formal_batch.json'}
        self.state['formal_batch'] = batch
        self.phase = 'formal'
        self._save()
        try:
            if self.config.formal_workers == 1:
                for index in range(self.config.formal_episodes):
                    self._record_formal(batch, self._run_episode(bundle_id, formal=True, formal_index=index))
            else:
                self._run_parallel_formal(bundle_id, batch)
            with self._state_lock:
                batch.update(status='completed', success_rate=batch['success_count']/batch['episode_count'])
        except BaseException:
            with self._state_lock:
                batch['status'] = 'interrupted'
            raise
        finally:
            with self._state_lock:
                batch['finished'] = time.time()
                self._save()
                # A small, agent-readable index; full per-call data stays in existing traces.
                formal = sorted((r for r in self.state['results'] if r.get('mode') == 'formal'),
                                key=lambda r: r['formal_episode_index'])
                report = json.loads(json.dumps({**batch, 'episodes': [{k: r[k] for k in (
                    'formal_episode_index', 'episode_id', 'formal_outcome', 'task_complete',
                    'reason', 'returncode', 'error', 'evaluation_failure', 'recording_errors',
                    'control_uncertain', 'worker_returncode', 'infrastructure_errors', 'timing', 'timeout_stage',
                    'finalization_timeout', 'artifacts') if k in r} for r in formal]}))
            self.published_root.mkdir(exist_ok=True)
            persist(self.published_root/'formal_batch.json', report)
            from services.robodojo.scoring import formal_scores
            # Private sibling of state.json: not mounted into agent containers.
            persist(self.config.root/'formal-scores.json', formal_scores(self.config.root, report))
        return {'mode': 'formal_batch', **batch, 'api_budget': self.budget()}

    def _record_formal(self, batch, result):
        """Count one finished episode; an interrupted one is kept but never counted."""
        with self._state_lock:
            if result.get('formal_outcome') != 'interrupted':
                result['formal_outcome'] = formal_outcome(result)
                batch['completed_episodes'] += 1
                batch[result['formal_outcome'] + '_count'] += 1
                if result.get('infrastructure_errors'):
                    # Overlaps task outcome counts; never subtract a verified success.
                    batch['infrastructure_error_count'] = batch.get('infrastructure_error_count', 0) + 1
            if result not in self.state['results']:
                self.state['results'].append(result)
            self._save()

    def _worker_result(self, bundle_id, index, returncode, *, interrupted=False):
        """The worker's own episode record is authoritative.

        A worker that started its episode but never finalized it counts as an error,
        like a sequential timeout. One that never started an episode is a systemic
        failure (spawn, preflight, bundle import), which interrupts the batch exactly
        as it would in the sequential loop. A later nonzero worker exit is
        recorded separately and cannot invalidate finalized task evidence.
        """
        from .formal_worker import worker_root
        manifest = self.state['bundles'][bundle_id]
        identity = {'mode': 'formal', 'bundle': bundle_id, 'sha256': manifest['sha256'],
                    'workspace_path': manifest['workspace_path'], 'formal_episode_index': index + 1,
                    'layout_id': self.config.formal_seed + index, 'eval_seed': self.config.formal_collection}
        try:
            state = json.loads((worker_root(self.config.root, index)/'state.json').read_text())
        except (OSError, ValueError):
            state = {'results': [], 'active': None}
        rows = [r for r in state.get('results', []) if r.get('mode') == 'formal']
        active = state.get('active') if (state.get('active') or {}).get('mode') == 'formal' else None
        if len(rows) > 1 or any(r.get('layout_id') != identity['layout_id'] or r.get('sha256') != identity['sha256']
                                or r.get('eval_seed') != identity['eval_seed'] for r in rows + [active or identity]):
            raise RuntimeError(f'Formal worker {index + 1} produced an inconsistent record')
        if rows:
            result = rows[0]
        elif active:
            result = {**active, 'reason': 'worker_lost', 'error_type': 'WorkerExit', 'task_complete': None}
            if active.get('container'):
                try:
                    remove_container(active['container'])
                except Exception as exc:
                    result['cleanup_error'] = type(exc).__name__
        elif interrupted:
            return None  # Never started; nothing ran on this layout.
        else:
            raise RuntimeError(f'Formal worker {index + 1} failed before starting its episode (exit {returncode})')
        result = {**result, **identity}
        note_worker_exit(result, returncode, finalized=bool(rows))
        if interrupted:
            result['formal_outcome'] = 'interrupted'
        return result

    def _recover_parallel_formal(self):
        """After a supervisor crash: stop this session's workers and keep their evidence."""
        batch = self.state['formal_batch']
        _stop_orphaned_workers(self.config.root, self.state.pop('formal_worker_pids', {}).values())
        recorded = {r.get('formal_episode_index') for r in self.state['results'] if r.get('mode') == 'formal'}
        for number in batch.pop('active_episodes', []):
            if number in recorded:
                continue
            try:
                result = self._worker_result(batch['bundle'], number - 1, None, interrupted=True)
            except Exception as exc:
                result = {'mode': 'formal', 'bundle': batch['bundle'], 'formal_episode_index': number,
                          'formal_outcome': 'interrupted', 'reason': 'worker_error', 'error_type': type(exc).__name__}
            if result:
                self.state['results'].append(result)

    def _run_parallel_formal(self, bundle_id, batch):
        """Run the reserved schedule on formal_workers concurrent episode processes.

        Every layout is started at most once. Gemini requests from all workers are
        brokered here, so the session ledger and key stay in this process.
        Episode numbers in batch state are 1-based, like formal_episode_index.
        """
        from .formal_worker import serve_broker, spawn
        factory = self.worker_factory or spawn
        pending = list(range(self.config.formal_episodes))
        running = {}  # index -> [worker, started, terminated_at]
        limit = self.config.formal.wall_seconds + WORKER_GRACE_SECONDS
        with self._state_lock:
            pids = self.state['formal_worker_pids'] = {}
            batch['active_episodes'] = []

        def track():
            batch['active_episodes'] = sorted(i + 1 for i in running)

        def untrack(index):
            with self._state_lock:
                running.pop(index, None)
                pids.pop(str(index + 1), None)
                track()
                self._save()

        def finish(index, code, *, interrupted=False):
            try:
                result = self._worker_result(bundle_id, index, code, interrupted=interrupted)
            except Exception:
                untrack(index)
                raise
            if result:
                self._record_formal(batch, result)
            # Keep the recovery pointer until the episode result is durable.
            untrack(index)

        try:
            while pending or running:
                while pending and len(running) < self.config.formal_workers:
                    index = pending.pop(0)
                    ours, theirs = socket.socketpair()
                    threading.Thread(target=serve_broker, args=(self, ours), daemon=True).start()
                    with self._state_lock:
                        running[index] = [None, time.monotonic(), None]
                        track()
                        self._save()  # Reserve before launch; a layout is never re-enqueued.
                    try:
                        running[index][0] = worker = factory(self.config.root, bundle_id, index, theirs)
                    except BaseException:
                        with self._state_lock:
                            running.pop(index)
                            track()
                        raise
                    finally:
                        theirs.close()
                    if getattr(worker, 'pid', None):
                        with self._state_lock:
                            pids[str(index + 1)] = worker.pid
                            self._save()
                for index, entry in list(running.items()):
                    worker, started, terminated = entry
                    code = worker.poll()
                    if code is None and terminated is None and time.monotonic() - started > limit:
                        worker.terminate()  # Once: the worker's cleanup must not be re-interrupted.
                        entry[2] = time.monotonic()
                    elif code is None and terminated is not None and time.monotonic() - terminated > WORKER_KILL_SECONDS:
                        worker.kill()
                        code = worker.wait(timeout=30)
                    if code is not None:
                        finish(index, code)
                if running:
                    time.sleep(1)
        finally:
            # A worker that finished before the interruption is a completed episode.
            for index, (worker, _, _) in list(running.items()):
                if worker.poll() is not None:
                    finish(index, worker.poll())
            for worker, _, _ in running.values():
                worker.terminate()
            # The launcher allows about 30 s after SIGTERM; wait on one shared deadline.
            # Workers still cleaning up stay in formal_worker_pids for restart recovery.
            end = time.monotonic() + 20
            for index, (worker, _, _) in list(running.items()):
                try:
                    code = worker.wait(timeout=max(0.1, end - time.monotonic()))
                except subprocess.TimeoutExpired:
                    continue
                try:
                    finish(index, code, interrupted=True)
                except Exception:
                    pass  # Preserve the original interruption.
            with self._state_lock:
                if not pids:
                    self.state.pop('formal_worker_pids', None)
                self._save()

    def _run_episode(self, bundle_id, *, formal, formal_index=0):
        manifest = self.state['bundles'][bundle_id]
        bundle = self.config.root/'bundles'/manifest['directory']
        verify(bundle, manifest)
        self.phase = "formal" if formal else "development"
        limits = getattr(self.config, self.phase)
        index = self.state["exploration_started"]
        if not formal and index >= len(self.config.exploration_seeds):
            raise RuntimeError("No exploration episode remains for rehearsal")
        if not formal:
            # A rehearsal must predict formal: same per-episode wall limit.
            limits = replace(limits, wall_seconds=self.config.formal.wall_seconds)
        runner = self.runner_factory(self.config.image, limits, gpu=self.config.controller_gpu)
        runner.preflight()  # No episode/attempt is consumed by local preflight.
        mode = "formal" if formal else "rehearsal"
        container = "robodojo-controller-" + uuid.uuid4().hex
        previous = self._begin_recording(mode, container.removeprefix("robodojo-controller-"), limits.artifact_bytes)
        record = {"mode": mode, "bundle": bundle_id, "sha256": manifest["sha256"],
                  "container": container, "started": time.time(), "workspace_path": manifest['workspace_path']}
        if formal:
            record['formal_episode_index'] = formal_index + 1
            record['layout_id'] = self.config.formal_seed + formal_index
            record['eval_seed'] = self.config.formal_collection
        self.state["active"] = record
        if formal:
            self.state["formal_reserved"] = True
        else:
            if self.exploration_pool is not None:
                reservation = self.exploration_pool.reserve_rehearsal()
                index = reservation['episode'] - 1
                self.state["exploration_started"] = reservation['episode']
            else:
                self.state["exploration_started"] += 1
        self._save()
        started = time.monotonic()
        timings = Timings()
        controller_mcp = Timings()
        timeout_stage = 'environment_setup'
        setup_complete = False
        result = dict(record)
        token = None
        try:
            with timings.measure('environment_setup_seconds'):
                with deadline(min(240, limits.wall_seconds)):
                    token = self._connect()
                    seed = self.config.formal_seed + formal_index if formal else self.config.exploration_seeds[index]
                    self.backend.start(formal=formal, seed=seed)
                setup_complete = True
                with deadline(max(0.000001, limits.wall_seconds - (time.monotonic() - started))):
                    self._boundary("start", token, propagate_timeout=True)
            remaining = limits.wall_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise ExecutionDeadline()
            timeout_stage = 'controller_execution'
            self.gateway.timings = controller_mcp
            try:
                with timings.measure('controller_run_seconds'):
                    result.update(runner.run(bundle, mode=mode, entrypoint="controller.py", gateway=self.gateway,
                        token=token, output=self.config.root / container, name=container,
                        frames=self._recording.frames, wall_seconds=remaining,
                        workspace_path=manifest['workspace_path']))
            finally:
                # Boundary observations belong to setup/evaluation, not the controller.
                self.gateway.timings = Timings()
        except ExecutionDeadline:
            result.update(reason="timeout", error_type="TimeoutError", timeout_seconds=limits.wall_seconds)
        except Exception as exc:
            result.update(reason="execution_error", error_type=type(exc).__name__,
                          error=public_failure(exc, operation=mode, stage='execution',
                              uncertain=bool(self.gateway and self.gateway.control_uncertain)))
        finally:
            if result.get("reason") == "finalization_timeout":
                # The episode already ended; only process exit overran. Evaluate it normally.
                result.update(reason="episode_ended", finalization_timeout=True)
            elif result.get("reason") == "timeout":
                result.update(reason="timeout", error_type="TimeoutError", timeout_seconds=limits.wall_seconds,
                              timeout_stage=timeout_stage)
            try:
                if setup_complete and result.get("reason") != "timeout":
                    timeout_stage = 'evaluation'
                    remaining = limits.wall_seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        raise ExecutionDeadline()
                    with timings.measure('evaluation_seconds'), deadline(remaining):
                        self._boundary("end", token, propagate_timeout=True)
                        if self.gateway:
                            self.gateway.revoke()
                        with deadline(60):
                            evaluation = self.backend.evaluate()
                    result["evaluation"] = evaluation
                    result["task_complete"] = evaluation.get("task_complete")
                else:
                    result["task_complete"] = None
            except ExecutionDeadline:
                result.update(reason="timeout", error_type="TimeoutError",
                              timeout_seconds=limits.wall_seconds, task_complete=None, timeout_stage=timeout_stage)
            except Exception as exc:
                result.update(evaluation_error=type(exc).__name__, task_complete=None,
                              evaluation_failure=public_failure(exc, operation=mode, stage='evaluation'))
            finally:
                try:
                    with timings.measure('environment_shutdown_seconds'):
                        self._disconnect()
                except Exception as exc:
                    # Evaluation already happened. Preserve its task result,
                    # disclose shutdown failure, and still propagate the fault.
                    result.setdefault('infrastructure_errors', []).append(
                        public_failure(exc, operation=mode, stage='environment_shutdown'))
                    raise
                finally:
                    total = time.monotonic() - started
                    run_seconds = timings.seconds.get('controller_run_seconds', 0.0)
                    execution = getattr(runner, 'execution_seconds', run_seconds)
                    mcp_seconds = controller_mcp.total()
                    result['timing'] = {
                        'schema': 'episode_wall_timing_v1', 'wall_limit_seconds': limits.wall_seconds,
                        'total_seconds': total,
                        'environment_setup_seconds': timings.seconds.get('environment_setup_seconds', 0.0),
                        'environment_step_seconds': sum(v for k, v in controller_mcp.seconds.items() if k in MUTATING),
                        'environment_other_mcp_seconds': sum(v for k, v in controller_mcp.seconds.items()
                            if k not in MUTATING and k not in {'gemini_generate', 'protocol_or_rejected'}),
                        'gemini_api_seconds': controller_mcp.seconds.get('gemini_generate', 0.0),
                        'protocol_seconds': controller_mcp.seconds.get('protocol_or_rejected', 0.0),
                        'controller_processing_seconds': max(0.0, execution - mcp_seconds),
                        'container_overhead_seconds': max(0.0, run_seconds - execution),
                        'evaluation_seconds': timings.seconds.get('evaluation_seconds', 0.0),
                        'environment_shutdown_seconds': timings.seconds.get('environment_shutdown_seconds', 0.0),
                        'supervisor_overhead_seconds': max(0.0, total - timings.total()),
                        'mcp_tools': controller_mcp.snapshot(),
                    }
                    self._end_recording(previous, bundle, self.config.root / container, result)
                    self.state["active"] = None
                    self.state["results"].append(result)
                    self._save()
                    if not formal and self.exploration_pool is not None:
                        self.exploration_pool.finish_reservation(reservation,
                            task_complete=result.get('task_complete', False))
        return result

    def close(self):
        try:
            if (self.state.get("active") or {}).get("mode") == "interactive":
                self.finish_interactive()
            else:
                self._disconnect()
        finally:
            self._lock.close()
