"""Host-owned robot/API MCP for a Codex agent running entirely in Docker."""
from contextlib import contextmanager
import argparse
import base64
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import selectors
import socket
import subprocess
import sys
import uuid

from .backend import NativeBackend
from services.mcp_contract import Contract, InvalidArguments, ToolUnavailable
from .errors import public_failure
from .config import Configuration
from services.robodojo.timeouts import task_timeouts
from .gateway import Gateway, MAX_REQUEST_BYTES, ROBOT_TOOLS, complete, sanitized
from .gemini import GeminiRouter, load_key
from .recording import read_regular
from .sandbox import deadline
from .supervisor import Supervisor
from .storage import verify, project_directory


IMAGE_REPLY_GUIDANCE = (' Returns text/image content blocks. In functions.exec, iterate reply.content: '
                        'use text(block.text) for text blocks and image(block) for selected image blocks. '
                        'Never text(reply) or stringify image data.')


def tool(name, description, properties=None, required=(), *, images=False):
    if images:
        description += IMAGE_REPLY_GUIDANCE
    return {'name': name, 'description': description, 'inputSchema': {'type': 'object',
            'properties': properties or {}, 'required': list(required), 'additionalProperties': False}}


STRING = {'type': 'string'}
SOURCE = {'source': {**STRING, 'description': 'Project under /workspace/code, e.g. code/reach. Isolated runs preserve this path.'}}
DEFINITIONS = [
    tool('exploration_status', 'Session lifecycle (not the robot): task, manual success evidence, episode state, episode timeout, remaining exploration episodes, registered bundles with rehearsal_qualified, workspace storage, and the Gemini API budget for the current and formal phases.'),
    tool('start_episode', 'Start the next distinct-seed exploration episode; closes the previous one. No seed arguments.', images=True),
    tool('evaluate', 'Trusted manual exploration evaluation. Success is guidance, not a programming gate.'),
    tool('finish', 'Close the current exploration episode. Does not submit formal.', images=True),
    tool('register', 'Freeze a controller.py bundle without executing it.', SOURCE, ['source']),
    tool('rehearse', 'Run the whole registered controller in a fresh exploration episode under the same wall limit as each formal episode (status.episode_timeout_seconds, startup included). Consumes one exploration episode; no cumulative rehearsal-time cap. Returns wall-time breakdown in timing; timeouts identify timeout_stage.', {'bundle': STRING}, ['bundle'], images=True),
    tool('submit', 'One final submission: run the exact successfully rehearsed bundle over the operator-configured formal batch (50 episodes for new sessions). Fresh container/environment per episode; no agent intervention, code changes or retries. Returns success_count / episode_count and success_rate; invalid or unconfirmed episode results count as non-success. Later worker crashes are reported separately without changing finalized task outcomes. Gemini budget is shared across the batch.', {'bundle': STRING}, ['bundle']),
]


class ResearchFrontend:
    def __init__(self, supervisor, workspace, *, agent_container=None):
        self.s = supervisor
        self.contract = Contract.from_config(supervisor.config)
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
        if agent_container is not None and not re.fullmatch(r'robodojo-agent-[a-f0-9]{32}', agent_container):
            raise ValueError('Invalid agent container identity')
        self.agent_container = agent_container
        self.robot_tools = self.contract.robot_definitions()
        class Inventory:
            """Episode-independent tools: Gemini and pure pose math (diagnostic profiles), no simulator."""
            def tools(_):
                return [t for t in self.robot_tools if t['name'] == 'robodojo_pose_math']

            def call(_, name, arguments):
                if name != 'robodojo_pose_math':
                    raise RuntimeError('No interactive episode')
                from services.robodojo.mcp_server import _jsonable
                from services.robodojo.pose_math import pose_math
                return _jsonable(pose_math(Path(__file__).resolve().parents[2], arguments)), []
        self.api = Gateway(Inventory(), self.s.gemini, audit=self.s.audit, charge=self.s.charge,
                           refund=self.s.refund, budget=self.s.budget,
                           contract=Contract.from_config(self.s.config, isolated=True))
        self.api_token = self.api.acquire()

    def tools(self):
        return self.contract.select(self.robot_tools + DEFINITIONS + [t for t in self.api.definitions() if t['name'] == 'gemini_generate'])

    @contextmanager
    def frozen(self):
        # The native shell can leave background writers. Freeze the whole agent
        # while copying/executing its submission, not just the foreground shell.
        if self.agent_container:
            subprocess.run(['docker', 'pause', self.agent_container], check=True, capture_output=True, timeout=30)
        try:
            yield
        finally:
            if self.agent_container:
                subprocess.run(['docker', 'unpause', self.agent_container], check=True, capture_output=True, timeout=30)

    def source(self, name):
        relative = project_directory(name).relative_to('/workspace')
        path = self.workspace.joinpath(*relative.parts)
        # snapshot() pins every ancestor with O_NOFOLLOW as well.
        if not path.is_dir() or path.is_symlink():
            raise ValueError('Invalid source directory')
        return path

    def status(self):
        state = self.s.state
        storage = os.statvfs(self.workspace)
        demonstration_path = self.s.config.root/'demonstration-context.json'
        demonstration = json.loads(demonstration_path.read_text()) if demonstration_path.is_file() else {
            'kind': self.s.config.demonstration_context, 'manifest_path': None, 'image_count': 0}
        return {'contract': self.contract.manifest(), 'task': self.s.config.task, 'manual_success_confirmed': bool(state['interactive_success']),
                'demonstration_context': demonstration,
                'workspace_storage': {'limit_bytes': self.s.config.workspace_mb*1024**2,
                    'capacity_bytes': storage.f_blocks*storage.f_frsize,
                    'used_bytes': (storage.f_blocks-storage.f_bfree)*storage.f_frsize,
                    'available_bytes': storage.f_bavail*storage.f_frsize},
                'active_episode': (state.get('active') or {}).get('episode'),
                'episode_ended_or_uncertain': bool(self.s.gateway and self.s.gateway.terminal),
                'exploration_remaining': len(self.s.config.exploration_seeds)-state['exploration_started'],
                'formal_reserved': state['formal_reserved'],
                'formal_episode_count': self.s.config.formal_episodes,
                'formal_batch': state.get('formal_batch'),
                # Rehearsal and each formal episode share this wall limit, startup included.
                'episode_timeout_seconds': self.s.config.formal.wall_seconds,
                'session_timeout_seconds': task_timeouts(self.s.config.task)['session_seconds'],
                'bundles': {ident: {'project_directory': str(project_directory(m['workspace_path'])),
                                    'sha256': m['sha256'],
                                    'rehearsal_qualified': self.s._has_successful_rehearsal(ident, m)}
                            for ident, m in state['bundles'].items()},
                'artifacts': self.s._recording.paths if self.s._recording else None,
                'api_budget': self.s.budget(),
                # Formal usage counters and the same shared session dollar balance.
                'formal_api_budget': self.s.budget('formal')}

    @staticmethod
    def public_result(result):
        return {k: result[k] for k in ('mode', 'reason', 'returncode', 'task_complete', 'episode',
            'episode_id', 'start_step_id', 'end_step_id', 'artifacts', 'recording_errors',
            'api_budget', 'control_uncertain', 'episode_ended_or_uncertain', 'sha256', 'error_type',
            'error', 'evaluation_error', 'evaluation_failure', 'timeout_seconds', 'timeout_stage', 'finalization_timeout', 'timing', 'output_error',
            'bundle', 'status', 'episode_count', 'completed_episodes', 'success_count',
            'unsuccessful_count', 'error_count', 'infrastructure_error_count', 'infrastructure_errors',
            'worker_returncode', 'success_rate', 'formal_episode_index', 'report_path',
            'formal_wall_seconds') if k in result}

    def robot(self, name, arguments):
        if self.s.state.get('formal_reserved'):
            raise RuntimeError('Formal attempt is closed to interactive control')
        gateway, token = self.s.gateway, self.s._interactive_token
        # Pose math is pure geometry: available with or without a live episode.
        if name == 'robodojo_pose_math' or (name == 'gemini_generate' and gateway is None):
            gateway, token = self.api, self.api_token
        if gateway is None:
            raise RuntimeError('No interactive episode')
        response = gateway.handle(token, {'jsonrpc': '2.0', 'id': uuid.uuid4().hex,
            'method': 'tools/call', 'params': {'name': name, 'arguments': arguments}})
        if 'error' in response:
            return {'isError': True, 'content': [{'type': 'text', 'text': json.dumps({
                'error': response['error'],
                'artifacts': self.s._recording.paths if self.s._recording else None,
            })}]}
        return response['result']

    def call(self, name, args):
        if name in ROBOT_TOOLS or name == 'gemini_generate':
            self.contract.require_robot(name, args)
            return self.robot(name, args)
        definition = next((t for t in DEFINITIONS if t['name'] == name), None)
        if definition is None:
            raise ToolUnavailable('Unknown tool')
        schema = definition['inputSchema']
        if not isinstance(args, dict) or set(args)-set(schema['properties']) or set(schema['required'])-set(args):
            raise InvalidArguments('Invalid tool arguments')
        value = None
        if name == 'exploration_status':
            value = self.status()
        elif name == 'start_episode':
            if self.s.state['formal_reserved'] or self.s.state['exploration_started'] >= len(self.s.config.exploration_seeds):
                raise RuntimeError('No exploration episode remains')
            if self.s.state['active']:
                self.s.finish_interactive()
            self.s.start_interactive()
            return self.robot('robodojo_observe', {})
        elif name == 'evaluate':
            if not self.s.gateway or self.s.state['formal_reserved'] or self.s.gateway.control_uncertain:
                raise RuntimeError('No trustworthy interactive episode')
            with deadline(60):
                evaluation = self.s.backend.evaluate()
            if complete(evaluation):
                self.s._success()
            value = {'evaluation': sanitized(evaluation), **self.status()}
        elif name == 'finish':
            closed = None
            if self.s.state.get('active'):
                closed = self.public_result(self.s.finish_interactive())
            value = self.status()
            if closed:
                value['closed_episode'] = closed
                value['artifacts'] = closed.get('artifacts')
        elif name == 'register':
            with self.frozen():
                bundle, manifest = self.s.register(self.source(args['source']), workspace_path=args['source'])
            value = {'bundle': bundle, 'sha256': manifest['sha256'],
                     'project_directory': str(project_directory(manifest['workspace_path']))}
        elif name in {'rehearse', 'submit'}:
            manifest = self.s.state['bundles'][args['bundle']]
            verify(self.s.config.root/'bundles'/manifest['directory'], manifest)
            formal = name == 'submit'
            if formal and not self.s._has_successful_rehearsal(args['bundle'], manifest):
                raise RuntimeError('Exact successful rehearsal required')
            with self.frozen():
                if self.s.state.get('active'):
                    self.s.finish_interactive()
                value = self.public_result(self.s.run(args['bundle'], formal=formal))
        return self.reply(value)

    def reply(self, value):
        """Small summary plus selected final frames, shared by both experiment modes."""
        result = {'content': [{'type': 'text', 'text': json.dumps(value, allow_nan=False)}]}
        if isinstance(value, dict) and value.get('reason') == 'timeout':
            result['isError'] = True
        artifacts = value.get('artifacts') if isinstance(value, dict) else None
        for image in (artifacts.get('final_images', []) if isinstance(artifacts, dict) else []):
            relative = PurePosixPath(image['path']).relative_to('runtime/autonomous_controller')
            if '..' in relative.parts:
                raise ValueError('Invalid image path')
            raw = read_regular(self.s.config.root/'published'/str(relative), 16*1024*1024)
            result['content'].append({'type': 'image', 'mimeType': 'image/png', 'data': base64.b64encode(raw).decode()})
        return result

    def close(self):
        self.api.revoke()
        self.s.close()

    def handle(self, request):
        if not isinstance(request, dict):
            return {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32600, 'message': 'Invalid request'}}
        ident = request.get('id')
        try:
            if request.get('jsonrpc') != '2.0':
                raise ValueError('Invalid JSON-RPC')
            method = request.get('method')
            if method == 'initialize':
                result = {'protocolVersion': '2025-06-18', 'capabilities': {'tools': {}},
                          'serverInfo': {'name': self.contract.mode, 'version': '3.0'}}
            elif method == 'notifications/initialized':
                return None
            elif method == 'tools/list':
                result = {'tools': self.tools()}
            elif method == 'tools/call':
                params = request['params']
                try:
                    result = self.call(params['name'], params.get('arguments', {}))
                except Exception as exc:
                    failure = public_failure(exc, operation=params.get('name'), stage='lifecycle',
                                             uncertain=bool(self.s.gateway and self.s.gateway.control_uncertain))
                    result = {'isError': True, 'content': [{'type': 'text',
                        'text': json.dumps({'error': failure})}]}
            else:
                raise PermissionError('Unsupported method')
            return {'jsonrpc': '2.0', 'id': ident, 'result': result}
        except Exception as exc:
            failure = public_failure(exc)
            return {'jsonrpc': '2.0', 'id': ident, 'error': {
                'code': -32000, 'message': failure['reason'], 'data': failure}}

    def serve_socket(self, listener):
        """Multiplex clients; execute every request on ONE main-thread dispatcher.

        An idle Codex connection must not block a Python client. Evaluation calls
        occupy this dispatcher, so development cannot act on their private episode.
        """
        listener.setblocking(False)
        with selectors.DefaultSelector() as poll:
            poll.register(listener, selectors.EVENT_READ)
            buffers = {}
            try:
                while True:
                    for key, _ in poll.select():
                        connection = key.fileobj
                        if connection is listener:
                            try:
                                client, _ = listener.accept()
                            except BlockingIOError:
                                continue
                            if len(buffers) >= 16:
                                client.close()
                                continue
                            client.setblocking(False)
                            buffers[client] = bytearray()
                            poll.register(client, selectors.EVENT_READ)
                            continue
                        try:
                            block = connection.recv(65536)
                            if not block:
                                raise ConnectionError('Client closed')
                            pending = buffers[connection]
                            pending.extend(block)
                            if len(pending) > MAX_REQUEST_BYTES:
                                raise ConnectionError('Request exceeds wire limit')
                            while b'\n' in pending:
                                line, _, tail = pending.partition(b'\n')
                                pending[:] = tail
                                try:
                                    request = json.loads(line)
                                except (ValueError, UnicodeError):
                                    response = {'jsonrpc': '2.0', 'id': None,
                                        'error': {'code': -32700, 'message': 'Invalid request'}}
                                else:
                                    response = self.handle(request)
                                if response is not None:
                                    connection.settimeout(5)
                                    connection.sendall(json.dumps(response, allow_nan=False).encode() + b'\n')
                                    connection.setblocking(False)
                        except (ConnectionError, OSError):
                            poll.unregister(connection)
                            buffers.pop(connection, None)
                            connection.close()
            finally:
                for client in buffers:
                    client.close()

    def serve(self, reader, writer):
        while line := reader.readline(MAX_REQUEST_BYTES + 1):
            if len(line.encode('utf-8')) > MAX_REQUEST_BYTES:
                writer.write(json.dumps({'jsonrpc': '2.0', 'id': None,
                    'error': {'code': -32600, 'message': 'Request exceeds wire limit'}}) + '\n')
                writer.flush()
                return  # Do not interpret the remaining fragment as another request.
            try:
                response = self.handle(json.loads(line))
            except Exception:
                response = {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'Invalid request'}}
            if response is not None:
                writer.write(json.dumps(response, allow_nan=False)+'\n')
                writer.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--socket', type=Path)
    parser.add_argument('--agent-container')
    parser.add_argument('--key-file', type=Path)
    args = parser.parse_args()
    config = Configuration.from_dict(json.loads(args.config.read_text()))
    contract = Contract.from_config(config)
    if contract.environments > 1 and not args.socket:
        raise ValueError('Parallel exploration requires the shared socket endpoint')
    key = load_key(args.key_file) if contract.gemini else None
    supervisor = Supervisor(config, lambda: NativeBackend(config), GeminiRouter(key) if key else None)
    frontend_class = ResearchFrontend
    if config.exploration_envs > 1:
        from services.exploration.research import ParallelResearchFrontend
        frontend_class = ParallelResearchFrontend
    frontend = frontend_class(supervisor, args.workspace, agent_container=args.agent_container)
    def stop(*_):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    try:
        if args.socket:
            with socket.socket(socket.AF_UNIX) as listener:
                listener.bind(str(args.socket))
                os.chmod(args.socket, 0o600)
                listener.listen(16)
                frontend.serve_socket(listener)
        else:
            frontend.serve(sys.stdin, sys.stdout)
    finally:
        frontend.close()


if __name__ == '__main__':
    main()
