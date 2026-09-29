"""Stdlib MCP client: development socket or isolated evaluation stdio.

Ordinary Python connects to the same external endpoint as the agent. Full evaluations
receive a private channel with robot calls only, without lifecycle authority, and
run the bundle through the official XPolicyLab adapter (run_official). Their Python
prints are log events; native/subprocess output goes to stderr.
"""
import json
import os
from pathlib import Path
import sys
import threading
import socket

_channel_lock = threading.RLock()


class PrintLog:
    """Keep Python prints separate from both MCP results and stderr."""
    encoding = "utf-8"

    def __init__(self, channel):
        self.channel = channel

    def write(self, text):
        with _channel_lock:
            for offset in range(0, len(text), 8192):
                self.channel.write(json.dumps({"jsonrpc": "2.0", "method": "logs/write",
                    "params": {"text": text[offset:offset + 8192]}}) + "\n")
            self.channel.flush()
        return len(text)

    def flush(self):
        with _channel_lock:
            self.channel.flush()

    def isatty(self):
        return False

    def fileno(self):
        return 1  # Native/subprocess writes are safely captured on stderr.


class Context:
    def __init__(self, reader=None, writer=None):
        """Connect ordinary development Python to the shared remote MCP endpoint.

        Full evaluations supply their private stdio transport instead.
        """
        self._socket = None
        self._request_lock = threading.RLock()
        # Same attribute as the official XPolicyLab adapter's context: write outputs here.
        self.output_dir = Path('/workspace/output')
        if reader is None and writer is None:
            self._socket = socket.socket(socket.AF_UNIX)
            try:
                self._socket.connect('/run/agent-relay/mcp.sock')
                reader, writer = self._socket.makefile('r'), self._socket.makefile('w')
            except BaseException:
                self._socket.close()
                raise
        self._reader, self._writer, self._sequence = reader, writer, 0
        if self._socket:
            try:
                self.request('initialize', {'protocolVersion': '2025-06-18', 'capabilities': {},
                             'clientInfo': {'name': 'python-controller', 'version': '2'}})
            except BaseException:
                self.close()
                raise

    def close(self):
        if self._socket:
            self._writer.close()
            self._reader.close()
            self._socket.close()
            self._socket = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def request(self, method, params):
        with self._request_lock:
            return self._request(method, params)

    def env(self, env_id):
        """Bind development robot calls to one operator-created exploration slot."""
        if type(env_id) is not int or env_id < 0:
            raise ValueError('env_id must be a nonnegative integer')
        return Environment(self, env_id)

    def _request(self, method, params):
        self._sequence += 1
        request = {"jsonrpc": "2.0", "id": self._sequence, "method": method, "params": params}
        self._writer.write(json.dumps(request, allow_nan=False) + "\n")
        self._writer.flush()
        line = self._reader.readline(32 * 1024 * 1024)
        response = json.loads(line)
        if response.get("id") != self._sequence:
            raise RuntimeError("MCP request failed: identity mismatch")
        if "error" in response:
            raise RuntimeError("MCP tool failed: " + json.dumps({"error": response["error"]}, allow_nan=False))
        return response["result"]

    def call(self, tool, **arguments):
        result = self.request("tools/call", {"name": tool, "arguments": arguments})
        if result.get("isError"):
            # Same detail as an isolated run's JSON-RPC error, so controllers see
            # identical failure text in development and rehearsal/formal.
            text = next((b.get("text", "") for b in result.get("content", []) if b.get("type") == "text"), "")
            raise RuntimeError("MCP tool failed: " + text)
        return result


class Environment:
    def __init__(self, context, env_id):
        self.context, self.env_id = context, env_id

    def call(self, tool, **arguments):
        # Pure math, the model API and the session lifecycle status are not robot-slot operations.
        if tool not in {'robodojo_pose_math', 'exploration_status', 'gemini_generate'}:
            if tool not in {'start_episode', 'finish', 'evaluate'} and not tool.startswith('robodojo_'):
                raise ValueError('Use the session Context for registration, rehearsal and submission')
            if 'env_id' in arguments:
                raise ValueError('Environment is already bound')
            arguments['env_id'] = self.env_id
        return self.context.call(tool, **arguments)


def _ended(value):
    if isinstance(value, dict):
        return value.get('episode_ended') is True or any(_ended(v) for v in value.values())
    return isinstance(value, list) and any(_ended(v) for v in value)


def episode_ended(reply):
    """True when a robodojo_* reply reports native termination or truncation."""
    return any(_ended(json.loads(block['text'])) for block in reply.get('content', [])
               if block.get('type') == 'text')


def official_observation(reply):
    """An official-profile robodojo_* reply as the observation XPolicyLab gives a policy."""
    import base64
    import io
    import numpy as np
    from PIL import Image
    content = reply['content']
    meta = json.loads(next(b['text'] for b in content if b['type'] == 'text'))
    states = np.asarray(meta['states'], dtype=np.float64)
    state = {}
    for index, arm in enumerate(('left', 'right')):
        offset = 7 * index
        state[f'{arm}_arm_joint_state'] = states[offset:offset + 6]
        state[f'{arm}_ee_joint_state'] = states[offset + 6:offset + 7]
        state[f'{arm}_ee_pose'] = np.r_[meta['eef_positions'][index], meta['eef_quaternions_wxyz'][index]]
    official = {camera: source for source, camera in (('cam_head', 'cam_high'),
                ('cam_left_wrist', 'cam_left_wrist'), ('cam_right_wrist', 'cam_right_wrist'))}
    vision = {}
    for attachment in meta.get('attachments', []):
        if attachment.get('kind', 'rgb') == 'rgb' and attachment.get('camera') in official:
            data = base64.b64decode(content[attachment['content_index']]['data'])
            vision[official[attachment['camera']]] = {
                'color': np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))}
    return {'state': state, 'vision': vision, 'instruction': meta.get('instruction')}


def _tool_rows(actions):
    """Official action dicts -> (tool, rows) calls of <= 50 rows, in execution order.

    Joint dicts go to robodojo_step, native EEF dicts (``*_ee_pose``) to robodojo_step_ee.
    """
    calls = []
    for action in actions:
        if 'left_ee_pose' in action:
            tool, keys = 'robodojo_step_ee', ('ee_pose', 'ee_joint_state')
        else:
            tool, keys = 'robodojo_step', ('arm_joint_state', 'ee_joint_state')
        row = [float(v) for arm in ('left', 'right') for key in keys for v in action[f'{arm}_{key}']]
        if not calls or calls[-1][0] != tool or len(calls[-1][1]) == 50:
            calls.append((tool, []))
        calls[-1][1].append(row)
    return calls


def run_official(context, project, *, action_wait_s=None, log=print):
    """Run project/controller.py:main exactly as the official XPolicyLab evaluation does.

    The adapter's Bridge owns the bundle: main(ctx) runs in a thread with only
    robodojo_observe/status/step/step_ee, get_action hands each step chunk to the
    environment, the pose is held while the bundle computes longer than action_wait_s
    and after main returns, and the bundle is cancelled when the episode ends.
    Isolated rehearsal and formal runs use this; development Python may too.
    Returns the bridge (``bridge.error`` holds an exception raised by main).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bundle_bridge import Bridge, load_main
    if action_wait_s is None:
        action_wait_s = float(os.environ.get('ROBODOJO_ACTION_WAIT_S', '90'))
    # gemini_generate is forwarded to the harness MCP, which holds the key and the
    # session budget; officially the adapter serves it on a self-hosted policy server.
    bridge = Bridge(load_main(project), action_wait_s=action_wait_s,
                    output_dir=getattr(context, 'output_dir', None), log=log,
                    gemini=lambda arguments: context.call('gemini_generate', **arguments))
    reply = context.call('robodojo_observe')
    try:
        while not episode_ended(reply):
            bridge.update_obs(official_observation(reply))
            for tool, rows in _tool_rows(bridge.get_action()):
                reply = context.call(tool, actions=rows)
                if episode_ended(reply):
                    break
    finally:
        bridge.cancel()
        if bridge.thread is not None:
            bridge.thread.join(timeout=10)
    return bridge


def main():
    mode, project, entrypoint = sys.argv[1:]
    # Keep the protocol on a private non-inherited descriptor. Native libraries
    # and subprocesses writing fd 1 cannot corrupt MCP: their output goes to the
    # captured stderr stream instead.
    channel = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), 1)
    sys.stdout = PrintLog(channel)
    os.chdir("/workspace")
    if Path(entrypoint).name != 'controller.py':
        raise SystemExit('Isolated runs execute controller.py:main through the official adapter')
    context = Context(sys.stdin, channel)
    context.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "python-controller", "version": "2"}})
    bridge = run_official(context, project, log=lambda text: print(text, file=sys.stderr, flush=True))
    if bridge.error is not None:
        # As officially, the episode still ran to its end with the pose held; the nonzero
        # exit keeps a raising bundle from qualifying as a successful rehearsal.
        print(f'[AgentBundle] main(ctx) raised {type(bridge.error).__name__}; see the traceback above',
              file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
