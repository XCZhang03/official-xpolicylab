"""Run a frozen agent bundle's main(ctx) inside the XPolicyLab policy contract.

The official evaluation loop owns the robot: it calls update_obs/get_action and
executes each returned action with take_action. The bundle instead calls
ctx.call(...). This bridge inverts control: a bundle's robodojo_step (joint) or
robodojo_step_ee (native EEF) call becomes the next get_action chunk, and returns
once the environment has executed it and delivered a fresh observation. Only
observation and step tools exist here; everything else a bundle needs must be
packaged inside the bundle itself.
"""
import base64
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import traceback

import numpy as np

# Official camera names -> the names agent bundles were written against.
CAMERAS = {'cam_head': 'cam_high', 'cam_left_wrist': 'cam_left_wrist', 'cam_right_wrist': 'cam_right_wrist'}
ARMS = ('left', 'right')
AVAILABLE = ('robodojo_observe', 'robodojo_status', 'robodojo_step', 'robodojo_step_ee', 'gemini_generate')


class EpisodeCancelled(Exception):
    """Raised inside the bundle thread when the official episode ends or resets."""


def _png(image):
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(np.asarray(image, dtype=np.uint8)[..., :3])).save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


def observation_reply(obs, step_id):
    """Official observation dict -> the MCP reply shape agent bundles parse."""
    state = obs.get('state', {})
    rows, poses = [], []
    for arm in ARMS:
        joints = np.asarray(state[f'{arm}_arm_joint_state'], dtype=np.float64).reshape(-1)
        gripper = np.asarray(state[f'{arm}_ee_joint_state'], dtype=np.float64).reshape(-1)
        rows.extend([*joints.tolist(), *gripper.tolist()])
        poses.append(np.asarray(state[f'{arm}_ee_pose'], dtype=np.float64).reshape(7))
    content, attachments = [None], []
    for source, name in CAMERAS.items():
        camera = obs.get('vision', {}).get(source)
        if camera is None or camera.get('color') is None:
            continue
        attachments.append({'kind': 'rgb', 'camera': name, 'content_index': len(content)})
        content.append({'type': 'image', 'mimeType': 'image/png', 'data': _png(camera['color'])})
    meta = {
        'step_id': step_id,
        'states': rows,
        'eef_positions': [p[:3].tolist() for p in poses],
        'eef_quaternions_wxyz': [p[3:].tolist() for p in poses],
        'instruction': obs.get('instruction'),
        'attachments': attachments,
        'observation_profile': 'official-rgb',
        # Official episodes end outside the policy; the bundle is cancelled instead.
        'transition': {'steps': []},
    }
    content[0] = {'type': 'text', 'text': json.dumps(meta)}
    return {'content': content}


def row_to_action(row):
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    if row.shape != (14,) or not np.isfinite(row).all():
        raise ValueError('robodojo_step expects finite 14-D absolute joint rows')
    if not (0 <= row[6] <= 1 and 0 <= row[13] <= 1):
        raise ValueError('Gripper openings must be in [0, 1]')
    return {'left_arm_joint_state': row[0:6], 'left_ee_joint_state': row[6:7],
            'right_arm_joint_state': row[7:13], 'right_ee_joint_state': row[13:14]}


def ee_row_to_action(row):
    """16-D [left x,y,z,qw,qx,qy,qz,gripper, right ...] -> the official EEF action dict.

    EvalEnv solves each arm's link6 pose with its own cuRobo IK (robot_manager.solve_ik).
    """
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    if row.shape != (16,) or not np.isfinite(row).all():
        raise ValueError('robodojo_step_ee expects finite 16-D rows [left pose7, gripper, right pose7, gripper]')
    if not (0 <= row[7] <= 1 and 0 <= row[15] <= 1):
        raise ValueError('Gripper openings must be in [0, 1]')
    for quaternion in (row[3:7], row[11:15]):
        if abs(float(np.linalg.norm(quaternion)) - 1) > 1e-3:
            raise ValueError('EEF quaternions must be unit wxyz')
    return {'left_ee_pose': row[0:7], 'left_ee_joint_state': row[7:8],
            'right_ee_pose': row[8:15], 'right_ee_joint_state': row[15:16]}


def hold_action(obs):
    """Keep the measured arm pose and the current gripper command for one step."""
    state = obs['state']
    return {f'{arm}_{kind}': np.asarray(state[f'{arm}_{kind}'], dtype=np.float32).reshape(-1)
            for arm in ARMS for kind in ('arm_joint_state', 'ee_joint_state')}


class OfficialContext:
    """The ctx passed to main(ctx): same call() shape as the harness runtime."""

    def __init__(self, bridge):
        self._bridge = bridge
        self.output_dir = bridge.output_dir

    def call(self, tool, **arguments):
        return self._bridge.handle(tool, arguments)


class Bridge:
    def __init__(self, main, *, action_wait_s=100.0, output_dir=None, log=print, gemini=None):
        self.main, self.action_wait_s, self.log = main, action_wait_s, log
        # gemini(arguments) -> MCP-style reply, or None when no model API is configured.
        self.gemini = gemini
        self.output_dir = Path(output_dir or tempfile.mkdtemp(prefix='agent-bundle-'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Condition()
        self.obs, self.step_id = None, 0
        self.request = None          # rows the bundle wants executed next
        self.in_flight = False       # a chunk was handed to the environment
        self.done = False
        self.cancelled = False
        self.error = None
        self.thread = None

    # ---- policy side (server thread) ----
    def update_obs(self, obs):
        with self.lock:
            self.obs = obs
            self.lock.notify_all()

    def get_action(self):
        with self.lock:
            if self.obs is None:
                raise RuntimeError('get_action before any observation')
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name='agent-bundle', daemon=True)
                self.thread.start()
            elif self.in_flight:
                # The previous chunk has executed; the latest obs follows it.
                self.in_flight = False
                self.lock.notify_all()
            self.lock.wait_for(lambda: self.request is not None or self.done, timeout=self.action_wait_s)
            if self.request is None:
                if not self.done:
                    self.log('[AgentBundle] bundle still computing; holding pose for one step')
                return [hold_action(self.obs)]
            actions, self.request = self.request, None
            self.in_flight = True
            self.step_id += len(actions)
            return actions

    def cancel(self):
        with self.lock:
            self.cancelled = True
            self.lock.notify_all()

    # ---- bundle side (bundle thread) ----
    def handle(self, tool, arguments):
        if tool == 'gemini_generate':
            # Runs in the bundle thread without the lock: the environment keeps its
            # pose (a hold step per get_action wait) while the model answers.
            self._check()
            if self.gemini is None:
                raise RuntimeError('MCP tool failed: gemini_generate is not configured on this policy server '
                                   '(set its key; see official/xpolicylab/README.md)')
            try:
                return self.gemini(dict(arguments))
            except EpisodeCancelled:
                raise
            except Exception as exc:
                # The same error shape a harness call raises, so bundles handle one form.
                raise RuntimeError('MCP tool failed: ' + json.dumps({'error': {
                    'type': type(exc).__name__, 'operation': 'gemini_generate', 'reason': str(exc)[:2000],
                    'no_action_executed': True}})) from None
        if tool in ('robodojo_observe', 'robodojo_status'):
            if arguments:
                raise ValueError(f'{tool} takes no arguments in official mode')
            with self.lock:
                self._check()
                return observation_reply(self.obs, self.step_id)
        if tool in ('robodojo_step', 'robodojo_step_ee'):
            convert = row_to_action if tool == 'robodojo_step' else ee_row_to_action
            if set(arguments) != {'actions'} or not isinstance(arguments['actions'], list) or not arguments['actions']:
                raise ValueError(f'{tool} expects only a nonempty actions list')
            actions = [convert(r) for r in arguments['actions']]  # Validate all before any motion.
            with self.lock:
                self._check()
                self.request = actions
                self.lock.notify_all()
                self.lock.wait_for(lambda: (self.request is None and not self.in_flight) or self.cancelled)
                self._check()
                return observation_reply(self.obs, self.step_id)
        raise PermissionError(f'{tool} is unavailable in official mode; available: {", ".join(AVAILABLE)}. '
                              'Use robodojo_toolkit or bundle code for planning and pose math.')

    def _check(self):
        if self.cancelled:
            raise EpisodeCancelled('Official episode ended')

    def _run(self):
        try:
            self.main(OfficialContext(self))
        except EpisodeCancelled:
            pass
        except BaseException as exc:
            self.error = exc
            self.log('[AgentBundle] bundle raised: ' + ''.join(traceback.format_exception(exc))[-2000:])
        finally:
            with self.lock:
                self.done = True
                self.lock.notify_all()


def load_main(bundle_dir):
    """Import controller.py:main from a frozen bundle directory."""
    bundle_dir = Path(bundle_dir).resolve()
    controller = bundle_dir / 'controller.py'
    if not controller.is_file():
        raise FileNotFoundError(f'No controller.py in bundle {bundle_dir}')
    sys.path.insert(0, str(bundle_dir))  # Bundle-local modules such as perception.py.
    spec = importlib.util.spec_from_file_location('agent_bundle_controller', controller)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'main', None)):
        raise AttributeError('Bundle controller.py must define main(ctx)')
    return module.main
