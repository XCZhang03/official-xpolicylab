"""XPolicyLab adapter that runs a frozen RoboDojo agent bundle (controller.py:main).

The bundle runs in this policy server. On our hosted agent API it runs as a worker
behind endpoint/agent_api.py, which authenticates with agentbundle_hello.
"""
import hmac
import os
from pathlib import Path

from XPolicyLab.model_template import ModelTemplate

from .bundle_bridge import Bridge, load_main
from .gemini_router import official_service

POLICY_DIR = Path(__file__).resolve().parent


def resolve_bundle(cfg):
    """Pick the frozen bundle for the task this policy server was started for.

    Official sweeps start one policy server per task with ``task_name`` in the
    config, so one checkpoint may hold one bundle per task:
    ``<checkpoint>/<task_name>/controller.py``. A checkpoint holding a single
    ``controller.py`` serves every task. ``bundle_path`` in deploy.yml overrides.
    """
    task = str(cfg.get('task_name') or '')
    if cfg.get('bundle_path'):
        root = Path(cfg['bundle_path'])
    else:
        name = str(cfg['ckpt_name'])
        root = Path(name) if Path(name).is_absolute() else POLICY_DIR / name
        if not root.is_dir():
            root = POLICY_DIR / 'checkpoints' / '-'.join(str(cfg[k]) for k in
                ('bench_name', 'ckpt_name', 'env_cfg_type', 'action_type', 'seed'))
    root = root if root.is_absolute() else POLICY_DIR / root
    for candidate in (root / task, root):
        if task and candidate == root / task and not candidate.is_dir():
            continue
        if (candidate / 'controller.py').is_file():
            return candidate
    tasks = sorted(p.name for p in root.iterdir() if (p / 'controller.py').is_file()) if root.is_dir() else []
    raise FileNotFoundError(f'No bundle for task {task!r} under {root}; available: {tasks}')


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        # The action type is only a label here: EvalEnv infers joint or EEF from each
        # action's keys, and bundles may send both (robodojo_step / robodojo_step_ee).
        if model_cfg.get('action_type') not in ('joint', 'ee') or model_cfg.get('env_cfg_type') != 'arx_x5':
            raise ValueError('AgentBundle supports env_cfg_type=arx_x5 with action_type joint or ee')
        # As a worker of our agent API, require the endpoint's agentbundle_hello token.
        self._token = os.environ.get(model_cfg.get('server_token_env') or 'AGENTBUNDLE_SERVER_TOKEN', '')
        self._authenticated = not self._token
        self.main = load_main(resolve_bundle(model_cfg))
        if model_cfg.get('warmup_planner', True):
            # Warp compiles cuRobo kernels on first use; do it while the server loads
            # (the client waits up to 15 min), not inside a 120 s get_action call.
            import shutil
            cache = POLICY_DIR / '.warp-cache'
            # A checkpoint may ship prebuilt portable (PTX) kernels next to its bundles.
            shipped = resolve_bundle(model_cfg).parent / 'warp-cache'
            if shipped.is_dir() and not cache.exists():
                shutil.copytree(shipped, cache)
            os.environ.setdefault('WARP_CACHE_PATH', str(cache))
            try:
                import robodojo_toolkit
            except ImportError:
                robodojo_toolkit = None  # Bundles without the toolkit still run.
            if robodojo_toolkit is not None:
                robodojo_toolkit.shared_planner(model_cfg.get('planner_config', 'official'))
        self.action_wait_s = float(model_cfg.get('action_wait_s', 100.0))
        # gemini_generate is available only when this (self-hosted) server holds a key.
        self.gemini = official_service(model_cfg)
        print(f"[AgentBundle] gemini_generate {'enabled' if self.gemini else 'not configured'}", flush=True)
        self.bridge = None
        self.reset()

    def agentbundle_hello(self, payload):
        """Worker handshake from our endpoint: its task must match and its token must be ours."""
        payload = payload if isinstance(payload, dict) else {}
        task = str(self.model_cfg.get('task_name') or '')
        if payload.get('task_name') != task:
            return {'ok': False, 'reason': f'this backend serves task {task!r}'}
        if self._token and not hmac.compare_digest(str(payload.get('token', '')), self._token):
            return {'ok': False, 'reason': 'invalid token'}
        self._authenticated = True
        return {'ok': True, 'task_name': task, 'gemini': self.gemini is not None}

    def _require_session(self):
        if not self._authenticated:
            raise PermissionError('Call agentbundle_hello with this backend\'s token first')

    def reset(self):
        if self.bridge is not None:
            self._require_session()
            self.bridge.cancel()
        self.bridge = Bridge(self.main, action_wait_s=self.action_wait_s,
                             output_dir=self.model_cfg.get('output_dir'), gemini=self.gemini)

    def update_obs(self, obs):
        self._require_session()
        self.bridge.update_obs(obs)

    def get_action(self):
        self._require_session()
        return self.bridge.get_action()

    def update_obs_batch(self, obs_list):
        raise NotImplementedError('AgentBundle runs one environment per policy server (eval_batch: false)')

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError('AgentBundle runs one environment per policy server (eval_batch: false)')
