"""XPolicyLab policy that queries a remote agent API for every action chunk.

No checkpoint, model weights or GPU are used here. Each get_action sends the latest
observation (cameras, proprioception, instruction) to the agent API and returns the
action chunk it answers with, like any hosted model API. The endpoint and key are
runtime configuration; nothing is hardcoded:

    export MOONCAKE_BASE_URL=https://<agent-api-host>      # required, no default
    export MOONCAKE_API_KEY=<key>                          # name set by api_key_env
"""
import os
import time
import uuid
from urllib.parse import urlsplit

import cv2
import msgpack
import msgpack_numpy
import numpy as np

from XPolicyLab.model_template import ModelTemplate

BASE_URL_ENV = 'MOONCAKE_BASE_URL'
IMAGE_KEYS = ('color', 'colors', 'rgb', 'image')


def encode_images(obs, image_format):
    """Compress camera frames for transfer; the API decodes them back to RGB arrays."""
    vision = obs.get('vision')
    if not isinstance(vision, dict) or image_format == 'raw':
        return obs
    obs = dict(obs, vision={name: dict(camera) if isinstance(camera, dict) else camera
                            for name, camera in vision.items()})
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if image_format == 'jpg' else [cv2.IMWRITE_PNG_COMPRESSION, 1]
    for camera in obs['vision'].values():
        if not isinstance(camera, dict):
            continue
        for key in IMAGE_KEYS:
            image = camera.get(key)
            if isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 3:
                # Channel order passes through unchanged (RGB in, RGB out on decode).
                ok, buffer = cv2.imencode('.' + image_format, np.ascontiguousarray(image), params)
                if not ok:
                    raise ValueError(f'Could not encode camera image {key}')
                camera[key] = {'__image__': image_format, 'data': buffer.tobytes()}
    return obs


class AgentAPI:
    """Minimal HTTP client: POST msgpack to {base_url}/v1/act with a bearer key."""

    def __init__(self, base_url, api_key, *, timeout_s, retries):
        import http.client
        parts = urlsplit(base_url)
        if parts.scheme not in ('http', 'https') or not parts.hostname:
            raise ValueError(f'{BASE_URL_ENV} must be an http(s) URL, got {base_url!r}')
        self._connection_class = http.client.HTTPSConnection if parts.scheme == 'https' else http.client.HTTPConnection
        self._host, self._port = parts.hostname, parts.port
        self._path = parts.path.rstrip('/') + '/v1/act'
        self._headers = {'Content-Type': 'application/msgpack', 'Accept': 'application/msgpack'}
        if api_key:
            self._headers['Authorization'] = f'Bearer {api_key}'
        self._timeout_s, self._retries = timeout_s, retries
        self._connection = None

    def post(self, body):
        data = msgpack.packb(body, default=msgpack_numpy.encode, use_bin_type=True)
        error = None
        for attempt in range(self._retries + 1):
            try:
                if self._connection is None:
                    self._connection = self._connection_class(self._host, self._port, timeout=self._timeout_s)
                self._connection.request('POST', self._path, body=data, headers=self._headers)
                response = self._connection.getresponse()
                payload = response.read()
            except (OSError, ConnectionError) as exc:
                # Retried with the same request_id; the API answers a repeat from its cache.
                error = exc
                self.close()
                time.sleep(min(2.0, 0.5 * (attempt + 1)))
                continue
            if response.status != 200:
                raise RuntimeError(f'Agent API error {response.status}: {payload[:500].decode(errors="replace")}')
            return msgpack.unpackb(payload, object_hook=msgpack_numpy.decode, raw=False)
        raise RuntimeError(f'Agent API unreachable: {error}')

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        base_url = os.environ.get(BASE_URL_ENV) or model_cfg.get('base_url')
        if not base_url:
            raise ValueError(f'Set {BASE_URL_ENV} (or base_url in deploy.yml) to the agent API endpoint')
        api_key = os.environ.get(model_cfg.get('api_key_env') or 'MOONCAKE_API_KEY', '')
        self.image_format = str(model_cfg.get('image_format', 'png'))
        if self.image_format not in ('png', 'jpg', 'raw'):
            raise ValueError('image_format must be png, jpg or raw')
        # Below the simulator client's own 120 s request timeout.
        self.api = AgentAPI(str(base_url), api_key, timeout_s=float(model_cfg.get('api_timeout_s', 110.0)),
                            retries=int(model_cfg.get('api_retries', 1)))
        self.session_id = uuid.uuid4().hex
        self.episode_id, self.obs, self.calls = None, None, 0
        self.reset()

    def reset(self):
        self.episode_id, self.obs, self.calls = uuid.uuid4().hex, None, 0

    def update_obs(self, obs):
        # Only the latest observation is sent, with the next action request.
        self.obs = obs

    def get_action(self):
        if self.obs is None:
            raise RuntimeError('get_action before any observation')
        self.calls += 1
        reply = self.api.post({
            'session_id': self.session_id, 'episode_id': self.episode_id,
            'request_id': f'{self.episode_id}:{self.calls}',
            'observation': encode_images(self.obs, self.image_format),
        })
        actions = reply.get('actions') if isinstance(reply, dict) else None
        if not isinstance(actions, list) or not actions:
            raise RuntimeError(f'Agent API returned no actions: {str(reply)[:500]}')
        return [{key: np.asarray(value, dtype=np.float32) for key, value in action.items()} for action in actions]

    def update_obs_batch(self, obs_list):
        raise NotImplementedError('One environment per policy server (eval_batch: false)')

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError('One environment per policy server (eval_batch: false)')
