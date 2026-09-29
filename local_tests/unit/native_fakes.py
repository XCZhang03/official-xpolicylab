"""Deterministic fake native backend for official joint steps: genuine MCP/runner path, no task claim."""
import base64
import io
import json

from PIL import Image

from services.robodojo.mcp_server import _agent_safe

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def native_row(t=0):
    return {'step_id': t+1, 'observation_step_id': t, 'executed_action': [0.]*6+[.7]+[0.]*6+[.7],
            'episode_ended': False}


def reply(t, row=None):
    image = io.BytesIO()
    Image.new('RGB', (32, 32), (t*20 % 256, 100, 50)).save(image, format='PNG')
    block = {'type': 'image', 'mimeType': 'image/png', 'data': base64.b64encode(image.getvalue()).decode()}
    value = {'episode_id': 'episode', 'step_id': t, 'states': [t*.01]*14,
             'eef_positions': [[-0.3, 0.0, 1.0], [0.3, 0.0, 1.0]], 'eef_quaternions_wxyz': [[1.0, 0.0, 0.0, 0.0]] * 2,
             'attachments': [{'kind': 'rgb', 'camera': c, 'content_index': i+1} for i, c in enumerate(CAMERAS)]}
    if row is not None:
        value['transition'] = {'steps': [row]}
    return {'content': [{'type': 'text', 'text': json.dumps(_agent_safe(value))}, block, block, block]}


def unpack(response):
    return json.loads(response['content'][0]['text'])


class NativeFrames:
    """Joint-step native backend that publishes 25 Hz frame sequences like the simulator."""
    def __init__(self, workspace, terminal_at=4):
        self.frame_workspace = workspace
        self.step, self.serial, self.success = 0, 0, False
        self.terminal_at, self.calls = terminal_at, []

    def start(self, **_):
        self.step = 0
        return {}, []

    def evaluate(self):
        return {'task_complete': self.success}

    def close(self):
        pass

    def tools(self):
        return [{'name': name} for name in ('robodojo_step', 'robodojo_observe', 'robodojo_status')]

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        if name in ('robodojo_observe', 'robodojo_status'):
            response = reply(self.step)
            return unpack(response), response['content'][1:] if name == 'robodojo_observe' else []
        self.serial += 1
        rows, frames = [], []
        directory = self.frame_workspace/f'runtime/frames/sequence-{self.serial}'
        directory.mkdir(parents=True)
        for target in arguments['actions']:
            row = native_row(self.step)
            row['executed_action'] = target
            self.step += 1
            row['terminated'] = self.step == self.terminal_at
            rows.append(row)
            response = reply(self.step, row)
            value = unpack(response)
            files = []
            for attachment in value['attachments']:
                path = directory/f'frame_{self.step:06d}_{attachment["camera"]}_rgb.png'
                path.write_bytes(base64.b64decode(response['content'][attachment['content_index']]['data']))
                files.append({'kind': 'rgb', 'camera': attachment['camera'],
                              'path': str(path.relative_to(self.frame_workspace))})
            frames.append({'step_id': self.step, 'states': value['states'], 'files': files})
            if row['terminated']:
                break
        manifest = directory/'manifest.json'
        manifest.write_text(json.dumps({'frequency_hz': 25, 'frames': frames}))
        value.update(transition={'steps': rows}, episode_ended=rows[-1]['terminated'],
                     frame_sequence={'manifest_path': str(manifest.relative_to(self.frame_workspace))})
        return _agent_safe(value), response['content'][1:]
