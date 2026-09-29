"""Decide which served task an episode is, from its first observation only.

The evaluator's client sends observations, never task names. The instruction names
the base task (make_toast, hang_mugs, ...); whether it is the `*_random` variant shows
in the scene (distractor clutter, randomized table/floor/lighting/background, other
object instances). Gemini reads the instruction and the head camera and chooses
among the served tasks; its answer is validated, and without a usable answer a word
match between the instruction and task names picks the base task and the `_random` bundle is preferred,
since it was developed for the harder scenes.
"""
import base64
import json
import re
from pathlib import Path

import cv2
import numpy as np
import yaml

RANDOM = '_random'


def _words(text):
    return {w for w in re.findall(r'[a-z]+', str(text).lower()) if len(w) > 2}


def _stem(word):
    return re.sub(r'(ing|es|s)$', '', word)


def variant_hints(tasks, configs_dir=None):
    """task -> a one-line scene description for the served `*_random` variants."""
    hints = {}
    for task in tasks:
        if not task.endswith(RANDOM):
            continue
        clutter = 0
        path = Path(configs_dir) / f'{task}.yml' if configs_dir else None
        if path is not None and path.is_file():
            config = yaml.safe_load(path.read_text()) or {}
            clutter = sum(int(g.get('nums', 0) or 0) for g in config.get('Clutter') or [] if isinstance(g, dict))
        hints[task] = ((f'about {clutter} unrelated distractor objects scattered on the table, and ' if clutter else '')
                       + 'domain randomization: table/floor materials, lighting and background can differ '
                       'from the plain default scene, and task objects can be other instances')
    return hints


def head_image(obs, width=640):
    vision = obs.get('vision') or {}
    camera = vision.get('cam_head') or next(iter(vision.values()), None)
    image = camera.get('color') if isinstance(camera, dict) else camera
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return None
    image = np.ascontiguousarray(image[..., :3])
    if image.shape[1] > width:
        image = cv2.resize(image, (width, int(image.shape[0] * width / image.shape[1])), interpolation=cv2.INTER_AREA)
    # cv2.imencode expects BGR; the observation is RGB.
    ok, buffer = cv2.imencode('.jpg', image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer.tobytes()).decode() if ok else None


class TaskRouter:
    def __init__(self, tasks, *, gemini=None, configs_dir=None, log=print):
        self.tasks = sorted(tasks)
        if not self.tasks:
            raise ValueError('No served tasks')
        self.gemini, self.log = gemini, log
        self.hints = variant_hints(self.tasks, configs_dir)

    def base_by_words(self, instruction):
        """Served tasks whose name words best overlap the instruction's words."""
        said = {_stem(w) for w in _words(instruction)}
        scores = {}
        for task in self.tasks:
            name = {_stem(w) for w in _words(task.replace('_', ' ')) - {'random'}}
            scores[task] = len(name & said) / max(1, len(name))
        best = max(scores.values())
        return [t for t in self.tasks if scores[t] == best and best > 0], scores

    def fallback(self, instruction):
        candidates, _ = self.base_by_words(instruction)
        candidates = candidates or self.tasks
        variants = [t for t in candidates if t.endswith(RANDOM)]
        return (variants or candidates)[0]

    def ask_gemini(self, instruction, image):
        lines = []
        for task in self.tasks:
            hint = self.hints.get(task, 'the standard scene: plain default table and room, no distractor objects'
                                  if f'{task}{RANDOM}' in self.tasks else 'the task scene')
            lines.append(f'- {task}: {hint}')
        prompt = ('You label robot evaluation episodes. Choose which task this episode is.\n'
                  f'Instruction given to the robot: {instruction!r}\n'
                  'The image is the head camera at the start of the episode. Tasks:\n' + '\n'.join(lines) +
                  '\nThe instruction identifies the base task. Where a task has a `_random` variant, decide from '
                  'the image: unrelated clutter objects or a non-default scene look mean the variant.')
        schema = {'type': 'object', 'additionalProperties': False, 'required': ['task', 'reason'],
                  'properties': {'task': {'type': 'string', 'enum': self.tasks}, 'reason': {'type': 'string'}}}
        content = [{'type': 'text', 'text': prompt}]
        if image:
            content.append({'type': 'image', 'mimeType': 'image/jpeg', 'data': image})
        reply = self.gemini({'messages': [{'role': 'user', 'content': content}], 'temperature': 0, 'max_tokens': 400,
                             'response_format': {'type': 'json_schema', 'json_schema': {
                                 'name': 'episode_task', 'strict': True, 'schema': schema}}})
        result = json.loads(reply['content'][0]['text'])
        message = result.get('message') or {}
        if result.get('finish_reason') not in (None, 'stop') or message.get('refusal'):
            raise ValueError(f'incomplete answer: {result.get("finish_reason")}')
        answer = json.loads(message.get('content') or '')
        if answer.get('task') not in self.tasks:
            raise ValueError(f'not a served task: {answer.get("task")!r}')
        return answer

    def detect(self, obs):
        """{'task', 'method', ...} for the episode whose first observation is obs."""
        instruction = str(obs.get('instruction') or '')
        if len(self.tasks) == 1:
            return {'task': self.tasks[0], 'method': 'single', 'instruction': instruction}
        # Gemini decides every episode (one call); word matching is only the fallback,
        # since instructions paraphrase task names ("toaster" for make_toast).
        words, _ = self.base_by_words(instruction)
        if self.gemini is not None:
            try:
                answer = self.ask_gemini(instruction, head_image(obs))
                return {'task': answer['task'], 'method': 'gemini', 'reason': answer.get('reason'),
                        'word_candidates': words, 'instruction': instruction}
            except Exception as exc:
                self.log(f'[endpoint] task detection by Gemini failed ({type(exc).__name__}: {exc}); '
                         'using the instruction fallback')
        return {'task': self.fallback(instruction), 'method': 'fallback', 'word_candidates': words,
                'instruction': instruction}
