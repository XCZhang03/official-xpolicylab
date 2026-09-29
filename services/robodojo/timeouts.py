"""Trusted task-scaled wall budgets; no simulator imports or agent overrides."""
import ast
from pathlib import Path
import re

TASKS = Path(__file__).resolve().parents[2] / 'RoboDojo/task/RoboDojo/tasks'
SECONDS_PER_STEP = 6
SESSION_SECONDS_PER_STEP = 24 * 60 * 60 // 400


def task_timeouts(task):
    if not isinstance(task, str) or not re.fullmatch(r'[A-Za-z0-9_]+', task):
        raise ValueError('Invalid task identifier for wall budgets')
    # Read the native literal without importing Isaac or executing task code.
    # Shared task classes (e.g. MakeToastCommon) declare the same unique horizon.
    try:
        tree = ast.parse((TASKS / f'{task}.py').read_text())
    except OSError as exc:
        raise ValueError(f'{task}: native task source unavailable') from exc
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
            and t.value.id == 'self' and t.attr == 'step_lim' for t in node.targets
        ):
            if not isinstance(node.value, ast.Constant) or type(node.value.value) is not int:
                raise ValueError(f'{task}: native step limit must be an integer literal')
            values.append(node.value.value)
    if len(set(values)) != 1 or values[0] <= 0:
        raise ValueError(f'{task}: missing or ambiguous native step limit')
    steps = values[0]
    return {'native_step_limit': steps, 'episode_seconds': steps * SECONDS_PER_STEP,
            'session_seconds': steps * SESSION_SECONDS_PER_STEP}
