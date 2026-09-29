"""TASK.md generalization-variant context, derived from task configs and saved layouts."""
import json

import yaml

from services.robodojo.task_variants import task_variant_context


def write_config(root, task, value):
    path = root/'task/RoboDojo/config'/f'{task}.yml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value))


def write_layout(root, task, index, table):
    path = root/'Assets/Eval_Layout/RoboDojo/arx_x5/0'/f'{task}_{index}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'Room': {'default': 'r'}, 'Table': {'default': table},
                                'Ground': {'materials': None}, 'Background': {'category_name': 'b'}}))


def test_random_variant_context_lists_changes_and_randomization_types(tmp_path):
    group = lambda index: [{'common': {'xlim': [0, 1], 'ylim': [0, 1], 'rotate_rand': True, 'rotate_deg': 15},
                            'category': [{'name': 'toaster', 'index': index}]}]
    write_config(tmp_path, 'make_toast', {'Articulation': group([0, 1])})
    write_config(tmp_path, 'make_toast_random', {'Articulation': group([2, 4]),
                                                 'Clutter': [{'nums': 15, 'yaml_path': 'Clutter/clutter.yml'}]})
    write_layout(tmp_path, 'make_toast', 0, 'wood')
    write_layout(tmp_path, 'make_toast_random_extra', 0, 'ignored')  # Another task's layouts.
    for index, table in enumerate(('wood', 'marble', 'metal')):
        write_layout(tmp_path, 'make_toast_random', index, table)
    text = task_variant_context('make_toast_random', root=tmp_path)
    assert '[2, 4] instead of the standard [0, 1]' in text
    assert 'about 15 random distractor objects' in text
    assert '3 different table models (standard: 1)' in text and 'room models' not in text
    for kind in ('`placement`', '`rotation`', '`instances`', '`clutter`', '`table`'):
        assert kind in text
    assert '`physics`' not in text and '`room`' not in text
    assert task_variant_context('make_toast', root=tmp_path) == ''
