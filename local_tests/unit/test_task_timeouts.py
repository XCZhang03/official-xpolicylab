from dataclasses import replace
from pathlib import Path

import pytest

from services.robodojo.timeouts import task_timeouts


@pytest.mark.parametrize('task,steps', [('plug_in_charger',400), ('make_kong',600),
    ('pour_by_language',800), ('make_toast',1400), ('make_toast_random',1400)])
def test_task_scaled_budgets(task, steps):
    assert task_timeouts(task) == {'native_step_limit':steps,
        'episode_seconds':6*steps, 'session_seconds':216*steps}


def test_every_installed_task_has_unambiguous_native_horizon():
    root = Path(__file__).resolve().parents[2]/'RoboDojo/task/RoboDojo/config'
    for path in root.glob('*.yml'):
        if not path.stem.startswith('_'):
            assert task_timeouts(path.stem)['native_step_limit'] > 0


def test_research_timeouts_apply_to_both_phases():
    from local_tests.unit.test_controller_backend import configuration
    from services.controller.config import with_task_timeouts
    config = replace(configuration(Path('/tmp')), root=Path('/mnt/ssd8/timeout-test'), task='make_toast')
    scaled = with_task_timeouts(config)
    assert scaled.formal.wall_seconds == scaled.development.wall_seconds == 8400
    assert scaled.formal.memory_mb == config.formal.memory_mb
