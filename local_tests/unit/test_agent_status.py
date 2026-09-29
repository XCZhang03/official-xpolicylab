"""Agent-visible status and the official-only tool surface of the auto-research MCP."""
from dataclasses import replace
import json

import pytest

from services.controller.frontend import ResearchFrontend
from test_controller_backend import Backend, Runner, source, supervisor  # noqa: F401


def test_status_reports_timeout_time_bundles_and_formal_allowance(supervisor, tmp_path):  # noqa: F811
    s, backend = supervisor
    object.__setattr__(s.config, 'formal', replace(s.config.formal, wall_seconds=45))
    front = ResearchFrontend(s, tmp_path/'workspace')
    ident, manifest = s.register(source(tmp_path))
    status = front.status()
    assert status['episode_timeout_seconds'] == 45
    assert 'development_seconds_remaining' not in status
    assert status['bundles'] == {ident: {'project_directory': '/workspace/code/controller',
                                         'sha256': manifest['sha256'], 'rehearsal_qualified': False}}
    assert status['api_budget']['session_cost_limit_usd'] == 10 and status['formal_api_budget']['phase'] == 'formal'
    backend.success = True
    s.run(ident, formal=False)
    assert front.status()['bundles'][ident]['rehearsal_qualified'] is True
    assert 'development_seconds_remaining' not in front.status()
    assert 'development_reserved_seconds' not in s.state




def test_only_official_robot_tools_and_lifecycle_are_offered(supervisor, tmp_path):  # noqa: F811
    s, _ = supervisor
    front = ResearchFrontend(s, tmp_path/'workspace')
    names = {t['name'] for t in front.tools()}
    from services.controller.frontend import DEFINITIONS
    assert len(DEFINITIONS) == 7
    assert names == {'robodojo_observe', 'robodojo_status', 'robodojo_step', 'robodojo_step_ee',
                     'gemini_generate'} | {t['name'] for t in DEFINITIONS}
    for name in ('robodojo_pose_math', 'robodojo_step_eef'):
        with pytest.raises(PermissionError):
            front.call(name, {})
