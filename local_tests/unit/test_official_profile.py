"""Official RoboDojo/XPolicyLab profile: contract, workspace and bundle bridge."""
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from services.mcp_contract import Contract, ObservationProfile, OFFICIAL_ROBOT_TOOLS

PROJECT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('mode', ['direct-control', 'teacher-student'])
def test_only_auto_research_mode_exists(mode):
    with pytest.raises(ValueError, match='Unknown MCP mode'):
        Contract(mode, 'exploration', 'official')


@pytest.mark.parametrize('profile', ['rgbd', 'rgb-only'])
def test_sessions_are_configured_only_with_the_official_profile(tmp_path, profile):
    from dataclasses import asdict
    from services.controller.config import Configuration
    from test_controller_backend import configuration
    raw = asdict(configuration(tmp_path))
    raw['root'] = '/mnt/ssd8/official-profile-test'
    assert Configuration.from_dict(raw).observation_profile == 'official'
    with pytest.raises(ValueError, match='official observation profile'):
        Configuration.from_dict({**raw, 'observation_profile': profile})
    with pytest.raises(ValueError, match='Teacher'):
        Configuration.from_dict({**raw, 'teacher_source_session': 'x'})
    legacy = Configuration.from_dict({**raw, 'student_model': 'm', 'student_reasoning_effort': 'high'})
    assert not hasattr(legacy, 'student_model')


def test_official_contract_offers_only_official_robot_tools():
    contract = Contract('auto-research', 'exploration', 'official')
    assert contract.robot_names == OFFICIAL_ROBOT_TOOLS and contract.gemini
    for name in ('robodojo_free_space_move', 'robodojo_pose_math', 'robodojo_pixel_to_position'):
        with pytest.raises(PermissionError):
            contract.require_robot(name, {})
    contract.require_robot('gemini_generate', {})  # Official via a self-hosted remote policy server.
    with pytest.raises(ValueError):
        Contract('auto-research', 'exploration', 'official', 2).require_robot('gemini_generate', {'env_id': 0})
    isolated = Contract('auto-research', 'isolated', 'official')
    assert {t['name'] for t in isolated.select([{'name': 'gemini_generate', 'description': '',
                                                 'inputSchema': {}}])} == {'gemini_generate'}
    assert isolated.manifest()['gemini'] is True
    listed = {t['name'] for t in contract.robot_definitions()}
    assert listed == OFFICIAL_ROBOT_TOOLS


def test_gripper_semantics_and_field_stripping():
    packet = {'states': list(range(14)), 'commanded_gripper_openings': [0.5, 0.25], 'depth': {},
              'frames': [{'states': list(range(14)), 'commanded_gripper_openings': [1.0, 0.0]}]}
    official = ObservationProfile('official').clean(packet)
    assert official['states'][6] == 0.5 and official['states'][13] == 0.25
    assert official['frames'][0]['states'][6] == 1.0 and 'depth' not in official
    for name in ('rgbd', 'rgb-only'):
        cleaned = ObservationProfile(name).clean(packet)
        assert cleaned['states'][6] == 6 and 'commanded_gripper_openings' not in json.dumps(cleaned)
    missing = ObservationProfile('official').clean({'states': list(range(14)),
                                                    'commanded_gripper_openings': [float('nan'), 0.2]})
    assert missing['states'][6] == 6  # Never publish a non-finite command.


def test_official_workspace_drops_unavailable_skills(tmp_path):
    from services.mcp_workspace import compose
    workspace = tmp_path / 'ws'
    shutil.copytree(PROJECT / 'auto_research_agent', workspace, symlinks=True)
    compose(workspace, Contract('auto-research', 'exploration', 'official'), task='make_kong')
    skills = {p.name for p in (workspace / '.agents/skills').iterdir()}
    assert not skills & {'robodojo-pose-math', 'robodojo_free_space_move', 'robot_only_action_preview',
                         'action-search', 'robodojo-pixel-to-position'}
    assert (workspace / '.agents/skills/motion-toolkit/SKILL.md').is_file()
    assert (workspace / '.agents/skills/rgb-position-calibration/CALIBRATION.md').is_file()
    session = (workspace / 'MCP_SESSION.md').read_text()
    assert 'robodojo_step' in session and 'robodojo_toolkit' in session
    assert 'robodojo_pose_math' not in session and 'depth' in session
    manifest = json.loads((workspace / 'MCP_CONTRACT.json').read_text())
    assert manifest['gemini'] is True and set(manifest['robot_tools']) == OFFICIAL_ROBOT_TOOLS
    for profile in ('rgbd', 'rgb-only'):
        with pytest.raises(ValueError, match='official observation profile'):
            compose(tmp_path / profile, Contract('auto-research', 'exploration', profile), task='make_kong')


def _bridge():
    path = PROJECT / 'official/xpolicylab/AgentBundle/bundle_bridge.py'
    spec = importlib.util.spec_from_file_location('bundle_bridge', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _official_obs(state):
    pose = np.array([0.1, 0.2, 0.9, 1, 0, 0, 0], np.float32)
    return {'instruction': 'probe', 'state': {**state, 'left_ee_pose': pose, 'right_ee_pose': pose},
            'vision': {c: {'color': np.zeros((8, 8, 3), np.uint8)} for c in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')}}


@pytest.mark.parametrize('limit', [3, 40])
def test_bridge_runs_bundle_under_official_episode_loop(tmp_path, limit):
    bridge_module = _bridge()
    seen = []

    def main(ctx):
        reply = ctx.call('robodojo_observe')
        meta = json.loads(reply['content'][0]['text'])
        assert [a['camera'] for a in meta['attachments']] == ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        with pytest.raises(PermissionError, match='unavailable in official mode'):
            ctx.call('robodojo_free_space_move', arm='left')
        target = meta['states'][:6] + [0.4] + meta['states'][7:13] + [0.6]
        for _ in range(5):
            meta = json.loads(ctx.call('robodojo_step', actions=[target, target])['content'][0]['text'])
            seen.append((meta['step_id'], meta['states'][6]))

    state = {f'{a}_{k}': np.zeros(6 if k == 'arm_joint_state' else 1, np.float32)
             for a in ('left', 'right') for k in ('arm_joint_state', 'ee_joint_state')}
    bridge = bridge_module.Bridge(main, action_wait_s=5, output_dir=tmp_path)
    executed = 0
    # Mirrors XPolicyLab demo deploy.py: update_obs after every executed action.
    bridge.update_obs(_official_obs(state))
    while executed < limit:
        for action in bridge.get_action():
            state = {k: np.asarray(v, np.float32) for k, v in action.items()}
            executed += 1
            bridge.update_obs(_official_obs(state))
            if executed >= limit:
                break
    bridge.cancel()
    bridge.thread.join(timeout=5)
    assert not bridge.thread.is_alive() and bridge.error is None
    if limit == 40:
        assert [s for s, _ in seen] == [2, 4, 6, 8, 10]
        assert [g for _, g in seen] == pytest.approx([0.4] * 5)  # Commanded gripper is observed.
    else:
        assert len(seen) == 1  # Episode ended mid-bundle; cancelled cleanly.


def test_bridge_and_runtime_pass_native_ee_actions():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'official/xpolicylab/AgentBundle'))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'auto_research_agent/api'))
    from bundle_bridge import ee_row_to_action, row_to_action
    from runtime import _tool_rows
    ee = [0.1, 0, 1, 1, 0, 0, 0, 0.5, 0.3, 0, 1, 1, 0, 0, 0, 1.0]
    action = ee_row_to_action(ee)
    assert set(action) == {'left_ee_pose', 'left_ee_joint_state', 'right_ee_pose', 'right_ee_joint_state'}
    for bad in (ee[:14], ee[:3] + [2, 0, 0, 0] + ee[7:], ee[:7] + [1.5] + ee[8:]):
        with pytest.raises(ValueError):
            ee_row_to_action(bad)
    joint = row_to_action([0.0] * 14)
    calls = _tool_rows([joint, joint, action] + [action] * 50 + [joint])
    assert [(tool, len(rows)) for tool, rows in calls] == [
        ('robodojo_step', 2), ('robodojo_step_ee', 50), ('robodojo_step_ee', 1), ('robodojo_step', 1)]
    assert calls[1][1][0] == pytest.approx(ee)


def test_contract_rejections_report_no_action_executed():
    from services.controller.errors import public_failure
    from services.mcp_contract import InvalidArguments
    contract = Contract('auto-research', observation_profile='official', environments=2)
    for name, arguments in (('robodojo_pose_math', {}), ('robodojo_step', {'actions': []})):
        with pytest.raises((PermissionError, ValueError)) as caught:
            contract.require_robot(name, arguments)
        failure = public_failure(caught.value, operation=name, stage='lifecycle')
        assert failure['no_action_executed'] is True and failure['control_uncertain'] is False
    assert public_failure(InvalidArguments('x'), stage='lifecycle', uncertain=True)['no_action_executed'] is False
    assert public_failure(RuntimeError('native'), stage='native_execution')['no_action_executed'] is False


def test_adapter_routes_std_and_random_variants_to_their_own_bundles(tmp_path, monkeypatch):
    import sys
    import types
    template = types.ModuleType('XPolicyLab.model_template')
    template.ModelTemplate = object
    monkeypatch.setitem(sys.modules, 'XPolicyLab', types.ModuleType('XPolicyLab'))
    monkeypatch.setitem(sys.modules, 'XPolicyLab.model_template', template)
    spec = importlib.util.spec_from_file_location(
        'agentbundle_model', Path(__file__).resolve().parents[2] / 'official/xpolicylab/AgentBundle/model.py',
        submodule_search_locations=[])
    module = importlib.util.module_from_spec(spec)
    module.__package__ = 'agentbundle'
    monkeypatch.setitem(sys.modules, 'agentbundle', types.ModuleType('agentbundle'))
    monkeypatch.setitem(sys.modules, 'agentbundle.bundle_bridge', types.SimpleNamespace(Bridge=None, load_main=None))
    monkeypatch.setitem(sys.modules, 'agentbundle.gemini_router', types.SimpleNamespace(official_service=lambda *a, **k: None))
    spec.loader.exec_module(module)
    checkpoint = tmp_path / 'ckpt'
    for task in ('make_toast', 'make_toast_random'):
        (checkpoint / task).mkdir(parents=True)
        (checkpoint / task / 'controller.py').write_text('def main(ctx): pass\n')
    for task in ('make_toast', 'make_toast_random'):
        assert module.resolve_bundle({'task_name': task, 'ckpt_name': str(checkpoint)}) == checkpoint / task
    # A variant without its own bundle is not silently served by another task's bundle.
    with pytest.raises(FileNotFoundError, match='make_kong'):
        module.resolve_bundle({'task_name': 'make_kong', 'ckpt_name': str(checkpoint)})
