"""Dedicated Claude experiment login: token storage, identity guards, launch and dashboard wiring."""
import json
import stat

import pytest

from harness.claude_cli import experiment_login as login


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path/'home'))
    monkeypatch.setenv('ROBODOJO_CLAUDE_EXPERIMENT_DIR', str(tmp_path/'robodojo'))
    (tmp_path/'home').mkdir()
    return tmp_path


def accounts(monkeypatch, experiment, personal):
    monkeypatch.setattr(login, 'account', lambda directory, claude=None:
                        {'email': experiment if directory is not None else personal,
                         'organization': 'Lab', 'org_id': 'o', 'subscription': 'max', 'logged_in': True})


def test_install_stores_a_private_token_and_identity_only(isolated):
    record = login.install('sk-ant-oat01-secret\n', {'email': 'robot@lab.example', 'organization': 'Lab'})
    token, metadata = login.token_file(), login.metadata_file()
    assert token.read_text() == 'sk-ant-oat01-secret\n'
    for path in (token, metadata):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(token.parent.stat().st_mode) == 0o700
    assert 'secret' not in metadata.read_text() and record['email'] == 'robot@lab.example'
    status = login.status()
    assert status['email'] == 'robot@lab.example' and status['token_age_days'] == 0
    for bad in ('', 'has space', '{"claudeAiOauth": {}}'):
        with pytest.raises(ValueError):
            login.install(bad, {'email': 'robot@lab.example'})


def test_status_explains_a_missing_token(isolated):
    with pytest.raises(RuntimeError, match='setup_claude_experiment_login'):
        login.status()


def test_install_command_refuses_the_personal_account(isolated, monkeypatch, capsys):
    accounts(monkeypatch, 'me@personal.example', 'me@personal.example')
    monkeypatch.setattr('sys.stdin', type('S', (), {'isatty': lambda self: False, 'readline': lambda self: 'tok\n'})())
    assert login.main(['install']) == 1
    assert 'personal Claude login' in capsys.readouterr().err
    assert not login.token_file().exists()
    accounts(monkeypatch, 'robot@lab.example', 'me@personal.example')
    assert login.main(['install']) == 0
    assert login.token_file().read_text() == 'tok\n'


def test_check_rejects_revoked_and_personal_tokens(isolated, monkeypatch):
    login.install('sk-ant-oat01-x', {'email': 'robot@lab.example'})
    responses = {}
    monkeypatch.setattr(login, '_request', lambda method, path, headers, body=None, timeout=30: responses[path])
    responses['/v1/messages/count_tokens'] = (401, b'{}')
    with pytest.raises(RuntimeError, match='rejected'):
        login.check()
    responses['/v1/messages/count_tokens'] = (200, b'{"input_tokens": 3}')
    responses['/api/oauth/profile'] = (200, json.dumps({'account': {'email_address': 'me@personal.example'}}).encode())
    with pytest.raises(RuntimeError, match='personal account'):
        login.check(personal_email='me@personal.example')
    responses['/api/oauth/profile'] = (200, json.dumps({'account': {'email_address': 'other@lab.example'}}).encode())
    with pytest.raises(RuntimeError, match='not the experiment account'):
        login.check()
    responses['/api/oauth/profile'] = (404, b'')
    result = login.check(personal_email='me@personal.example')
    assert result['verified'] and result['token_account'] is None
    responses['/api/oauth/profile'] = (200, json.dumps({'account': {'email_address': 'ROBOT@lab.example'}}).encode())
    assert login.check()['token_account'] == 'ROBOT@lab.example'


def test_deploy_defaults_to_the_experiment_token_and_records_the_account(isolated, monkeypatch):
    from harness.claude_cli import auto_research as claude
    from test_controller_backend import configuration
    login.install('sk-ant-oat01-x', {'email': 'robot@lab.example', 'organization': 'Lab'})
    config = configuration(isolated/'session')
    monkeypatch.setattr(claude, 'claude_binary', lambda path=None: isolated/'claude')
    monkeypatch.setattr(claude, 'prepare_workspace', lambda config, name: {
        'workspace': (isolated/'ws').resolve(), 'home': (isolated/'claude-home').resolve(),
        'configuration': isolated/'c.json', 'bridge': isolated/'b.py'})
    (isolated/'ws').mkdir(); (isolated/'claude-home').mkdir()
    deployed = claude.deploy(config, provider='claude-login')
    relay = json.loads(deployed['relay_configuration'].read_text())
    assert relay['key_file'] == str(login.token_file()) and 'credentials_file' not in relay
    assert relay['account'] == {'email': 'robot@lab.example', 'organization': 'Lab'}
    assert 'sk-ant' not in deployed['relay_configuration'].read_text()


def test_sessions_prepared_with_the_personal_login_cannot_launch(isolated, monkeypatch):
    from harness.claude_cli import auto_research as claude
    from test_controller_backend import configuration
    config = configuration(isolated/'old')
    relay = isolated/'model-relay.json'
    relay.write_text(json.dumps({'provider': 'claude-login', 'key_file': None,
                                 'credentials_file': str(isolated/'home/.claude/.credentials.json')}))
    monkeypatch.setattr(claude, 'DockerSandbox', lambda *a, **k: type('D', (), {'preflight': lambda self: None})())
    monkeypatch.setattr(claude.workspace_storage, 'validate', lambda *a: None)
    with pytest.raises(RuntimeError, match='personal Claude login'):
        claude.launch(config, {'workspace': isolated, 'claude_home': isolated, 'relay_configuration': relay}, prompt='x')


def test_dashboard_prepare_reports_a_missing_token_and_shows_the_account(isolated):
    from services.dashboard.sessions import Sessions
    model = Sessions(isolated/'project', lambda: ['make_kong'])
    with pytest.raises(ValueError, match='setup_claude_experiment_login'):
        model.prepare({'agent_cli': 'claude', 'agent_provider': 'claude-login'})
    session = isolated/'session'
    session.mkdir()
    config = {'agent_cli': 'claude', 'agent_provider': 'claude-login'}
    login.install('sk-ant-oat01-x', {'email': 'robot@lab.example'})
    assert Sessions.agent(session, config) == {'cli': 'claude', 'provider': 'claude-login',
                                               'account': 'robot@lab.example', 'pending': True}
    (session/'model-relay.json').write_text(json.dumps({'account': {'email': 'launched@lab.example'}}))
    assert Sessions.agent(session, config)['account'] == 'launched@lab.example'
    assert Sessions.agent(session, {'agent_cli': 'codex'}) == {'cli': 'codex', 'provider': None, 'account': None}
