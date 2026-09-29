"""The dedicated Claude account for experiments: its setup-token and identity.

Experiments never use the operator's personal Claude login (~/.claude). A separate
account signs in once in its own CLAUDE_CONFIG_DIR and issues a long-lived
``claude setup-token`` token. The relay (messages_relay.Credential) reads only
that token file:

    ~/.config/robodojo/                  (0700; ROBODOJO_CLAUDE_EXPERIMENT_DIR overrides)
      claude-experiment-home/            dedicated CLAUDE_CONFIG_DIR, for sign-in only
      claude-experiment.token            the token (0600), re-read by the relay per request
      claude-experiment.json             identity (0600): email, organization, created_at

Set up or rotate with ``bash scripts/setup_claude_experiment_login.sh``.
"""
from __future__ import annotations

import argparse
import getpass
import http.client
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from harness.claude_cli.messages_relay import OAUTH_BETA, PROVIDERS, Credential, _private_text

SETUP = 'bash scripts/setup_claude_experiment_login.sh'


def root() -> Path:
    return Path(os.environ.get('ROBODOJO_CLAUDE_EXPERIMENT_DIR', Path.home() / '.config/robodojo'))


def config_dir() -> Path:
    return root() / 'claude-experiment-home'


def token_file() -> Path:
    return root() / 'claude-experiment.token'


def metadata_file() -> Path:
    return root() / 'claude-experiment.json'


def _write_private(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.')
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def account(directory: Path | None, claude: str | None = None) -> dict:
    """Signed-in account of a Claude config dir (None: the operator's default login)."""
    binary = claude or shutil.which('claude')
    if not binary:
        raise RuntimeError('Claude Code is not installed on the host')
    environment = {k: v for k, v in os.environ.items() if k != 'CLAUDE_CONFIG_DIR'}
    if directory is not None:
        environment['CLAUDE_CONFIG_DIR'] = str(directory)
    result = subprocess.run([binary, 'auth', 'status', '--json'], env=environment,
                            capture_output=True, text=True, timeout=60)
    try:
        status = json.loads(result.stdout or '{}')
    except ValueError:
        status = {}
    if not isinstance(status, dict):
        status = {}
    keys = {'email': ('email',), 'organization': ('orgName',), 'org_id': ('orgId',),
            'subscription': ('subscriptionType',)}
    found = {name: next((status[k] for k in options if isinstance(status.get(k), str)), None)
             for name, options in keys.items()}
    found['logged_in'] = status.get('loggedIn') is True
    return found


def install(token: str, identity: dict) -> dict:
    """Store the experiment token (0600) and its non-secret identity."""
    token = token.strip()
    if not token or len(token) > 16384 or any(c.isspace() for c in token) or token.startswith('{'):
        raise ValueError('That does not look like a `claude setup-token` token')
    if not identity.get('email'):
        raise ValueError('Sign in to the experiment account first (no account email found)')
    _write_private(token_file(), token + '\n')
    record = {'email': identity['email'], 'organization': identity.get('organization'),
              'org_id': identity.get('org_id'), 'subscription': identity.get('subscription'),
              'created_at': time.time()}
    _write_private(metadata_file(), json.dumps(record, indent=2) + '\n')
    return record


def identity() -> dict:
    try:
        value = json.loads(_private_text(metadata_file()))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def status() -> dict:
    """Presence and identity without network access (dashboard and preflight)."""
    path = token_file()
    try:
        Credential('claude-login', key_file=path)
    except FileNotFoundError:
        raise RuntimeError(f'No Claude experiment token at {path}; run {SETUP}') from None
    except (OSError, ValueError) as exc:
        raise RuntimeError(f'Claude experiment token at {path} is unusable ({exc}); run {SETUP}') from None
    record = identity()
    created = record.get('created_at')
    return {'token_file': str(path), 'email': record.get('email'), 'organization': record.get('organization'),
            'subscription': record.get('subscription'),
            'token_age_days': round((time.time() - created) / 86400, 1) if isinstance(created, (int, float)) else None}


def _request(method, path, headers, body=None, timeout=30):
    connection = http.client.HTTPSConnection('api.anthropic.com', timeout=timeout)
    try:
        connection.request(method, path, body, headers)
        response = connection.getresponse()
        return response.status, response.read(65536)
    finally:
        connection.close()


def token_account(headers, timeout=30):
    """Best effort: the account behind the token (Claude Code's OAuth profile endpoint)."""
    try:
        code, data = _request('GET', '/api/oauth/profile', headers, timeout=timeout)
        value = json.loads(data) if code == 200 else {}
    except (OSError, ValueError, http.client.HTTPException):
        return None
    found = value.get('account') if isinstance(value, dict) and isinstance(value.get('account'), dict) else value
    email = next((found.get(k) for k in ('email_address', 'email') if isinstance(found, dict)
                  and isinstance(found.get(k), str)), None)
    return email


def check(timeout: float = 30, *, personal_email: str | None = None) -> dict:
    """status() plus a free count_tokens call proving the token is accepted.

    When the profile endpoint names the token's account, it must be the recorded
    experiment account and must not be the operator's personal login.
    """
    result = status()
    headers = {**Credential('claude-login', key_file=token_file()).headers(),
               'anthropic-version': '2023-06-01', 'anthropic-beta': OAUTH_BETA, 'Content-Type': 'application/json'}
    body = json.dumps({'model': PROVIDERS['claude-login']['model'],
                       'messages': [{'role': 'user', 'content': 'ping'}]})
    code, _ = _request('POST', '/v1/messages/count_tokens', headers, body, timeout)
    if code in (401, 403):
        raise RuntimeError(f'The Claude experiment token was rejected (HTTP {code}); '
                           f'it may be revoked or expired: run {SETUP}')
    result.update(verified=code == 200, check_status=code)
    owner = token_account(headers, timeout)
    result['token_account'] = owner
    if owner and personal_email and owner.lower() == personal_email.lower():
        raise RuntimeError(f'The token belongs to your personal account {owner}; issue it while signed in '
                           f'to the experiment account and run {SETUP} again')
    if owner and result.get('email') and owner.lower() != result['email'].lower():
        raise RuntimeError(f'The token belongs to {owner}, not the experiment account {result["email"]}; '
                           f'run {SETUP} again')
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('paths', help='print the dedicated config dir and token path')
    commands.add_parser('status', help='token presence and identity, no network')
    commands.add_parser('check', help='status plus one free API call')
    commands.add_parser('install', help='store a pasted setup-token (read without echo)')
    args = parser.parse_args(argv)
    try:
        if args.command == 'paths':
            print(json.dumps({'config_dir': str(config_dir()), 'token_file': str(token_file())}))
            return 0
        if args.command == 'install':
            experiment = account(config_dir())
            personal = account(None)
            if not experiment.get('email'):
                raise RuntimeError(f'Not signed in under {config_dir()}; sign in to the experiment account first')
            if personal.get('email') and personal['email'] == experiment['email']:
                raise RuntimeError(f'{experiment["email"]} is also your personal Claude login; '
                                   'sign in to a separate account dedicated to experiments')
            token = getpass.getpass('Paste the `claude setup-token` token (hidden): ') if sys.stdin.isatty() \
                else sys.stdin.readline()
            record = install(token, experiment)
            print(json.dumps({'installed': str(token_file()), **record}, indent=2))
            return 0
        if args.command == 'check':
            result = check(personal_email=account(None).get('email'))
            if not result['token_account']:
                print("note: the token's account could not be verified; make sure setup-token ran in a "
                      'browser signed in to the experiment account', file=sys.stderr)
            print(json.dumps(result, indent=2))
            return 0
        print(json.dumps(status(), indent=2))
        return 0
    except RuntimeError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
