#!/usr/bin/env python3
"""One entry point for the dashboard, session preparation, the launcher and the contract."""
import argparse
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from services.mcp_contract import Contract


def command(action, arguments, project=PROJECT):
    project = Path(project)
    if action == 'dashboard':
        return ['bash', str(project/'scripts/run_dashboard.sh'), *arguments]
    python = os.environ.get('ROBODOJO_PYTHON', str(project/'runtime/envs/robodojo/bin/python'))
    if action == 'prepare':
        return [python, str(project/'scripts/configure_auto_research.py'), *arguments]
    if action == 'research':
        return [python, '-m', 'harness.codex_cli.auto_research', *arguments]
    raise ValueError('Unknown command')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['dashboard', 'prepare', 'research', 'contract'])
    args, remaining = parser.parse_known_args()
    if args.action == 'contract':
        options = argparse.ArgumentParser(description='Inspect the session capability contract without launching anything.')
        options.add_argument('--exploration-envs', type=int, default=1)
        options.add_argument('--isolated', action='store_true')
        chosen = options.parse_args(remaining)
        contract = Contract('auto-research', 'isolated' if chosen.isolated else 'exploration',
                            'official', chosen.exploration_envs)
        print(json.dumps(contract.manifest(), indent=2))
        return
    argv = command(args.action, remaining)
    os.chdir(PROJECT)
    os.execvp(argv[0], argv)


if __name__ == '__main__':
    main()
