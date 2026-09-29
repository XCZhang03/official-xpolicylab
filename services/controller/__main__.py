"""Operator-only backend CLI. Never expose Docker or this CLI to generated code."""
import argparse
import json
import os
from pathlib import Path

from .backend import NativeBackend
from .config import Configuration
from .gemini import GeminiRouter, load_key
from .sandbox import DockerSandbox
from .supervisor import Supervisor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Trusted operator JSON, outside agent workspace")
    parser.add_argument("--openrouter-key-file", type=Path,
                        help="Private host-only key file; otherwise use OPENROUTER_API_KEY")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    commands.add_parser("status")
    registration = commands.add_parser("register")
    registration.add_argument("--source", type=Path, required=True)
    registration.add_argument("--workspace-path", default="code/controller",
                              help="Preserved path relative to /workspace, under code/")
    for name in ("rehearse", "formal"):
        commands.add_parser(name).add_argument("--bundle", required=True)
    args = parser.parse_args()
    config = Configuration.from_dict(json.loads(args.config.read_text()))
    key = load_key(args.openrouter_key_file)
    if args.command == "check":
        DockerSandbox(config.image, config.development).preflight()
        print(json.dumps({"docker": "ready", "image": config.image,
                          "gemini_key_configured": bool(key),
                          "native_python": os.environ.get("ROBODOJO_PYTHON", "current interpreter")}))
        return
    supervisor = Supervisor(config, lambda: NativeBackend(config), GeminiRouter(key) if key else None)
    try:
        if args.command == "status":
            result = {k: supervisor.state[k] for k in ("exploration_started", "interactive_success", "formal_reserved", "active", "results", "usage")}
        elif args.command == "register":
            ident, manifest = supervisor.register(args.source, workspace_path=args.workspace_path)
            result = {"bundle": ident, "manifest": manifest}
        elif args.command in {"rehearse", "formal"}:
            result = supervisor.run(args.bundle, formal=args.command == "formal")
        print(json.dumps(result, indent=2))
    finally:
        supervisor.close()


if __name__ == "__main__":
    main()
