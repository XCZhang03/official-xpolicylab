# RoboDojo MCP

The maintained agent-facing contract is [docs/ROBODOJO_MCP.md](docs/ROBODOJO_MCP.md).
It uses the official profile: `robodojo_observe`, `robodojo_status` and
`robodojo_step`, plus the harness lifecycle tools.

Related documents:

- [agent workspace boundary](docs/AGENT_WORKSPACE.md)
- [architecture and contract](docs/ARCHITECTURE_AND_CONTRACT.md)
- [auto-research infrastructure contract](docs/controller_backend.md)
- [control-frequency and cuRobo audit](docs/ROBODOJO_CONTROL_AUDIT.md)
- [official XPolicyLab mode](official/xpolicylab/README.md)

Sessions are prepared and launched from the dashboard
(`python3 scripts/robot_lab.py dashboard`) or with
`scripts/configure_auto_research.py` plus `scripts/start_auto_research_agent.sh`.
