"""Render trusted workspace templates from the same contract as the MCP endpoint.

Only call compose() on a newly copied workspace, before giving it to an agent.
Conditional content lives beside the original guidance, not in branch copies.
"""
import json
from pathlib import Path
import re

from services.mcp_contract import Contract

BLOCK = re.compile(
    r"<!-- capability:([a-z-]+) -->\n(.*?)<!-- otherwise -->\n(.*?)<!-- end-capability -->\n?",
    re.DOTALL,
)


def render(text: str, contract: Contract) -> str:
    capabilities = {"camera-geometry": contract.observation.geometry,
                    "multi-env": contract.environments > 1,
                    "official-tools": contract.observation.official}

    def choose(match):
        name, enabled, disabled = match.groups()
        if name not in capabilities:
            raise ValueError(f"Unknown workspace capability: {name}")
        return enabled if capabilities[name] else disabled

    rendered = BLOCK.sub(choose, text)
    if "<!-- capability:" in rendered or "<!-- end-capability" in rendered:
        raise ValueError("Malformed workspace capability block")
    return rendered


def describe(contract: Contract) -> str:
    interface = ("Robot tools are exactly those of the official XPolicyLab evaluation: "
                 "robodojo_observe, robodojo_status, robodojo_step (25 Hz absolute joint targets) and "
                 "robodojo_step_ee (25 Hz link6 pose targets solved by the environment's own IK). "
                 "Gripper values in states are the last commanded openings, as in official observations. "
                 "Planning, IK and pose math come from the installed robodojo_toolkit package, which the "
                 "official policy environment installs from the same wheels. gemini_generate (Gemini 3.8 "
                 "Flash, $10 per session) is the one model API; officially it is served only by our own "
                 "remote policy server. There is no preview or pixel-to-position tool.")
    perception = ("Only RGB images and robot state are supplied; observations carry no depth or camera "
                  "calibration. cam_high is fixed in the environment; the wrist cameras have fixed mounts "
                  "and intrinsics, and their world poses follow the measured link6 poses. The "
                  "[RGB position calibration](.agents/skills/rgb-position-calibration/SKILL.md) skill's "
                  "CALIBRATION.md gives their derived models (intrinsics and poses) and the task's table "
                  "geometry, and task_source/ gives object sizes. Combine these with feedback-guided "
                  "estimates; image pixels alone are not metric coordinates.")
    if contract.environments > 1:
        environments = (
            f"Exploration has {contract.environments} independent environments. Environment-scoped MCP "
            "calls require an explicit env_id; use the live schema. Starts and resets share one total "
            "episode budget. Calls serialize within an environment and can run concurrently across "
            "environments. Assign a different env_id to each active subagent. Use exploration_status for the "
            "environment inventory and shared budget; it is global and takes no env_id. Wait for "
            "outstanding actions before changing episode lifecycle. Rehearsal and formal use one "
            "environment and no env_id. In development Python, bind once with robot = ctx.env(i) and "
            "pass robot to main(robot) or your helper. Keep submitted main(ctx) independent of "
            "environment indices; isolated runners supply the unbound ctx. Use a separate Context "
            "connection per concurrent Python worker.")
    else:
        environments = "This context has one environment. Do not pass env_id to MCP calls."
    return (f"# Session MCP contract\n\nMode: {contract.mode}. Phase: {contract.phase}. "
            f"Observation profile: {contract.observation_profile}.\n\n{interface}\n\n"
            f"{perception}\n\n{environments}\n")


def write_contract(workspace, contract):
    workspace = Path(workspace)
    (workspace / "MCP_CONTRACT.json").write_text(json.dumps(contract.manifest(), indent=2) + "\n")
    (workspace / "MCP_SESSION.md").write_text(describe(contract))


def compose(workspace, contract, *, task):
    workspace = Path(workspace)
    skills = workspace / ".agents/skills"
    if not contract.observation.official:
        raise ValueError("The workspace templates target the official observation profile")
    if skills.is_symlink() or any(path.is_symlink() for path in skills.iterdir()):
        raise ValueError("Workspace templates must not contain symlinked skills")
    paths = list(workspace.glob("*.md"))
    for directory in (skills, workspace / "prompts"):
        if directory.is_dir():
            paths.extend(directory.rglob("*.md"))
    for path in paths:
        if path.is_symlink():
            raise ValueError("Workspace templates must not contain symlinked instructions")
        original = path.read_text()
        result = render(original, contract)
        if result != original:
            path.write_text(result)
    write_contract(workspace, contract)
    from services.robodojo.task_calibration import task_calibration_context
    calibration = skills / "rgb-position-calibration" / "CALIBRATION.md"
    calibration.write_text(task_calibration_context(task))
