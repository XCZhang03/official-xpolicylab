"""The agent workspace template holds only agent-visible material; trusted code lives outside."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = PROJECT_ROOT / "auto_research_agent"


def test_agent_workspace_contains_only_agent_visible_surface():
    visible = {path.name for path in AGENT_ROOT.iterdir() if path.name != "__pycache__"}
    assert visible == {".agents", "AGENTS.md", "START_PROMPT.md", "api"}
    assert not (AGENT_ROOT / "runtime").exists()
    files = [path for path in AGENT_ROOT.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    assert files
    # The only code is the client-side development Context and self-contained skill
    # helpers (NumPy only, observation inputs); no launchers or configs.
    assert sorted(p.relative_to(AGENT_ROOT) for p in files if p.suffix in {".py", ".sh", ".toml"}) == [
        Path(".agents/skills/rgb-position-calibration/triangulation.py"), Path("api/runtime.py")]


def test_trusted_harness_is_outside_agent_workspace():
    assert (PROJECT_ROOT / "harness/codex_cli/auto_research.py").is_file()
    assert (PROJECT_ROOT / "services/controller/frontend.py").is_file()
    assert (PROJECT_ROOT / "services/robodojo/mcp_server.py").is_file()
    assert (PROJECT_ROOT / "packages/robodojo_toolkit/pyproject.toml").is_file()
    for name in ("harness", "mcp", "docs", "services", "packages"):
        assert not (AGENT_ROOT / name).exists()
