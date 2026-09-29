"""Agent documentation must be usable without internet; retain legal notices."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]/'auto_research_agent'
# Written into the deployed workspace from the selected task (services.mcp_workspace.compose).
GENERATED = {ROOT/'.agents/skills/rgb-position-calibration/CALIBRATION.md'}


def test_agent_authored_files_have_no_web_references():
    for path in ROOT.rglob('*'):
        if not path.is_file() or path.suffix not in {'.md', '.py', '.txt', '.yaml'}:
            continue
        if 'licenses' in path.relative_to(ROOT).parts:
            continue  # Do not alter third-party license texts, including their URLs.
        assert not re.search(r'https?://|www\.', path.read_text()), path


def test_stage_and_skill_entry_links_resolve():
    documents = [*ROOT.glob('*.md'), *ROOT.glob('.agents/skills/**/*.md')]
    assert documents
    for document in documents:
        # Templates contain example links to files the agent will create later.
        prose = re.sub(r'^```[^\n]*\n.*?^```\s*$', '', document.read_text(), flags=re.M | re.S)
        for target in re.findall(r'\]\(([^)]+)\)', prose):
            assert '://' not in target, (document, target)
            relative = target.split('#', 1)[0]
            path = document.parent/relative if relative else document
            assert path.is_file() or path.resolve() in GENERATED, (document, target)


def test_retired_workflows_are_not_referenced():
    for document in ROOT.rglob('*.md'):
        text = document.read_text()
        # Gemini is back as gemini_generate (host-brokered; official via a remote policy server).
        assert not re.search(r'teacher|student|direct.control|robot_preview|policies/', text, flags=re.I), document
