from pathlib import Path

import pytest

from services.robodojo.task_rubrics import task_rubric_context, _snapshot
from scripts.fetch_task_rubrics import parse_scoring


@pytest.mark.upstream
def test_public_snapshot_covers_all_installed_tasks():
    root = Path(__file__).resolve().parents[2]
    tasks = [p.stem for p in (root / "RoboDojo/task/RoboDojo/config").glob("*.yml")
             if not p.stem.startswith("_")]
    assert len(_snapshot()["tasks"]) == 42
    if not tasks:
        pytest.skip("Bootstrap the pinned RoboDojo checkout to check task coverage")
    for task in tasks:
        text = task_rubric_context(task)
        assert "https://robodojo-benchmark.com/doc/sim-tasks/" in text
        assert "| Published score | Condition |" in text


def test_variants_and_unknown_tasks():
    assert "base task's published rubric" in task_rubric_context("push_T_random")
    assert "/play-xylophone/" in task_rubric_context("play_Xylophone")
    with pytest.raises(ValueError, match="No reviewed public scoring rubric"):
        task_rubric_context("unknown_task")


@pytest.mark.parametrize("task", ["make_kong", "make_kong_random"])
@pytest.mark.parametrize("include_source_url", [True, False])
def test_make_kong_demo_clarification_is_in_shared_rubric(task, include_source_url):
    context = task_rubric_context(task, include_source_url=include_source_url)
    assert 'Use the supplied demonstration images to identify the "target tile" in the rubric, which is moved from the left pile to stand upright at the right end of the front row, next to its rightmost tile, with its symbol oriented the same way.' in context
    assert "Demo clarification:" in context
    for row in _snapshot()["tasks"]["make_kong"]["criteria"]:
        assert f"| {row['points']} | {row['condition']} |" in context
    assert "Demo clarification:" not in task_rubric_context("plug_in_charger")


def test_offline_rubric_preserves_criteria_without_external_references():
    for task in _snapshot()["tasks"]:
        public = task_rubric_context(task)
        offline = task_rubric_context(task, include_source_url=False)
        assert "https://" not in offline
        assert "Source: bundled public wiki snapshot" in offline
        assert public.split("| Published score |")[1] == offline.split("| Published score |")[1]


@pytest.mark.parametrize("task", ["make_toast", "make_toast_random"])
@pytest.mark.parametrize("include_source_url", [True, False])
def test_toast_details_supplement_published_table(task, include_source_url):
    context = task_rubric_context(task, include_source_url=include_source_url)
    for detail in ("one bread slice in each toaster slot", "insertion depth and orientation",
                   "all four slices upright", "exactly two slices on the shelf",
                   "lever fully down", "ensure it stays down",
                   "after withdrawing the gripper and returning both arms home"):
        assert detail in context
    for row in _snapshot()["tasks"]["make_toast"]["criteria"]:
        assert f"| {row['points']} | {row['condition']} |" in context


@pytest.mark.parametrize("include_source_url", [True, False])
def test_pour_demo_checkpoints_supplement_published_table(include_source_url):
    context = task_rubric_context("pour_by_language", include_source_url=include_source_url)
    assert "Complete the language-specified pours in order" in context
    assert "return that bottle upright, then return both robot arms to their initial poses" in context
    assert "each pour, including the final one" in context
    assert "Both arms must leave their home regions" in context
    assert "Tilt each bottle far enough and hold it tilted long enough" in context
    assert "Aim to nearly fill the bowl without spilling" in context
    assert "bottle is nearly empty; some visible liquid in the bowl is not enough" in context
    assert "outside those regions at the same time" in context
    assert "raise the idle arm more than 15 cm above its initial end-effector height" in context
    for row in _snapshot()["tasks"]["pour_by_language"]["criteria"]:
        assert f"| {row['points']} | {row['condition']} |" in context


@pytest.mark.parametrize("include_source_url", [True, False])
def test_swap_details_supplement_published_table(include_source_url):
    context = task_rubric_context("swap_blocks", include_source_url=include_source_url)
    for detail in ("Either block may move first", "more than 3 cm", "less than 3 cm apart in 3D",
                   "above 95% to below 50%", "above 90%", "fourth counted press fails",
                   "both arms", "15 cm on each position axis and 20 degrees", "not a return-home transition"):
        assert detail in context
    for row in _snapshot()["tasks"]["swap_blocks"]["criteria"]:
        assert f"| {row['points']} | {row['condition']} |" in context


@pytest.mark.parametrize("include_source_url", [True, False])
@pytest.mark.parametrize("task,details", [
    ("pour_by_language", ("return that bottle upright", "each pour, including the final one",
                          "Tilt each bottle far enough and hold it tilted long enough",
                          "raise the idle arm more than 15 cm above its initial end-effector height")),
    ("swap_blocks", ("more than 3 cm", "above 95% to below 50%", "not a return-home transition")),
    ("make_toast", ("one bread slice in each toaster slot", "lever fully down", "ensure it stays down")),
    ("make_toast_random", ("one bread slice in each toaster slot", "lever fully down", "ensure it stays down")),
])
def test_task_clarifications_reach_offline_and_public_rubrics(include_source_url, task, details):
    # The launcher injects exactly this context into TASK.md (see test_auto_research_frontend).
    text = task_rubric_context(task, include_source_url=include_source_url)
    for detail in details:
        assert detail in text


def test_wiki_parser_extracts_only_scoring_table():
    html = '<table><tr><th>Private-looking irrelevant text</th></tr></table>'
    html += '<h2 id="scoring">Scoring</h2><table><tr><th>Score</th><td>Condition</td></tr>'
    html += '<tr><th>100</th><td>Place <strong>all</strong> blocks &amp; return.</td></tr></table>'
    assert parse_scoring(html) == [{"points": "100", "condition": "Place all blocks & return."}]
