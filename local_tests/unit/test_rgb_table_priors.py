"""Guard agent-facing table priors against changes to the installed layouts."""
import json
from pathlib import Path

import numpy as np
import pytest
from services.robodojo.task_calibration import (
    STANDARD_TABLE_TASKS, NO_TABLE_TASKS, table_geometry, task_calibration_context,
)

PROJECT = Path(__file__).resolve().parents[2]


def test_calibration_guidance_is_task_independent():
    copies = [(PROJECT / "auto_research_agent/.agents/skills/rgb-position-calibration/SKILL.md").read_text()]
    for task in STANDARD_TABLE_TASKS | NO_TABLE_TASKS | {"put_bottles_into_dustbin"}:
        assert task not in copies[0].lower()
    assert "0.765" not in copies[0]
    assert "CALIBRATION.md" in copies[0]


@pytest.mark.parametrize("task", sorted(STANDARD_TABLE_TASKS | NO_TABLE_TASKS
                                      | {"put_bottles_into_dustbin", "unreviewed_task"}))
def test_injected_reference_is_task_scoped(task):
    text = task_calibration_context(task)
    assert f"Task: `{task}`" in text
    # Task identifiers occur only in the selected-task header, never a catalogue.
    body = text.split(f"Task: `{task}`", 1)[1]
    assert not any(other in body for other in STANDARD_TABLE_TASKS | NO_TABLE_TASKS
                   | {"put_bottles_into_dustbin"})
    if table_geometry(task):
        assert "0.765" in body and "| x-min, y-min |" in body
    else:
        assert "0.765" not in body and "| x-min, y-min |" not in body


@pytest.mark.upstream
def test_installed_table_layouts_match_documented_priors():
    root = PROJECT / "RoboDojo/Assets/Eval_Layout/RoboDojo/arx_x5"
    if not root.is_dir():
        pytest.skip("Installed RoboDojo evaluation layouts are unavailable")
    layouts = list(root.rglob("*.json"))
    assert layouts, "No layouts were checked"
    for path in layouts:
        task = path.stem.rsplit("_", 1)[0]
        table = json.loads(path.read_text()).get("Table")
        if task.lower() in NO_TABLE_TASKS:
            assert table is None, path
            continue
        assert table is not None, path
        dustbin = task == "put_bottles_into_dustbin"
        geometry = table_geometry(task)
        assert geometry is not None, path
        centre, size = geometry
        assert table["default_pos"] == list(centre), path
        assert table["scale"] == list(size), path
        assert table["default_ori"] == [1, 0, 0, 0], path
        assert table.get("static", True) is True, path
        corners = np.array([
            [centre[0] + sx * size[0] / 2, centre[1] + sy * size[1] / 2,
             centre[2] + size[2] / 2]
            for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        ])
        expected = (
            [[-0.41, -0.60, 0.765], [0.59, -0.60, 0.765],
             [0.59, 0.50, 0.765], [-0.41, 0.50, 0.765]] if dustbin else
            [[-0.70, -0.60, 0.765], [0.70, -0.60, 0.765],
             [0.70, 0.50, 0.765], [-0.70, 0.50, 0.765]])
        np.testing.assert_allclose(corners, expected, err_msg=str(path))
