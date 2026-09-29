from pathlib import Path

import pytest


@pytest.fixture
def isaac_math_root():
    root = Path(__file__).resolve().parents[2]
    math = root / "RoboDojo/third_party/IsaacLab/source/isaaclab/isaaclab/utils/math.py"
    if not math.is_file():
        pytest.skip("Bootstrap the pinned RoboDojo submodules to test Isaac Lab pose math")
    return root
