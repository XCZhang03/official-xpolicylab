"""The rgb-position-calibration skill's self-contained triangulation helper (no GPU)."""
from pathlib import Path
import re
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[2]
SKILL = PROJECT / 'auto_research_agent/.agents/skills/rgb-position-calibration'
sys.path.insert(0, str(SKILL))

import triangulation as cameras  # noqa: E402
from triangulation import camera_pose, pixel_ray, project, ray_plane, triangulate  # noqa: E402
from services.robodojo.task_calibration import CAMERA_MODEL  # noqa: E402

# A right wrist camera looking down at the table from about 0.25 m.
LINK6 = np.array([0.15, -0.05, 1.05])
QUATERNION = Rotation.from_euler('xyz', [0, np.pi/2, 0]).as_quat()[[3, 0, 1, 2]]
META = {'eef_positions': [[-0.3, -0.2, 1.1], LINK6.tolist()],
        'eef_quaternions_wxyz': [[1, 0, 0, 0], QUATERNION.tolist()]}


def _matrices(text):
    rows = re.findall(r'\[([-\d.,\s]+)\]', text)
    return [float(x) for row in rows for x in row.split(',') if x.strip()]


def test_constants_match_the_calibration_reference_and_configs():
    numbers = _matrices('\n'.join(CAMERA_MODEL))
    for matrix in (cameras.K_HIGH, cameras.T_ENV_FROM_HIGH, cameras.K_WRIST, cameras.T_LINK6_FROM_WRIST):
        flat = matrix.reshape(-1).tolist()
        assert any(np.allclose(numbers[i:i+len(flat)], flat) for i in range(len(numbers)-len(flat)+1))
    assert cameras.K_HIGH[0, 0] == pytest.approx(10*640/22.212, abs=1e-4)
    assert cameras.K_WRIST[0, 0] == pytest.approx(13*640/20.955, abs=1e-4)
    for matrix in (cameras.T_ENV_FROM_HIGH, cameras.T_LINK6_FROM_WRIST):
        assert np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-4)


def test_wrist_pose_follows_measured_link6_and_needs_meta():
    with pytest.raises(ValueError):
        camera_pose('cam_left_wrist')
    pose = camera_pose('cam_right_wrist', META)
    assert np.allclose(pose[:3, 3], LINK6 + Rotation.from_quat(QUATERNION[[1, 2, 3, 0]]).apply(
        cameras.T_LINK6_FROM_WRIST[:3, 3]))


@pytest.mark.parametrize('camera', ['cam_high', 'cam_right_wrist'])
def test_ray_projection_and_plane_round_trip(camera):
    point = np.array([0.12, -0.02, 0.80])
    pixel, depth = project(camera, point, META)
    assert depth > 0 and 0 <= pixel[0] < 640 and 0 <= pixel[1] < 480
    origin, direction = pixel_ray(camera, pixel, META)
    assert np.linalg.norm(np.cross(point - origin, direction)) < 1e-6
    assert np.allclose(ray_plane(camera, pixel, 0.80, META), point, atol=1e-6)


def test_triangulation_recovers_an_elevated_point_without_height():
    point = np.array([0.12, -0.02, 0.83])
    views = [{'camera': c, 'pixel': project(c, point, META)[0], 'meta': META}
             for c in ('cam_high', 'cam_right_wrist')]
    result = triangulate(views)
    assert np.allclose(result['point'], point, atol=1e-6)
    assert result['max_ray_angle_deg'] > 15 and not result['warnings']
    # A table-plane assumption would misplace it by centimetres.
    assert np.linalg.norm(ray_plane('cam_high', views[0]['pixel'], 0.765) - point) > 0.05


def test_pixel_noise_and_wrong_correspondence_are_flagged():
    point = np.array([0.12, -0.02, 0.83])
    views = [{'camera': c, 'pixel': project(c, point, META)[0], 'meta': META}
             for c in ('cam_high', 'cam_right_wrist')]
    noisy = [dict(v, pixel=np.add(v['pixel'], [1.5, -1.0]).tolist()) for v in views]
    assert np.linalg.norm(np.subtract(triangulate(noisy)['point'], point)) < 0.01
    wrong = [views[0], dict(views[1], pixel=project('cam_right_wrist', point + [0.04, 0, 0], META)[0])]
    assert any('miss' in w or 'reprojection' in w for w in triangulate(wrong)['warnings'])


def test_degenerate_views_are_rejected_or_warned():
    point = np.array([0.12, -0.02, 0.80])
    view = {'camera': 'cam_high', 'pixel': project('cam_high', point)[0]}
    with pytest.raises(ValueError):
        triangulate([view])
    with pytest.raises(ValueError):
        triangulate([view, dict(view)])
    moved = {'eef_positions': META['eef_positions'], 'eef_quaternions_wxyz': META['eef_quaternions_wxyz']}
    moved['eef_positions'] = [META['eef_positions'][0], (LINK6 + [0.01, 0, 0]).tolist()]
    near = [{'camera': 'cam_right_wrist', 'pixel': project('cam_right_wrist', point, m)[0], 'meta': m}
            for m in (META, moved)]
    assert any('ray angle' in w for w in triangulate(near)['warnings'])


def test_quaternion_transform_matches_scipy_and_helper_needs_only_numpy():
    q = Rotation.from_euler('xyz', [0.3, -0.7, 1.9]).as_quat()[[3, 0, 1, 2]]
    assert np.allclose(cameras.transform([1, 2, 3], q)[:3, :3],
                       Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix(), atol=1e-12)
    imports = re.findall(r'^(?:from|import) (\w+)', (SKILL/'triangulation.py').read_text(), re.M)
    assert set(imports) <= {'__future__', 'itertools', 'json', 'sys', 'numpy'}


def test_command_line_prints_triangulation(tmp_path):
    import json, subprocess
    point = np.array([0.12, -0.02, 0.83])
    views = [{'camera': c, 'pixel': project(c, point, META)[0], 'meta': META}
             for c in ('cam_high', 'cam_right_wrist')]
    (tmp_path/'views.json').write_text(json.dumps(views))
    out = subprocess.run([sys.executable, str(SKILL/'triangulation.py'), str(tmp_path/'views.json')],
                         capture_output=True, text=True, check=True).stdout
    assert np.allclose(json.loads(out)['point'], point, atol=1e-6)
