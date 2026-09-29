"""Exercise the upstream PR48 methods without importing Isaac Sim."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_drive_reset_patch_is_pinned_and_installed_by_bootstrap():
    root = Path(__file__).resolve().parents[2]
    patch = root/'patches/robodojo-pr48-drive-reset.patch'
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    assert f'ROBODOJO_DRIVE_RESET_PATCH_SHA256={digest}' in (root/'dependencies.lock').read_text()
    bootstrap = (root/'scripts/bootstrap_sources.sh').read_text()
    assert 'patches/robodojo-pr48-drive-reset.patch' in bootstrap
    assert '"$ROBODOJO_DRIVE_RESET_PATCH_SHA256"' in bootstrap


@pytest.fixture
def articulation():
    path = Path(__file__).resolve().parents[2]/'RoboDojo/env/scene_manager/objects/articulation.py'
    if not path.exists():
        pytest.skip('Requires the bootstrapped RoboDojo checkout')
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ArticulationObject')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_dof_drive_targets')
    scope = {'np':np, 'get_prim_at_path':lambda p:SimpleNamespace(IsValid=lambda:True)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    instance = SimpleNamespace(dof_names=['slide','hinge'],
        _get_joint_drive_type=lambda p:'angular' if p.angular else 'linear')
    scope['get_prim_at_path'] = lambda p:SimpleNamespace(IsValid=lambda:True, angular=p.endswith('hinge'))
    return instance, scope['_dof_drive_targets'], scope


def test_targets_follow_dof_order_and_convert_angular_units(articulation):
    obj, convert, _ = articulation
    np.testing.assert_allclose(convert(obj, {'/joint/hinge':180, '/joint/slide':.01}), [.01,np.pi])


@pytest.mark.parametrize('targets', [
    {'/joint/slide':.01},
    {'/joint/slide':None, '/joint/hinge':0},
    {'/one/slide':.01, '/two/slide':.02, '/joint/hinge':0},
])
def test_ambiguous_or_missing_targets_do_not_override_runtime(articulation, targets):
    obj, convert, _ = articulation
    assert convert(obj, targets) is None


def test_invalid_prim_does_not_override_runtime(articulation):
    obj, convert, scope = articulation
    scope['get_prim_at_path'] = lambda p:None
    assert convert(obj, {'/joint/slide':.01, '/joint/hinge':0}) is None
