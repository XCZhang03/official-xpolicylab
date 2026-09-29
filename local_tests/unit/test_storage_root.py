"""The artifact storage root is configurable, so the harness runs without /mnt/ssd8."""
from dataclasses import replace
from pathlib import Path

import pytest

from services import storage_root


def test_environment_selects_the_root(tmp_path, monkeypatch):
    monkeypatch.setenv(storage_root.ENVIRONMENT, str(tmp_path))
    assert storage_root.artifact_root() == tmp_path.resolve()
    assert storage_root.require_artifact_path(tmp_path/'session') == (tmp_path/'session').resolve()
    with pytest.raises(ValueError, match=storage_root.ENVIRONMENT):
        storage_root.require_artifact_path('/elsewhere/session')


def test_without_ssd_mount_the_runtime_link_target_is_used(monkeypatch):
    monkeypatch.delenv(storage_root.ENVIRONMENT, raising=False)
    real_is_dir = Path.is_dir
    monkeypatch.setattr(Path, 'is_dir', lambda p: False if str(p) == '/mnt/ssd8' else real_is_dir(p))
    assert storage_root.artifact_root() == (storage_root.PROJECT/'runtime').resolve()


def test_configuration_accepts_sessions_under_a_configured_root(tmp_path, monkeypatch):
    from test_controller_backend import configuration
    base = configuration(tmp_path)  # Built under the default root.
    monkeypatch.setenv(storage_root.ENVIRONMENT, str(tmp_path))
    config = replace(base, root=tmp_path/'auto-research'/'session')
    assert config.root == tmp_path/'auto-research'/'session'
    with pytest.raises(ValueError, match='Controller artifacts must be under'):
        replace(config, root=Path('/mnt/ssd8/elsewhere') if not str(tmp_path).startswith('/mnt/ssd8') else Path('/opt/x'))

