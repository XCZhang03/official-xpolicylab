"""Generated held-out layouts keep the official schema and task rules."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'scripts'))
generate_layouts = pytest.importorskip('generate_layouts')
if not generate_layouts.official_layouts('plug_in_charger'):
    pytest.skip('RoboDojo assets are not installed', allow_module_level=True)


def keys(value):
    if isinstance(value, dict):
        return {k: keys(v) for k, v in value.items() if k not in ('default_pos', 'default_ori')}
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return [keys(value[0])]
    return type(value).__name__


@pytest.mark.parametrize('task', ['plug_in_charger', 'make_toast', 'make_kong'])
def test_schema_matches_official_and_is_seeded(task):
    generator = generate_layouts.Generator(task)
    first, again = generator.generate(7), generator.generate(7)
    assert first == again
    assert keys(first) == keys(generator.template)


def test_table_objects_stay_in_configured_regions():
    # The sampled point is the placement frame, a few centimetres from the object
    # origin, so official origins also fall slightly outside xlim/ylim.
    generator = generate_layouts.Generator('plug_in_charger')
    official = [json.loads(p.read_text()) for p in generator.sources]
    for layout in official + [generator.generate(seed) for seed in range(20)]:
        for label in ('charger', 'socket'):
            record = layout['Rigid'][label][0]
            assert generate_layouts.region(record).buffer(.03).contains(
                generate_layouts.box(*record['default_pos'][:2], *record['default_pos'][:2]))


def test_bread_follows_shelf():
    generator = generate_layouts.Generator('make_toast')
    template = generator.template
    shelf = template['Geometry']['bread_shelf'][0]
    offsets = [np.linalg.inv(generate_layouts.pose_matrix(shelf['default_pos'], shelf['default_ori']))
               @ generate_layouts.pose_matrix(b['default_pos'], b['default_ori']) for b in template['Rigid']['bread']]
    layout = generator.generate(3)
    shelf = layout['Geometry']['bread_shelf'][0]
    for bread, offset in zip(layout['Rigid']['bread'], offsets):
        placed = generate_layouts.pose_matrix(shelf['default_pos'], shelf['default_ori'])@offset
        assert np.allclose(placed[:3, 3], bread['default_pos'], atol=1e-9)


def test_kong_faces_follow_select_mode_rules():
    generator = generate_layouts.Generator('make_kong')
    for seed in range(20):
        face = {i['label']: i['category_idx'] for i in generator.generate(seed)['Rigid']['mahjong']}
        groups = [face[f'mahjong{g}_0'] for g in range(5)] + [face['other0'], face['other1'], face['other2']]
        assert len(set(groups)) == 8
        assert all(face[f'mahjong{g}_{k}'] == face[f'mahjong{g}_0'] for g in range(4) for k in range(3))
        assert all(face[f'mahjong{5+g}_0'] == face[f'mahjong{g}_0'] for g in range(4))
        assert face['mahjong4_1'] == face['mahjong9_0'] == face['mahjong4_0']


def test_refuses_to_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['generate_layouts.py', '--task', 'plug_in_charger', '--count', '1',
                                      '--output', str(tmp_path)])
    generate_layouts.main()
    assert json.loads((tmp_path/'manifest.json').read_text())['plug_in_charger']['stability_checked'] is False
    with pytest.raises(SystemExit):
        generate_layouts.main()


def test_refuses_tasks_with_clutter():
    with pytest.raises(ValueError, match='clutter'):
        generate_layouts.Generator('make_toast_random')
