#!/usr/bin/env python3
"""Trusted operator: generate novel saved layouts for a held-out collection.

Upstream RoboDojo ships only saved layouts, not the generator that produced them.
This script resamples what a task's config leaves random and copies everything
else from an existing official layout of the same task:

- Table objects are placed by upstream's own ``ClutteredGenerator``, with each
  record's saved sampling fields (xlim/ylim/zlim, qpos, rotate_deg, place_tag,
  margin, check_mode), so footprints never overlap or leave the table.
- Model choices come from the task config's ``category.index`` lists.
- Objects placed relative to another object keep their official constant offset.
- make_kong keeps its fixed tile poses and resamples tile faces under its
  select_mode rules.

Physical stability is not checked here; the native simulator rejects unstable
layouts when they load. The output directory contains ``<task>_<n>.json`` files
and a ``manifest.json`` recording seeds and sources.
"""
import argparse
from copy import deepcopy
import glob
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
import transforms3d as t3d
import yaml

PROJECT = Path(__file__).resolve().parents[1]
ROBODOJO = PROJECT/'RoboDojo'
sys.path.insert(0, str(ROBODOJO))
from utils.cluttered_generator import ClutteredGenerator  # noqa: E402

LAYOUTS = ROBODOJO/'Assets/Eval_Layout/RoboDojo/arx_x5'
OBJECTS = ROBODOJO/'Assets/Object/RoboDojo'
TASK_CONFIGS = ROBODOJO/'task/RoboDojo/config'
OBJECT_TYPES = ('Rigid', 'Dynamic', 'Geometry', 'Articulation', 'Garment', 'Fluid')
OFFICIAL_COLLECTIONS = ('0', '1', '2')
# Table inset used by upstream LayoutManager.cluttered_generator_init.
TABLE_INSET = 0.05


def official_layouts(task):
    pattern = re.compile(rf'{re.escape(task)}_\d+\.json')
    paths = [Path(p) for c in OFFICIAL_COLLECTIONS for p in glob.glob(str(LAYOUTS/c/f'{task}_*.json'))]
    return sorted(p for p in paths if pattern.fullmatch(p.name))


def records(layout):
    for kind in OBJECT_TYPES:
        for category, instances in (layout.get(kind) or {}).items():
            for instance in instances:
                yield kind, category, instance


def pose_matrix(position, quaternion):
    matrix = np.eye(4)
    matrix[:3, :3] = t3d.quaternions.quat2mat(quaternion)
    matrix[:3, 3] = position
    return matrix


def matrix_pose(matrix):
    quaternion = t3d.quaternions.mat2quat(matrix[:3, :3])
    return matrix[:3, 3].tolist(), (quaternion*np.sign(quaternion[0] or 1)).tolist()


def region(instance):
    """Allowed XY region; several x ranges form a union, as in the task configs."""
    xlim, ylim = instance['xlim'], instance['ylim']
    ranges = xlim if isinstance(xlim[0], (list, tuple)) else [xlim]
    return unary_union([box(x0, ylim[0], x1, ylim[1]) for x0, x1 in ranges])


def table_generator(template):
    table = template['Table']
    (cx, cy, top), (sx, sy, sz) = table['default_pos'], table['scale']
    container = box(cx-sx/2+TABLE_INSET, cy-sy/2+TABLE_INSET, cx+sx/2-TABLE_INSET, cy+sy/2-TABLE_INSET)
    return ClutteredGenerator(global_container=container, frame=np.array([0., 0., top+sz/2, 1., 0., 0., 0.]))


def config_order(task):
    """Labels in task-config file order (parents before children), with candidate model indices."""
    config = yaml.safe_load((TASK_CONFIGS/f'{task}.yml').read_text())
    if 'Clutter' in config or 'ProhibitedArea' in config:
        raise ValueError(f'{task} places clutter, which this generator does not resample yet')
    order, choices = [], {}
    for kind, groups in config.items():
        if kind not in OBJECT_TYPES:
            continue
        for group in groups or []:
            indices = [i for category in group['category'] for i in category.get('index', [0])]
            for label in group['select_mode']['label']:
                order.append(label)
                choices[label] = indices
    return order, choices


class Generator:
    def __init__(self, task):
        self.task = task
        self.sources = official_layouts(task)
        if not self.sources:
            raise ValueError(f'No official layouts for {task}')
        self.template_path = self.sources[0]
        self.template = json.loads(self.template_path.read_text())
        self.order, self.choices = config_order(task)
        # Model-specific fields (physics, sizes) of every official record, by model.
        self.by_model = {}
        for path in self.sources:
            for kind, category, instance in records(json.loads(path.read_text())):
                self.by_model.setdefault((kind, category, instance['category_idx']), instance)
        self.metadata = {}

    def object_metadata(self, kind, category, index):
        key = (kind, category, index)
        if key not in self.metadata:
            path = OBJECTS/kind/category/f'{index:05d}'/'metadata.json'
            self.metadata[key] = json.loads(path.read_text())
        return self.metadata[key]

    def with_model(self, kind, category, instance, index):
        source = self.by_model.get((kind, category, index))
        if source is None:
            raise ValueError(f'No official record of {category} model {index}')
        result = deepcopy(instance)
        for key, value in source.items():
            if key not in ('label', 'default_pos', 'default_ori', 'group'):
                result[key] = deepcopy(value)
        result['category_idx'] = index
        return result

    def generate(self, seed, attempts=200):
        for _ in range(attempts):
            random.seed(seed)
            np.random.seed(seed % 2**32)
            layout = self.kong(seed) if self.task == 'make_kong' else self.place()
            if layout is not None:
                return layout
            seed += 1_000_000_007
        raise RuntimeError(f'Could not place {self.task} with seed {seed}')

    def place(self):
        layout = deepcopy(self.template)
        entries = {instance['label']: (kind, category, instance) for kind, category, instance in records(layout)}
        original = {label: deepcopy(entry[2]) for label, entry in entries.items()}
        generator = table_generator(layout)
        for label in self.order:
            kind, category, instance = entries[label]
            index = int(np.random.choice(self.choices[label]))
            if index != instance['category_idx']:
                instance.clear()
                instance.update(self.with_model(kind, category, original[label], index))
            plane = instance['relative_plane']
            if plane == 'Table':
                ok, world, _ = generator.add_model_from_config(
                    self.object_metadata(kind, category, index), place_tag=instance.get('place_tag'),
                    rotate_rand=instance['rotate_rand'], rotate_deg=instance['rotate_deg'],
                    margin=instance['margin'], name=label, allowed_region=region(instance),
                    zlim=instance.get('zlim'), qpos=instance.get('qpos'), check_mode=instance['check_mode'])
                if not ok:
                    return None
                instance['default_pos'], instance['default_ori'] = list(map(float, world[:3])), list(map(float, world[3:]))
            else:
                # Keep the official constant offset from the parent object.
                parent_label = plane.split('/')[0]
                parent_old, parent_new = original[parent_label], entries[parent_label][2]
                local = np.linalg.inv(pose_matrix(parent_old['default_pos'], parent_old['default_ori'])) \
                    @ pose_matrix(original[label]['default_pos'], original[label]['default_ori'])
                instance['default_pos'], instance['default_ori'] = matrix_pose(
                    pose_matrix(parent_new['default_pos'], parent_new['default_ori'])@local)
        return layout

    def kong(self, seed):
        """Fixed tile poses; resample faces under the make_kong select_mode rules."""
        layout = deepcopy(self.template)
        faces = sorted({key[2] for key in self.by_model if key[1] == 'mahjong'})
        chosen = random.sample(faces, 8)   # 4 triples, 1 pair, 3 face-down stack tiles.
        face = {f'mahjong{g}_{k}': chosen[g] for g in range(4) for k in range(3)}
        face.update({'mahjong4_0': chosen[4], 'mahjong4_1': chosen[4],
                     'other0': chosen[5], 'other1': chosen[6], 'other2': chosen[7],
                     'mahjong9_0': chosen[4]})   # The drawn tile completes the pair.
        face.update({f'mahjong{5+g}_0': chosen[g] for g in range(4)})   # Opponent tiles match triples.
        instances = layout['Rigid']['mahjong']
        # Within a triple, labels permute across the three fixed slots.
        for g in range(4):
            group = [i for i in instances if i['label'].startswith(f'mahjong{g}_')]
            slots = [(i['default_pos'], i['default_ori']) for i in group]
            random.shuffle(slots)
            for instance, (position, orientation) in zip(group, slots):
                instance['default_pos'], instance['default_ori'] = position, orientation
        for n, instance in enumerate(instances):
            updated = self.with_model('Rigid', 'mahjong', instance, face[instance['label']])
            updated['default_pos'], updated['default_ori'] = instance['default_pos'], instance['default_ori']
            instances[n] = updated
        return layout


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Collection directory for the generated files')
    parser.add_argument('--seed', type=int, default=3_000_000, help='Seed of layout 0; layout n uses seed+n')
    args = parser.parse_args()
    generator = Generator(args.task)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if args.task in manifest:
        raise SystemExit(f'{args.task} already generated in {args.output}; use a fresh collection')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, capture_output=True, text=True).stdout.strip()
    rows = []
    for n in range(args.count):
        layout = generator.generate(args.seed+n)
        path = args.output/f'{args.task}_{n}.json'
        with path.open('x') as stream:
            json.dump(layout, stream, indent=1)
        rows.append({'file': path.name, 'seed': args.seed+n,
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest[args.task] = {'generator': 'scripts/generate_layouts.py', 'commit': commit,
                           'template': str(generator.template_path.relative_to(LAYOUTS)),
                           'stability_checked': False, 'layouts': rows}
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f'Wrote {args.count} {args.task} layouts to {args.output}')


if __name__ == '__main__':
    main()
