#!/usr/bin/env python3
"""Pack a read-only reference copy of each official RoboDojo task for the agent workspace.

For every task config under RoboDojo/task/RoboDojo/config, write
runtime/task-sources/<task>/ with only what that task uses:

- source/: the task module whole, plus per-definition excerpts of exactly the RoboDojo
  code it depends on (its reward checks and their implementations, the official
  episode flow, framework methods it calls; see task_source_deps.py), its task
  config, the task-level defaults and the env configs it selects;
- assets/: the task's own objects only (categories its task config declares; no
  scene fixtures or random distractors): each model the config names or that
  appears in an official Eval_Layout (collections 0, 1, 2) of this task: metadata.json
  (bounding boxes, place/functional frames), description.json, the original
  object.usdz, and mesh.npz (triangles per link) plus articulation.json exported from it (metres, object
  frame) so the geometry is readable with NumPy alone;
- trajectories the task code loads from Assets/Traj, when small;
- README.md (rules and an object table) and MANIFEST.json (source commit, sha256).

Saved layouts themselves are never copied: they fix the test scenes' object poses.
The launcher copies runtime/task-sources/<task> into the new workspace as task_source/.

Run with a Python that has usd-core, numpy and pyyaml (runtime/envs/usd-tools):
    runtime/envs/usd-tools/bin/python scripts/build_task_sources.py [--task make_toast ...]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
ROBODOJO = PROJECT / "RoboDojo"
OBJECTS = ROBODOJO / "Assets/Object/RoboDojo"
LAYOUTS = ROBODOJO / "Assets/Eval_Layout/RoboDojo/arx_x5"
TRAJ = ROBODOJO / "Assets/Traj/RoboDojo"
CONFIGS = ROBODOJO / "task/RoboDojo/config"
OUTPUT = PROJECT / "runtime/task-sources"
COLLECTIONS = (0, 1, 2)
OBJECT_TYPES = ("Rigid", "Geometry", "Articulation", "Dynamic", "Garment", "Fluid")
MAX_TRAJ_BYTES = 64 * 1024 * 1024
SCHEMA = "robodojo_task_source_v1"


# ---------------------------------------------------------------- configs
def _yaml(path):
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def task_settings(task):
    defaults = _yaml(CONFIGS / "_task.yml")
    return {**defaults.get("common", {}), **(defaults.get("tasks", {}) or {}).get(task, {})}


def config_files(task):
    settings = task_settings(task)
    files = [CONFIGS / f"{task}.yml", CONFIGS / "_task.yml", ROBODOJO / "env_cfg/arx_x5.yml",
             ROBODOJO / "env_cfg/robot/_robot_info.json", ROBODOJO / "env_cfg/sim/sim_config.yml",
             ROBODOJO / f"env_cfg/robot/{settings.get('robot_config', 'dual_x5')}.yml",
             ROBODOJO / f"env_cfg/camera/{settings.get('camera_config', 'camera_config')}.yml",
             ROBODOJO / f"env_cfg/scene/{settings.get('scene_config', 'default')}.yml"]
    config = _yaml(CONFIGS / f"{task}.yml")
    return [f for f in dict.fromkeys(files) if f.is_file()], settings, config


# ---------------------------------------------------------------- objects
def layout_files(task):
    pattern = re.compile(rf"{re.escape(task)}_\d+\.json")
    for collection in COLLECTIONS:
        directory = LAYOUTS / str(collection)
        if directory.is_dir():
            yield from sorted(p for p in directory.iterdir() if pattern.fullmatch(p.name))


def task_categories(config):
    """{(type, category)} the task config itself declares: the task's own objects.

    Scene fixtures from env_cfg/scene (the camera stand) and random Clutter
    distractors are not task objects and are never packed.
    """
    return {(kind, category["name"]) for kind in OBJECT_TYPES
            for entry in config.get(kind, []) or [] for category in entry.get("category", []) or []}


def task_objects(task, config):
    """{(type, category, model_id): {labels, layouts}} for the task's own object categories.

    Models are those the config names by index plus every model of those categories
    that appears in an official Eval_Layout of this task.
    """
    declared = task_categories(config)
    objects = {}

    def add(kind, category, model, label=None, layout=False):
        row = objects.setdefault((kind, category, int(model)), {"labels": set(), "layout_count": 0})
        if label:
            row["labels"].add(label)
        row["layout_count"] += int(layout)

    count = 0
    for path in layout_files(task):
        count += 1
        layout = json.loads(path.read_text())
        for kind, categories in layout.items():
            if kind not in OBJECT_TYPES or not isinstance(categories, dict):
                continue
            for category, instances in categories.items():
                for inst in instances:
                    if inst.get("type") == "cluttered" or (kind, category) not in declared:
                        continue
                    add(kind, category, inst["category_idx"], inst.get("label"), layout=True)
    for kind in OBJECT_TYPES:
        for entry in config.get(kind, []) or []:
            labels = (entry.get("select_mode") or {}).get("label") or []
            for category in entry.get("category", []) or []:
                for model in category.get("index", []) or []:
                    add(kind, category["name"], model)
                for key, row in objects.items():
                    if key[:2] == (kind, category["name"]):
                        row["labels"].update(labels)
    return objects, count


def export_meshes(usd, out_dir):
    """mesh.npz (visual meshes per link, object frame, metres, triangles) and articulation.json."""
    from pxr import Usd, UsdGeom, UsdPhysics
    import numpy as np
    stage = Usd.Stage.Open(str(usd))
    scale = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    cache = UsdGeom.XformCache()
    groups, parts = {}, {}
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh) or "collision" in str(prim.GetPath()).lower():
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts:
            continue
        matrix = np.array(cache.GetLocalToWorldTransform(prim), dtype=float).T
        local = np.c_[np.asarray(points, dtype=float), np.ones(len(points))]
        world = (local @ matrix.T)[:, :3] * scale
        counts, indices = np.asarray(counts), np.asarray(indices)
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        triangles = [np.c_[np.full(n - 2, indices[s0]), indices[s0 + 1:s0 + n - 1], indices[s0 + 2:s0 + n]]
                     for s0, n in zip(starts, counts) if n >= 3]  # Fan-triangulate polygons.
        link = _link_name(prim, root)
        groups.setdefault(link, []).append(str(prim.GetPath()))
        vertices, faces = parts.setdefault(link, ([], []))
        offset = sum(len(v) for v in vertices)
        vertices.append(world.astype(np.float32))
        faces.append((np.concatenate(triangles) + offset).astype(np.int32))
    arrays = {}
    for link, (vertices, faces) in parts.items():
        arrays[f"{link}/vertices"] = np.concatenate(vertices)
        arrays[f"{link}/faces"] = np.concatenate(faces)
    if arrays:
        np.savez_compressed(out_dir / "mesh.npz", **arrays)
    bounds = [v for k, v in arrays.items() if k.endswith("/vertices")]
    joints = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.Joint) or prim.GetTypeName() == "PhysicsFixedJoint":
            continue
        joint = UsdPhysics.Joint(prim)
        attributes = {a.GetName(): a.Get() for a in prim.GetAttributes()}
        value = lambda key: (list(attributes[key]) if hasattr(attributes.get(key), "__len__")
                             and not isinstance(attributes[key], str) else attributes.get(key))
        joints.append({
            "name": prim.GetName(), "type": prim.GetTypeName().removeprefix("Physics").removesuffix("Joint").lower(),
            "parent": [str(p).rsplit("/", 1)[-1] for p in joint.GetBody0Rel().GetTargets()],
            "child": [str(p).rsplit("/", 1)[-1] for p in joint.GetBody1Rel().GetTargets()],
            "axis": value("physics:axis"), "lower": value("physics:lowerLimit"), "upper": value("physics:upperLimit"),
            "units": "degrees" if prim.GetTypeName() == "PhysicsRevoluteJoint" else "metres",
            "local_pos0": value("physics:localPos0"), "local_rot0_wxyz": _quat(attributes.get("physics:localRot0")),
            "drive": {k.split(":", 1)[1]: value(k) for k in attributes if k.startswith("drive:")},
        })
    everything = np.concatenate(bounds) if bounds else None
    summary = {"links": groups, "joints": joints, "meters_per_unit": scale,
               "mesh_bounds_min": everything.min(0).round(5).tolist() if everything is not None else None,
               "mesh_bounds_max": everything.max(0).round(5).tolist() if everything is not None else None}
    if joints:
        (out_dir / "articulation.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def _link_name(prim, root):
    while prim.GetParent() and prim.GetParent() != root:
        prim = prim.GetParent()
    return prim.GetName()


def _quat(value):
    if value is None:
        return None
    return [value.GetReal(), *value.GetImaginary()]


# ---------------------------------------------------------------- package
def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _extent(metadata):
    try:
        return metadata["geometry"]["aligned_bbox"]["extents"]
    except (KeyError, TypeError):
        return None


def _functional_tags(metadata):
    tags = []
    for side in ("active", "passive"):
        for kind, values in (metadata.get(side) or {}).items():
            if isinstance(values, dict):
                tags.extend(f"{side}.{kind}.{name}" for name in values)
    return tags


def readme(task, settings, instruction, objects, files, traj_note, clutter):
    rows = []
    for (kind, category, model), row in sorted(objects.items()):
        rows.append(f"| {', '.join(sorted(row['labels'])) or '—'} | {kind}/{category} | {model:05d} | "
                    f"{row.get('extent') or '—'} | {row['layout_count']} | "
                    f"{', '.join(row.get('tags', [])) or '—'} | `{row['path']}` |")
    return f"""# Task source: {task} (read-only reference)

This is the official RoboDojo source for **{task}**, packed for this task only:
the task module, excerpts of exactly the RoboDojo definitions it depends on, its
configs, and the 3D assets of the task's own objects. It is here so you can understand the task exactly and plan
motions: which objects exist, their shapes and sizes, articulated parts and joint
ranges, how the scene is randomized, and precisely how success and score are
computed.

## Rules

- **Read-only and reference-only.** Do not edit, run or import anything here. It is
  not part of the environment and is absent from isolated rehearsal, formal runs and
  the official evaluation. Nothing here can observe or change the live scene.
- **The controller must be closed-loop.** At run time it may use only the official
  observations (RGB images, joint state, link6 poses, instruction) and robot actions.
  Localize every object from images and robot feedback, act, then check the result in
  new observations. Never assume an object pose from this code.
- **Static facts may be copied.** Object dimensions, grasp-relevant geometry,
  functional frames (e.g. slot or button offsets), joint travel and the success
  thresholds are fixed properties of the task. You may copy such numbers into your
  bundle as constants or data files. Do not copy this folder into the bundle.
- **No test layouts.** The saved evaluation layouts are deliberately not included;
  the config's randomization ranges describe how scenes vary.

## Where to look

| What | Where |
|---|---|
| Task logic, step limit, success (`run_reward`) and score (`get_score`) | `source/task/RoboDojo/tasks/{task}.py` |
| Meaning of each check the task uses (`is_...`): argument defaults, then the implementation | `source/env/reward_manager/reward_manager.py`, then `func_parser.py` (excerpts) |
| Official episode flow: action stepping, step limit, when success and score are evaluated, episode end | `source/src/eval_client/eval_env.py` (excerpt) |
| Helpers those use (geometry, transforms) and framework methods the task calls | other files under `source/` (excerpts; the header of each names its original file) |
| Object categories, labels, placement ranges and randomization | `source/task/RoboDojo/config/{task}.yml` |
| Task-level defaults (evaluation count, robot/camera/scene config) | `source/task/RoboDojo/config/_task.yml`; this task: `{json.dumps(settings)}` |
| Robot, cameras, scene, simulation | `source/env_cfg/` |
| Object geometry | `assets/<Type>/<category>/<model>/`: `metadata.json` (aligned/oriented bounding boxes, `active`/`passive` place and functional frames in the object frame), `description.json` (captions), `mesh.npz` (visual triangle mesh per link, object frame, metres: `np.load(...)["<link>/vertices"]`, `["<link>/faces"]`, or `trimesh.Trimesh(v, f)`), `articulation.json` (joints: type, axis, limits, drives), `object.usdz` (original) |
{traj_note}
Instruction given to the policy: {json.dumps(instruction) if instruction else 'see gen_instruction in the task module'}

## Objects

Labels are the names the task code uses. "Layouts" counts appearances in the
official evaluation layouts (collections 0–2); 0 means named by the config only.

| Labels | Type/category | Model | Extents (m) | Layouts | Frames | Path |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}
{clutter}
`MANIFEST.json` lists every file with its sha256 and the RoboDojo commit it came from.
"""


def clutter_note(config):
    if not config.get("Clutter"):
        return ""
    return ("\nScenes also contain random distractor objects (`Clutter` in the task config). They are "
            "not task objects and their assets are not included.\n")


def instruction_of(path):
    match = re.search(r"def gen_instruction.*?return\s*\[\s*(\"[^\"]*\"|'[^']*')", path.read_text(), re.DOTALL)
    return ast.literal_eval(match.group(1)) if match else None


def build(task, output, commit):
    target = output / task
    temporary = output / f".{task}.partial"
    shutil.rmtree(temporary, ignore_errors=True)
    (temporary / "source").mkdir(parents=True)
    files, settings, config = config_files(task)
    from task_source_deps import collect
    for relative, text in collect(task).items():
        destination = temporary / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    for path in files:
        destination = temporary / "source" / path.relative_to(ROBODOJO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "_task.yml":  # Only the shared defaults and this task's entry.
            import yaml
            defaults = _yaml(path)
            entry = {"common": defaults.get("common", {}), "tasks": {task: (defaults.get("tasks") or {}).get(task) or {}}}
            destination.write_text(f"# Excerpt of RoboDojo task/RoboDojo/config/_task.yml: shared defaults and {task}.\n"
                                   + yaml.safe_dump(entry, sort_keys=False))
        else:
            shutil.copyfile(path, destination)
    if (ROBODOJO / "LICENSE").is_file():
        shutil.copyfile(ROBODOJO / "LICENSE", temporary / "LICENSE")
    objects, layout_count = task_objects(task, config)
    missing = []
    for (kind, category, model), row in objects.items():
        source = OBJECTS / kind / category / f"{model:05d}"
        relative = Path("assets") / kind / category / f"{model:05d}"
        row["path"] = str(relative)
        if not source.is_dir():
            missing.append(str(source.relative_to(ROBODOJO)))
            continue
        destination = temporary / relative
        destination.mkdir(parents=True)
        for name in ("metadata.json", "description.json"):
            if (source / name).is_file():
                shutil.copyfile(source / name, destination / name)
        metadata = json.loads((source / "metadata.json").read_text()) if (source / "metadata.json").is_file() else {}
        row["extent"], row["tags"] = _extent(metadata), _functional_tags(metadata)
        usd = next((source / n for n in ("object.usdz", "object.usd") if (source / n).is_file()), None)
        if usd is not None:
            shutil.copyfile(usd, destination / usd.name)
            try:
                export_meshes(usd, destination)
            except Exception as exc:  # Keep the original asset even if export fails.
                (destination / "EXPORT_ERROR.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    traj_note = ""
    task_code = (ROBODOJO / f"task/RoboDojo/tasks/{task}.py").read_text()
    if '"Traj"' in task_code and (TRAJ / task).is_dir():
        size = sum(p.stat().st_size for p in (TRAJ / task).rglob("*") if p.is_file())
        if size <= MAX_TRAJ_BYTES:
            shutil.copytree(TRAJ / task, temporary / "assets/Traj" / task)
            traj_note = (f"| Scripted trajectories the task code loads (`Assets/Traj`) | `assets/Traj/{task}/` |\n")
        else:
            traj_note = (f"| Scripted trajectories (`Assets/Traj/RoboDojo/{task}`, {size >> 20} MB) | "
                         "not included (too large); see how the task code loads them |\n")
    (temporary / "README.md").write_text(readme(task, settings, instruction_of(ROBODOJO / f"task/RoboDojo/tasks/{task}.py"),
                                                objects, files, traj_note, clutter_note(config)))
    entries = {str(p.relative_to(temporary)): _sha(p) for p in sorted(temporary.rglob("*")) if p.is_file()}
    manifest = {"schema": SCHEMA, "task": task, "robodojo_commit": commit, "layout_files_scanned": layout_count,
                "objects": [{"type": k[0], "category": k[1], "model": k[2], "labels": sorted(v["labels"]),
                             "layout_count": v["layout_count"], "path": v["path"]} for k, v in sorted(objects.items())],
                "missing_assets": missing, "files": entries,
                "bytes": sum((temporary / f).stat().st_size for f in entries)}
    (temporary / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.rmtree(target, ignore_errors=True)
    temporary.rename(target)
    return manifest


def tasks():
    return sorted(p.stem for p in CONFIGS.glob("*.yml") if not p.stem.startswith("_"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", action="append", help="Task to pack (repeatable); default: every task")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        import pxr  # noqa: F401
        import yaml  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"{exc}: run with runtime/envs/usd-tools/bin/python (usd-core, numpy, pyyaml)")
    commit = subprocess.check_output(["git", "-C", str(ROBODOJO), "rev-parse", "HEAD"], text=True).strip()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = args.task or tasks()
    unknown = set(selected) - set(tasks())
    if unknown:
        parser.error(f"Unknown tasks: {sorted(unknown)}")
    for task in selected:
        manifest = build(task, args.output, commit)
        print(f"{task}: {len(manifest['objects'])} objects, {len(manifest['files'])} files, "
              f"{manifest['bytes'] / 2**20:.1f} MB" + (f", missing {manifest['missing_assets']}" if manifest["missing_assets"] else ""),
              flush=True)


if __name__ == "__main__":
    main()
