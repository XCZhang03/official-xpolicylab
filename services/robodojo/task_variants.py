"""TASK.md context for RoboDojo `*_random` generalization variants.

Derived from the difference between the variant's task config and its base task's,
and from the variety of scene appearance in the saved evaluation layouts, so the
agent learns concretely what changes and which randomization types the RoboDojo
source applies (object instances, clutter, placement, room/table/ground).
"""
import json
from pathlib import Path

import yaml

from services.robodojo.source import source_root

SUFFIX = "_random"


def _load(root, task):
    path = Path(root) / "task/RoboDojo/config" / f"{task}.yml"
    return yaml.safe_load(path.read_text()) or {}


def _categories(config):
    """{(section, category name): [instance indices]} for every object group."""
    found = {}
    for section, groups in config.items():
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            for category in group.get("category", []) or []:
                if isinstance(category, dict) and category.get("name") is not None:
                    found.setdefault((section, str(category["name"])), []).extend(
                        category.get("index", []) or [])
    return found


def _placements(config):
    """Placement settings per object group, for detecting changed ranges."""
    keys = ("xlim", "ylim", "zlim", "rotate_rand", "rotate_deg")
    rows = {}
    for section, groups in config.items():
        if not isinstance(groups, list):
            continue
        for index, group in enumerate(groups):
            if isinstance(group, dict) and isinstance(group.get("common"), dict):
                rows[f"{section}[{index}]"] = {k: group["common"].get(k) for k in keys}
    return rows


# RoboDojo randomization types: config key -> (what it does, upstream implementation).
RANDOMIZATION_TYPES = {
    "placement": ("object positions sampled uniformly in `xlim`/`ylim` (`zlim`) per object group",
                  "env/scene_manager/layout_manager.py"),
    "rotation": ("yaw sampled in `rotate_deg` where `rotate_rand: True`",
                 "env/scene_manager/layout_manager.py"),
    "instances": ("object models drawn from each category's `index` list (`select_mode`)",
                  "env/scene_manager/layout_manager.py"),
    "clutter": ("distractor objects from `Clutter/clutter.yml`, placed with their own ranges, "
                "rotation and margin", "env/scene_manager/layout_manager.py"),
    "physics": ("ratio-based physics randomization (mass, density, linear/angular velocity) where an "
                "object's physics config gives ranges", "env/scene_manager/objects/articulation.py"),
    "room": ("room model", "env/scene_manager/layout_manager.py"),
    "table": ("table model", "env/scene_manager/layout_manager.py"),
    "ground": ("ground material", "env/scene_manager/objects/ground.py"),
    "background": ("background environment map", "env/scene_manager/layout_manager.py"),
}
APPEARANCE = {"room": ("Room", "default"), "table": ("Table", "default"),
              "ground": ("Ground", "materials"), "background": ("Background", "category_name")}
PLURAL = {"room": "room models", "table": "table models", "ground": "ground materials",
          "background": "background maps"}


def _appearance(root, task):
    """Distinct room/table/ground/background values over the saved layouts of collections 0-2."""
    seen = {name: set() for name in APPEARANCE}
    count = 0
    for collection in "012":
        directory = Path(root) / "Assets/Eval_Layout/RoboDojo/arx_x5" / collection
        for path in directory.glob(f"{task}_*.json"):
            if not path.stem[len(task) + 1:].isdigit():
                continue
            try:
                layout = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            count += 1
            for name, (section, key) in APPEARANCE.items():
                value = (layout.get(section) or {}).get(key)
                seen[name].add(json.dumps(value, sort_keys=True))
    return count, {name: len(values) for name, values in seen.items()}


def _config_types(config):
    """Randomization types a task config uses."""
    types = set()
    for section, groups in config.items():
        for group in groups if isinstance(groups, list) else []:
            if not isinstance(group, dict):
                continue
            common = group.get("common") if isinstance(group.get("common"), dict) else group
            if any(k in common for k in ("xlim", "ylim")):
                types.add("placement")
            if common.get("rotate_rand"):
                types.add("rotation")
            if group.get("category"):
                types.add("instances")
            if section == "Clutter":
                types.add("clutter")
            physics = common.get("physics") or group.get("physics")
            if isinstance(physics, dict) and any(isinstance(physics.get(k), list)
                                                 for k in ("mass", "density", "linear_velocity", "angular_velocity")):
                types.add("physics")
    return types


def task_variant_context(task, root=None):
    """Markdown for TASK.md; empty for tasks that are not `*_random` variants."""
    if not task.endswith(SUFFIX):
        return ""
    root = root or source_root()
    base = task[:-len(SUFFIX)]
    variant, standard = _load(root, task), _load(root, base)
    lines = [
        "",
        "## Generalization variant",
        "",
        f"`{task}` is the domain-randomized generalization variant of `{base}`. The",
        "official evaluation scores it as its own task, on its own held-out layouts.",
        "Expect scenes that differ from the standard task:",
        "",
    ]
    changed = []
    variant_objects, standard_objects = _categories(variant), _categories(standard)
    for (section, name), indices in sorted(variant_objects.items()):
        before = standard_objects.get((section, name))
        if before is None:
            changed.append(f"- **{name}** ({section}): a different object category than the "
                           f"standard task, instances {sorted(set(indices))}.")
        elif sorted(set(before)) != sorted(set(indices)):
            changed.append(f"- **{name}** ({section}): other object instances, "
                           f"{sorted(set(indices))} instead of the standard {sorted(set(before))}. "
                           "Their shape, size, colour and functional geometry can differ, "
                           "so identify them by function, not by appearance learned elsewhere.")
    clutter = [g for g in variant.get("Clutter", []) or [] if isinstance(g, dict)]
    if clutter:
        count = sum(int(g.get("nums", 0) or 0) for g in clutter)
        changed.append(f"- **Clutter:** about {count} random distractor objects are scattered on the "
                       "table. They are not task objects: never target them, plan around them, and "
                       "make perception robust to look-alike colours and shapes and to occlusion.")
    before, after = _placements(standard), _placements(variant)
    moved = sorted(k for k in after if k in before and after[k] != before[k])
    if moved:
        changed.append(f"- **Placement ranges** differ for {', '.join(moved)}; read them in the "
                       "task config.")
    if variant.get("ProhibitedArea") != standard.get("ProhibitedArea"):
        changed.append("- **Keep-out areas** (`ProhibitedArea`) differ, which shifts where objects "
                       "and clutter can appear.")
    lines += changed or ["- Object and scene settings differ from the standard task; compare the "
                         "two task configs."]
    layouts, variant_look = _appearance(root, task)
    _, standard_look = _appearance(root, base)
    varied = [name for name in APPEARANCE if variant_look[name] > max(1, standard_look[name])]
    if varied:
        lines.append("- **Scene appearance (domain randomization):** across this variant's "
                     f"{layouts} saved evaluation layouts, "
                     + ", ".join(f"{variant_look[n]} different {PLURAL[n]}"
                                 f" (standard: {standard_look[n]})" for n in varied)
                     + ". Colours, textures and brightness of the table and surroundings change "
                     "between episodes: never segment by a fixed table or background colour."
                     + (" The table's pose and size stay the same (top at 0.765 m), so metric "
                        "calibration still holds; only its look changes." if "table" in varied else ""))
    lines.append("- **Official domain randomization** (RoboDojo documentation) can vary five aspects "
                 "per episode: cluttered tabletop objects, table material, floor material, lighting "
                 "(type, intensity, colour temperature) and the background environment map. Expect any "
                 "of them, even those not visible in the counts above. Do not depend on exact colours, "
                 "brightness or a fixed scene look.")
    types = _config_types(variant) | set(varied)
    lines += ["", "Randomization types present in the RoboDojo source for this task (implemented "
              "upstream; the code is not in `task_source/`):", ""]
    lines += [f"- `{name}`: {RANDOMIZATION_TYPES[name][0]} ({RANDOMIZATION_TYPES[name][1]})"
              for name in RANDOMIZATION_TYPES if name in types]
    lines += [
        "",
        f"Develop and validate this controller on this variant's own scenes. The task config "
        f"(`task_source/source/task/RoboDojo/config/{task}.yml`) has the exact settings.",
        "",
    ]
    return "\n".join(lines)
