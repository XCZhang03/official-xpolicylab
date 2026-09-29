---
name: motion-toolkit
description: Move the robot with the official step tools and the installed robodojo_toolkit package (cuRobo planning, bounded and native end-effector motion, pose math, kinematics, reply parsing). Use whenever you move the robot, manually or in a controller.
---

# Motion toolkit

The robot tools are exactly those of the official RoboDojo evaluation:
- `robodojo_observe` and `robodojo_status`: look;
- `robodojo_step`: 14-D absolute joint rows;
- `robodojo_step_ee`: 16-D link6 pose rows that the environment's own IK solves.

Each row is one 25 Hz environment step, and each call takes 1 to 50 rows. The
installed `robodojo_toolkit` package builds those rows and plans the motions. It is
the same version in your development Python, in rehearsal and formal runs, and on
the official policy server. Import it; do not copy it into a bundle.

Full signatures, return fields and step timing:
[references/api.md](references/api.md).

## Entry points

```python
import robodojo_toolkit as tk

meta, images = tk.parse_reply(ctx.call("robodojo_observe"))   # images: {camera: HxWx3 RGB}
state = meta["states"]            # [left j1..6, left grip, right j1..6, right grip]
target = {"position": [x, y, z], "quaternion_wxyz": [w, qx, qy, qz]}   # link6, env frame

# Free space: cuRobo plan (robot + table collision), executed in chunks of 50.
plan = tk.shared_planner().plan("left", state, target)
if plan["status"] == "Success":
    reply, meta = tk.execute(ctx, plan["actions"])

# Near objects, closed loop: bounded steps until the measured pose reaches the target.
reply, meta, info = tk.servo(ctx, "left", target, max_steps=40)

# Near objects, one call: bounded rows predicted open loop.
rows, _ = tk.eef_rows(state, [{"left": target}] * 10)
reply, meta = tk.execute(ctx, rows)

# Official native EEF actions: the environment's IK, one step per pose.
reply, meta = tk.execute(ctx, tk.ee_rows(meta, [{"left": p} for p in path]))

# Gripper, waiting, relative targets:
reply, meta = tk.execute(ctx, tk.set_gripper(meta["states"], "left", 0.0, count=10))
reply, meta = tk.execute(ctx, tk.hold(meta["states"], 5))
tk.pose_math({...})              # compose / relative poses, rotation conversions
tk.pose_error(meta, "left", target)   # measured (m, rad) error
```

## Which motion

- **Planner (`shared_planner().plan`):**
  - Use it for approach, transfer and retreat in free space.
  - It is the most accurate way to a pose (endpoint within 1 mm / 0.005 rad) and
    avoids robot self-collision and the table.
  - It does not see task objects or a held object, so plan to poses above or clear
    of them.
- **Bounded end-effector steps (`servo`, `step_eef`, `eef_rows`, `approach`):**
  - Use them to descend, insert, slide or lift near contact.
  - Each row moves at most 2 cm / 0.1 rad in pose (0.05 rad per joint), so motion
    stays smooth and predictable.
  - `servo` and `step_eef` re-solve from the measured state every step. Prefer them
    when accuracy matters.
- **Native EEF rows (`ee_rows`, `robodojo_step_ee`):**
  - The environment solves each pose with its own IK and has no step bound.
  - Accurate: official-stack smoke moves of 10–21 cm reached the goal within
    0.1 mm / 0.0003 rad, both as one repeated goal pose and as a 25-pose straight line.
  - A single far pose is covered in one step, with no collision check, and the IK
    may choose another arm configuration. Near objects, give a path of close poses
    (a few mm per row).
  - If IK fails, that arm keeps its previous target.
- **Gripper (`set_gripper`):** a closed command reads 0 in `states` even when an
  object blocks the jaws. Verify a grasp from images and a small test lift.

## Rules

- Take targets from current observations: `eef_*` are measured link6 poses. Objects
  move, and layouts change between episodes.
- Compose relative moves ("5 cm up", "rotate the wrist") with `tk.pose_math` from the
  measured pose. Never add quaternion components.
- Every row counts toward the step limit. Shorter chunks give more feedback, not more
  steps.
- Planned or predicted poses are not measurements. Check `pose_error` after moving.
- In a controller, call `tk.shared_planner()` once at the start of `main`. In
  development, all scripts share one warmed planner process, so only the first call
  in the container is slow.
- The official evaluation never reports the episode end: the next call raises
  instead. Do not wait for a success signal.
