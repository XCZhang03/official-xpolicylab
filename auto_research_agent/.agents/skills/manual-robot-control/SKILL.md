---
name: manual-robot-control
description: Solve RoboDojo tasks by choosing robot motions yourself from observations during exploration, using the official robot tools and robodojo_toolkit. Use for the manual-success stage, recovery and preparing scenes for script tests; not for isolated rehearsal or formal submission.
---

# Manual robot control

First complete the task yourself. Read `TASK.md` and use current observations to
satisfy every condition for 100 points. Confirmed manual success leads to
autonomous development (AGENTS.md), not to a formal episode.

## Look with tools, move with Python

- **Look:** call `robodojo_observe` (images come back to you directly) or
  `robodojo_status` (state only).
- **Move:** in short Python scripts, build rows with
  [motion-toolkit](../motion-toolkit/SKILL.md) and send them with
  `tk.execute(ctx, rows)`. It sends joint rows through `robodojo_step` and native
  EEF rows through `robodojo_step_ee`, in chunks of up to 50, and stops early if the
  episode ends. Then look again.

The same toolkit calls become your controller later, so a manual procedure that
works carries over directly.

```python
# /workspace/code/scratch/move.py  (run with: python code/scratch/move.py)
import robodojo_toolkit as tk
from api.runtime import Context

with Context() as ctx:
    meta, _ = tk.parse_reply(ctx.call("robodojo_observe"), images=False)
    pose = tk.pose_math({"operation": "compose_pose", "delta_frame": "environment",
        "base_pose": {"position_m": meta["eef_positions"][0],
                      "rotation": {"representation": "quaternion_wxyz", "value": meta["eef_quaternions_wxyz"][0]}},
        "delta_pose": {"position_m": [0.0, 0.05, 0.08]}})["pose"]
    plan = tk.shared_planner().plan("left", meta["states"],
        {"position": pose["position_m"], "quaternion_wxyz": pose["quaternion_wxyz"]})
    print(plan["status"], plan.get("diagnostics"))
    if plan["status"] == "Success":
        reply, after = tk.execute(ctx, plan["actions"])
        print("left link6 now", after["eef_positions"][0])
```

All development scripts share one warmed planner process: the first
`shared_planner()` call in the container takes about 20 s, later scripts connect
in well under a second. Short single-purpose scripts are therefore cheap.

## First exploration

1. Read the rubric and list the required object relationships, orientation,
   placement and robot end state as a short completion checklist in memory.
   Check it against `task_source/`. `run_reward` and `get_score` in the task
   module are the exact success and scoring checks (thresholds, required gripper
   and arm end state). The object assets give sizes, grasp surfaces and
   functional frames for planning. They are reference only: always localize
   objects in the live scene.
2. Call `exploration_status`. If no episode is active and budget remains, call `start_episode`;
   otherwise inspect the active scene. Check whether the episode has ended before
   moving.
3. Inspect the camera images and any demonstration. Read
   [in-context-action-learning](../in-context-action-learning/SKILL.md) when
   demonstrations are supplied. Identify the objects and a short sequence of
   manipulation stages.
4. Localize targets from RGB. There is no depth or pixel-to-3D tool. Use
   [RGB position calibration](../rgb-position-calibration/SKILL.md): known table
   geometry, measured link6 poses and small feedback-guided motions.
   - For unclear details, move an empty arm closer:
     [wrist-camera inspection](../wrist-camera-inspection/SKILL.md).
5. Move in small, inspectable chunks.
   - Plan free-space approaches with the planner.
   - Use `approach` rows for short explicit paths near contact.
   - Use `set_gripper` for grasp and release.
6. Inspect new images after each chunk, revise, and save useful parameters and
   evidence. Work through the checklist, then evaluate.

## Choosing a motion

- **Reach a pose (approach, retreat, transfer):** `shared_planner().plan(arm, state,
  target)`. This is the most accurate path to a pose, and it avoids robot
  self-collision and the table. It does not see task objects or a held object:
  choose waypoints clear of them.
- **Explicit short path (descend, insert, slide, lift, small rotations):**
  - `servo(ctx, arm, target)` or `step_eef(ctx, waypoints)` re-solve one bounded
    update (at most 2 cm / 0.1 rad per step) from the *measured* state every step.
    This is the safest choice near contact.
  - `eef_rows(state, waypoints)` or `approach(...)` predict bounded rows open loop
    for one call. Send a few, observe, and recompute from the new measured state.
- **Native EEF rows (`ee_rows` → `robodojo_step_ee`):** the environment's own IK
  solves each link6 pose, as in the official evaluation. It has no step bound and
  no collision check: a single far pose is covered in one step, and the IK may
  choose another arm configuration. Near objects, send a path of close poses
  (a few mm apart) and check the measured pose.
- **Gripper only:** `set_gripper(state, arm, opening, count)`. Closing on an
  object leaves `states` at the commanded value, so check the grasp in images
  and by lifting slightly.
- **Wait or settle:** `hold(state, count)`.
- **Relative intent (lift 5 cm, rotate the wrist):** compose an absolute target
  with `tk.pose_math` from the measured pose. Never add quaternion components.

## Step accounting and feedback

- Every row is one environment step against the native limit; batching reduces
  calls, not steps.
- `robodojo_step` and `robodojo_step_ee` return the observation after their last
  row, including images. In exploration they also carry harness transition
  information, and a chunk may stop early if the episode ends. Under the official
  adapter (rehearsal, formal, official evaluation) there is no terminal feedback.
- Planned or predicted poses are not measurements: compare the returned
  `eef_positions` with your target.
- If a call fails or control becomes uncertain, stop, inspect `exploration_status` and a fresh
  observation, and do not blindly repeat the motion. Never ask the operator for
  task or action advice during an episode.

## Confirm success and hand off

- **Confirming:** after strong visible evidence of completion, call `evaluate`.
  Native `task_complete=true` also confirms success. Partial progress, a
  successful plan or a clean return is not success. An ended episode without
  confirmed success is not evidence of completion.
- **After `evaluate` returns false:** recover within the remaining steps when
  possible, and re-evaluate only after meaningful new action.
- **New episodes:** start one only when needed and budget remains, keeping
  episodes for rehearsals.
- **Handing off:** record the successful procedure, its parameters and evidence
  in `memory/`, then read [autonomous-control](../autonomous-control/SKILL.md).
  Use `finish` to close exploration; it is not a submission.
