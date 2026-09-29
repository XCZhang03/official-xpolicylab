# robodojo_toolkit API

`import robodojo_toolkit as tk`. The same package version is installed in your
development container, in isolated rehearsal and formal runs, and on the official
policy server.

## Conventions

- **Frame:** poses are link6 in the environment frame of observations: metres, unit
  wxyz quaternions, `[left, right]` order.
- **Joint row (14 numbers):** `[left joint1..6, left gripper, right joint1..6, right gripper]`.
  Joints are absolute radians. Grippers are openings in [0, 1]: 0 closed, 1 open.
- **Native EEF row (16 numbers):**
  `[left x,y,z, qw,qx,qy,qz, left gripper, right x,y,z, qw,qx,qy,qz, right gripper]`.
- **Pose target (dict):** `{"position": [x,y,z], "quaternion_wxyz": [w,x,y,z]}`. Optionally
  add `"gripper_opening": 0..1` or `"gripper_closed": bool`. Without either, the arm
  keeps its commanded opening.
- **State:** `meta["states"]` uses the joint-row layout. The arm joints are measured.
  The gripper values are the last **commanded** openings, as in official observations,
  so a closed command reads 0 even when an object blocks the jaws.

## How a step executes

- **One row is one environment step (25 Hz, 0.04 s).** The environment moves the
  arm joints linearly from their current position to the row's targets over the
  first 8 of 10 physics ticks, then holds for the last 2. Measured joints may still
  lag a target under load or contact.
- **Step budget:** every row counts against the task's native step limit.
- **Chunks:** `robodojo_step` and `robodojo_step_ee` each take 1 to 50 rows.
  - The reply is the observation after the chunk's last row, so no feedback arrives
    in between.
  - Batching saves calls and wall time, not steps. Short chunks give more feedback.
- **Episode end (harness):** there, a reply that ends the episode includes
  `episode_ended`; `tk.episode_ended(reply)` checks for it.
- **Episode end (official evaluation):** nothing is reported. The bundle is cancelled
  instead, and the next `ctx.call` raises. Never wait for a success signal.

## Choosing a motion

| Goal | Use | Feedback | Steps |
|---|---|---|---|
| Reach a pose in free space (approach, transfer, retreat) | `shared_planner().plan` | none within the plan | as planned |
| Short Cartesian move near objects (descend, insert, slide, lift) | `servo` or `step_eef` | measured every step or chunk | one per update |
| Short move, fewest calls | `eef_rows` + `execute` | once per chunk | one per row |
| Pose path executed by the official native IK | `ee_rows` + `execute` (`robodojo_step_ee`) | once per chunk | one per row |
| Open or close the gripper | `set_gripper` | once per chunk | `count` |
| Wait or let an object settle | `hold` | once per chunk | `count` |
| Check whether a pose is reachable | `ik` | none (kinematics only) | 0 |

**Bounded versus native EEF control:**
- **Bounded (`eef_*`, `step_eef`, `servo`, `approach`):** each row moves each arm by at
  most 2 cm and 0.1 rad in pose, and 0.05 rad per joint. Motion is predictable and
  continuous; a far target takes many steps.
- **Native (`ee_*`, `robodojo_step_ee`):** the environment solves the target with its
  own IK and has no step bound. A far target is covered in a single step (10 physics
  ticks), with no collision check, and IK may pick another arm configuration.
  - Measured on the official stack (make_kong, both arms, 10–21 cm moves): the goal was
    reached within 0.09 mm / 0.0003 rad, whether sent as the goal repeated for 15 steps
    or as a 25-pose straight line plus 5 settle steps.
  - Near objects, keep consecutive native targets close (a few mm per row) and check
    the measured pose.

## Observations

`parse_reply(reply, images=True) -> (meta, {camera: np.uint8 HxWx3})` reads a reply from
`robodojo_observe`, `robodojo_status`, `robodojo_step` or `robodojo_step_ee`.
- Cameras: `cam_high`, `cam_left_wrist`, `cam_right_wrist` (RGB only).
- `meta` fields: `step_id`, `states`, `eef_positions` (`[[x,y,z] left, right]`),
  `eef_quaternions_wxyz`, `instruction`, `attachments`.
- In exploration the harness adds `transition` (per-step records, including
  `episode_ended`), `frame_sequence` and `artifacts`. Under the official adapter,
  `transition` is always the empty compatibility value `{"steps": []}` and the others
  are absent, so there is no terminal feedback.

`pose_error(meta, arm, target) -> (position m, rotation rad)` is the measured link6
error to a target.

## Free-space planning

`planner = shared_planner()` returns the process-wide cuRobo planner.
- **Development:** every script shares one warmed planner process. The first call in
  the container takes about 20 s; later scripts connect at once.
- **Rehearsal, formal and official:** the planner is built in-process. The official
  adapter warms it before the episode starts.

`planner.plan(arm, state, target, gripper_opening=None) -> dict`
- **Arguments:**
  - `state`: the 14-D state to start from (normally `meta["states"]`).
  - `target`: a pose target.
  - `gripper_opening`: this arm's opening to ramp to during the move (default: keep).
- **On success:** `{"status": "Success", "actions": N x 14 rows (numpy), "action_count",
  "diagnostics": {"final_position_error_m", "final_rotation_error_rad", "final_joint_state"}}`.
  - The other arm holds.
  - A plan is accepted only if its endpoint is within 1 mm / 0.005 rad of the target.
    This is kinematic: check the measured pose after executing.
- **On failure:** `{"status": "Planning_Failed", "reason", "failure_stage"?, "diagnostics"?}`.
  The cause is no collision-free path or no IK solution. Try a nearer waypoint,
  another orientation or the other arm.
- **Collision model:** robot self-collision and the table (top at z = 0.74 m).
  Task objects and a held object are not modelled, so plan to poses clear of them
  and finish near contact with bounded steps.
- **Timing:** about 0.03 s per plan. Send long plans with `execute`, which splits them
  into chunks of 50.

## End-effector motion

- **`eef_row(state, targets) -> (row14, diagnostics)`:** one bounded update.
  - `targets = {"left": target, "right": target}`; either arm may be omitted, and an
    omitted arm holds its joints and gripper.
  - `diagnostics[arm]` gives `target_position_error_m` and `target_rotation_error_rad`
    (before the step), plus `proposed_joint_delta`.
- **`eef_rows(state, waypoints) -> (rows, diagnostics)`:** one bounded update per waypoint,
  each starting from the previous *predicted* joints (open loop). Send the rows with
  one `execute`.
- **`step_eef(ctx, waypoints, state=None) -> (reply, meta, diagnostics)`:** the bounded
  closed-loop EEF stepping of earlier harness versions.
  - One row per waypoint and one call per row; each update starts from the joints
    measured after the previous step.
  - `state` defaults to a fresh observation.
- **`servo(ctx, arm, target, max_steps=50, rows_per_call=1, position_tol_m=0.001,
  rotation_tol_rad=0.005) -> (reply, meta, info)`:** repeats bounded updates toward one
  pose until the **measured** pose is within tolerance.
  - `info` = `{"reached", "steps", "position_error_m", "rotation_error_rad"}`.
  - `rows_per_call > 1` predicts that many rows per call: fewer calls, coarser feedback.
- **`approach(arm, state, target, max_steps=50, position_tol_m=0.001,
  rotation_tol_rad=0.005) -> (rows, diagnostics)`:** bounded rows for one arm, predicted
  open loop until they are predicted to reach the target.
- **`ee_row(meta, targets) -> row16`:** one native EEF row for `robodojo_step_ee`.
  - Arms without a target keep their measured pose and commanded gripper from `meta`.
  - The environment's IK (cuRobo, many seeds, starting from the current joints) turns
    the row into joint targets, with no step bound and no collision check.
  - If IK fails for an arm, that arm keeps its previous target.
- **`ee_rows(meta, waypoints) -> N x 16 rows`.**
- **`execute(ctx, rows, chunk=50) -> (reply, meta)`:** sends 14-D rows with
  `robodojo_step` and 16-D rows with `robodojo_step_ee`, `chunk` rows per call. It stops
  early if the harness reports that the episode ended.
- **`ik(arm, seed_joints6, target, max_iterations=400) -> (joints6, info)`:** a local IK
  solution near the seed, from iterated bounded updates.
  - It does not check collisions.
  - `info["converged"]` reports whether the pose is reachable from this seed.
  - Move there with the planner or bounded rows, never in one jump.

Examples:

```python
meta, _ = tk.parse_reply(ctx.call("robodojo_observe"), images=False)
down = {"position": [x, y, z - 0.03], "quaternion_wxyz": q}

# Descend 3 cm under measured feedback, one step per call:
reply, meta, info = tk.servo(ctx, "left", down, max_steps=40)

# The same with the official native IK, as a 6-row straight line:
path = [{"left": {"position": [x, y, z - 0.005 * k], "quaternion_wxyz": q}} for k in range(1, 7)]
reply, meta = tk.execute(ctx, tk.ee_rows(meta, path))
print(tk.pose_error(meta, "left", down))
```

## Rows

- `hold(state, count=1)`: repeat the current targets for `count` steps.
- `set_gripper(state, arm, opening, count=10)`: ramp one gripper linearly over `count`
  steps; arm joints hold.
- `policy_actions(positions, state, arm, opening, planner_dt=0.004)`: converts dense
  planner samples into 25 Hz rows (used by `plan`).
- Constants:
  - `MAX_TRAJECTORY_ACTIONS` = 50;
  - `GOAL_POSITION_TOLERANCE_M` = 0.001;
  - `GOAL_ROTATION_TOLERANCE_RAD` = 0.005.

## Kinematics

`DualArm()` is the X5 model: the packaged URDF and the official arm roots.
- `eef(arm, joints6) -> target`: forward kinematics.
- `check(states, eef_positions, eef_quaternions_wxyz)`: model-versus-measured link6
  error per arm (normally well under 1 mm).
- `step_toward(arm, joints6, target) -> (joints6, diagnostics)`: one bounded update,
  for your own loops.
- `preview(rows14)`: predicted link6 poses and gripper openings for joint rows. These
  are kinematic predictions only: no physics, contact or objects.
- `spec`: joint names, limits (joints 1–5 ±10 rad, joint 6 ±3.14 rad), arm roots,
  table height.

## Pose math

`pose_math(arguments) -> dict`:
- `convert_rotation`: `rotation` in; result in `output_representation`, plus
  `quaternion_wxyz`.
- `compose_pose`: `base_pose` and `delta_pose` (each part optional) with `delta_frame`.
  - `local` means T_base · T_delta (a delta in the gripper's own axes).
  - `environment` adds the translation in environment axes and left-multiplies the
    rotation.
  - Returns `pose`.
- `relative_pose`: the exact inverse: `base_pose`, `target_pose` -> `delta_pose`.
- `format_target` / `extract_target`: convert between poses and targets.

Supported rotations:
- representations: `quaternion_wxyz`, `quaternion_xyzw`, `euler_xyz_extrinsic_rad`,
  `axis_angle_vector_rad`, `rotation_matrix_row_major`;
- input form: `{"representation": ..., "value": [...]}`.

A pose is `{"position_m": [...], "rotation": {...}}`. Results contain `position_m` and
`quaternion_wxyz`. Never add quaternion components by hand.

```python
pose = tk.pose_math({"operation": "compose_pose", "delta_frame": "environment",
    "base_pose": {"position_m": meta["eef_positions"][0],
                  "rotation": {"representation": "quaternion_wxyz", "value": meta["eef_quaternions_wxyz"][0]}},
    "delta_pose": {"position_m": [0, 0, 0.05],
                   "rotation": {"representation": "euler_xyz_extrinsic_rad", "value": [0, 0, 0.3]}}})["pose"]
target = {"position": pose["position_m"], "quaternion_wxyz": pose["quaternion_wxyz"]}
```

## Under the official adapter

Isolated rehearsal and formal runs execute `controller.py:main(ctx)` through the
official XPolicyLab adapter, exactly as the evaluation does:
- **Tools:** `ctx` has only `call(tool, **args)` for `robodojo_observe`,
  `robodojo_status`, `robodojo_step` and `robodojo_step_ee`, plus `output_dir`.
- **Chunks:** each step call is one official action chunk. The reply carries the
  adapter's step count, `transition` = `{"steps": []}` (never an episode end) and no
  `frame_sequence`.
- **Slow code:** if the bundle computes for more than 90 s between chunks, the
  environment holds the pose one step at a time until the next chunk.
- **After `main` returns or raises:** the pose is held until the episode ends. A
  raising `main` fails the rehearsal.
- **Episode end:** the bundle is cancelled and the next call raises.

In development, `from api.runtime import run_official; run_official(ctx, "/workspace/code/<project>")`
runs a bundle the same way on the current exploration episode.
