# RoboDojo and cuRobo control audit

> **Scope.** This audit covers the native MCP layer
> (`services/robodojo/mcp_server.py`), including tools that the official session
> profile no longer exposes to agents: `robodojo_step_eef`,
> `robodojo_free_space_move` and `robodojo_pose_math`. Agents see only
> `robodojo_observe`, `robodojo_status` and `robodojo_step`. The motion
> semantics below are reproduced in bundles by `packages/robodojo_toolkit`, whose
> planner uses the official `curobo_tmp.yml` configuration rather than the
> harness `x5_v2` one. The robot-only preview service has been removed.

This audit records the controller layer and timing used by the MCP integration.
It is intended to keep future skills from accidentally treating planner
samples as executable policy actions or measured states.

## Confirmed RoboDojo control path

| Layer | Standard rate | Value | Owner |
|---|---:|---|---|
| Agent/evaluation action | 25 Hz | absolute dual-arm joint positions plus normalized grippers | policy/MCP |
| Action interpolation | 250 Hz, 10 ticks/action | position targets; regular actions use zero desired arm velocity | `EvalEnv` |
| cuRobo internal reference | 250 Hz (`interpolation_dt=0.004`) | sampled into 25Hz 14D endpoints before execution; velocities discarded | cuRobo adapter |
| Actuation | 250 Hz physics | position and velocity targets | RoboDojo `RobotManager` |
| Effort | 250 Hz physics | implicit PD effort, clipped by actuator limits | IsaacLab/PhysX |
| Observation/private evaluation/budget | 25 Hz | post-block cameras/state and native task checks | `EvalEnv` |

RoboDojo's standard simulation config is `dt=0.004`, `decimation=1`, and
`render_interval=10`. Its observation config collects at 25 Hz, so one
benchmark step spans ten physics ticks. All published policy deploy adapters
call `take_action`/`take_action_batch`; they do not submit a torque stream.

All three MCP motion interfaces execute ordinary native `take_action` rows.
This retains the standard interpolation, support-arm behavior, evaluation,
and budget accounting. Dense cuRobo execution is no longer a separate MCP path.

## Confirmed EEF pose representation

The source path agrees at every boundary:

- IsaacLab body poses are `[x,y,z,qw,qx,qy,qz]` in world coordinates.
- `RobotManager.get_link_pose(..., is_relative=True)` subtracts only the
  environment-origin translation; quaternion axes and order are unchanged.
- Observation `left_ee_pose`/`right_ee_pose` and MCP
  `eef_quaternions_wxyz` therefore describe `link6` in the environment-origin
  frame, ordered `[left,right]`.
- The MCP DLS adapter converts incoming `wxyz` to SciPy's internal `xyzw`,
  solves against `link6`, then emits absolute joint targets.
- The cuRobo adapter concatenates `[x,y,z,qw,qx,qy,qz]`; its `Pose` type and
  RoboDojo frame transform both use `wxyz`.

`robodojo_step_eef` is an MCP convenience layer, not RoboDojo's raw 14-D
action. Each arm receives `{position, quaternion_wxyz, gripper_closed,
gripper_opening?}`. Both arms are mandatory. `robodojo_free_space_move` uses
the same position, link, frame, and quaternion convention for one arm. Its
optional continuous opening holds the measured plan-start opening when omitted.
Ordinary 14-D actions contain joints and gripper openings only.

RoboDojo's native policy EEF action uses `left_ee_pose`/`right_ee_pose` packed
as `[x,y,z,qw,qx,qy,qz]` plus separate gripper commands. The MCP does not expose
that untyped vector interface: its bounded EEF adapter validates named fields,
computes a local DLS joint target, and then uses the normal joint-action path.

The scene-independent `robodojo_pose_math` adapter loads the vendored
`isaaclab.utils.math` module directly. It exposes conversion, composition,
relative-pose, and target-format operations but not low-level math code. It
does not start Isaac, advance physics, or read episode/scene state.

## cuRobo semantics

`plan_pose` solves for motion, and `get_interpolated_plan()` returns a
time-parameterized `JointState`. The rows are not torques and they do not move
Isaac Sim until a controller applies them. Upstream cuRobo defines motion time
from state intervals; for N trajectory states the duration is `(N-1)*dt`.
The adapter checks state zero against measured joints, then takes samples
10, 20, ... and the final sample if necessary. Each becomes an absolute 14D
target executed by `take_action`; the last short interval is rounded up to
one 40ms step. Planned velocities are discarded. Passing every dense sample
as a policy action would still incorrectly slow the trajectory roughly 10x.

The motion planner explicitly uses 1mm and 0.005rad tolerances (not just the
separate IK solver). An independent FK endpoint check also enforces them.
Measured `goal_reached` is reported separately from execution `Success`.
No hidden settling actions are appended.

Primary implementations reviewed:

- [NVLabs cuRobo](https://github.com/NVlabs/curobo), including
  `solver_trajopt_result.py` and the motion-planning examples.
- [NVIDIA IsaacLab](https://github.com/isaac-sim/IsaacLab), whose current
  mimic cuRobo test turns planned poses into environment actions at the
  environment's configured cadence.
- [NVIDIA ENPIRE](https://github.com/NVlabs/ENPIRE), where free-space planning,
  cached preview, timestamps, and controller-owned execution are separate.
- [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo), including its
  evaluator, published policy deploy loops, control manager, robot manager,
  simulation config, and support-arm trajectory path.

The important common rule is that sampling and execution cadence must agree.
Our chosen adapter converts internal dense plans to native 25Hz policy actions,
so recording and replay use one command representation.
The earlier dense-versus-ENPIRE comparison is historical evidence in
[JOINT_REPLAY_BENCHMARK.md](JOINT_REPLAY_BENCHMARK.md), not the current API.

## Collision information and privilege boundary

The current RoboDojo cuRobo world contains:

- X5 collision spheres and self-collision exclusions from the robot config;
- one 3x3x0.05 m cuboid representing the table.

It does not contain task objects, task labels, object poses, a depth-derived
world, or the other arm. The cuRobo cspace also fixes gripper joints at the
configured retract/default values while returning only six active arm joints.
Consequences:

- No privileged task-object position is used by `free_space_move`.
- The target pose and measured robot joints are the only episode-varying
  planning inputs.
- A successful plan is valid only for the modeled robot/table world. It cannot
  certify clearance from task objects, the other arm, or a differently opened
  gripper.
- Even modeled collision geometry can differ from the contact shapes used by
  Isaac Sim, so measured tracking and post-motion images remain authoritative.

## MCP transaction

```text
agent target
    -> plan_pose (no physics advance)
    -> convert to and cache complete 25Hz joint targets
    -> execute each row through native take_action
    -> private evaluation + RGB/depth/calibration at 25 Hz
    -> MCP exposes perception, state, executed targets, measured goal error
    -> terminal task_complete when available; scores remain hidden
    -> final frame attached; complete ordered sequence saved under runtime/frames
```

A cached plan is stale after any intervening action or when any measured state
component changes by more than 0.002. This follows ENPIRE's useful preview/cache pattern while
using RoboDojo's actual controller and evaluator timing.
