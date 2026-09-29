# Joint trajectory replay benchmark

> **Historical record.** These measurements were taken on the native MCP layer
> with the since-removed robot-only preview service and the EEF/free-space tools.
> The official session profile exposes only `robodojo_observe`, `robodojo_status`
> and `robodojo_step`. For current planning accuracy on the official path, see the
> [official README](../official/xpolicylab/README.md). The report paths below are
> machine-local `runtime/` files.

## Current 25Hz contract — September 23, 2026

All MCP motion tools now execute ordinary native 25Hz absolute joint targets.
The GPU test uses actual MCP handlers for FK, cuRobo planning, robot-only
preview, cached and direct execution, EEF steps, and fresh-episode joint replay.
It checks final inline RGB/depth images and complete 25Hz frame manifests.

Run with the bootstrapped runtime and an idle GPU:

```bash
runtime/envs/robodojo/bin/python -m pytest -vs local_tests/unit/test_gpu_joint_replay.py
```

Latest report: `runtime/integration-tests/preview-parity-jj7jevqc/joint_replay_accuracy.json`.
Full images, manifests, and native logs remain alongside it on SSD.

| Motion | Native steps | Final position error | Final rotation error |
| --- | ---: | ---: | ---: |
| Left combined translation/rotation (~21.7cm / 49°) | 22 | 0.02897mm | 0.00030876rad |
| Right wrist rotation (~69°) | 28 | 0.02933mm | 0.00030261rad |

Errors are measured immediately at motion end, with no added settling steps.
The combined pose previously missed by 1.7459mm; explicitly setting motion
planning tolerances to 1mm / 0.005rad resolved that planner residual in this case.
A separate FK check rejects inaccurate plans, and execution reports measured
`goal_reached` separately from command completion.

The two motions plus five EEF steps were replayed using only returned
`transition.steps[*].executed_action` through `robodojo_step` in a fresh
matching MakeKong episode. Maximum joint, normalized-gripper, and EEF-position
differences were zero; quaternion comparison roundoff was 4.22e-8rad.

Robot-only preview final EEF-position mismatch was at most 0.00105mm.
An additional state-semantics bug was corrected: upstream RoboDojo's
observation grippers were last commands, not measurements. MCP and preview now
share physical measured-state extraction. The combined-motion normalized
gripper mismatch fell from 0.05045 to 0.0001622.

This validates the tested free-space motions, not arbitrary contacts, grasp
retention, task success, or full scene reconstruction from a 14D state.
Preview contains no task objects, table, or ground. Old dense-execution endpoint
records are not version-2 replay data.

Final standalone-protocol validation: 140 non-GPU tests and both real GPU
planning/preview tests passed. Both RoboDojo and preview MCP stdio readiness
checks passed. The separate small-motion smoke report is
`runtime/integration-tests/preview-parity-ap5qug6z/accuracy.json`.
Independent runtime and agent-workspace reviews found no remaining blocking
issue in the reviewed contracts and skills. Test-owned simulators were closed.

Planning and preview share a 50-action limit, with transport capacity sized for
all returned frames without relying on compression. Malformed motion batches
and preview requests are validated before native RPC or action execution.
This patch covers interactive robot control only; the future auto-research
setting (script development, policy training, and associated services and
sandboxes) is separate and is not required by these tests.

## Historical executor comparison

> Historical benchmark of the former dense executor. The current version-2 MCP
> executes 25Hz joint targets directly; dense/reference and ENPIRE modes below
> are comparison results, not supported agent interfaces.

This is a diagnostic comparison, not a production execution-mode change or a
claim of task success. All candidates use fresh MakeKong episodes with matching
seeds, the same dual-X5 robot, native gains, physics timestep, and step budget.
The reference trajectory is planned once; replay candidates do not re-plan.

The former test launcher has been removed after selecting the unified policy
execution path. Retained reports below preserve its results. The current
`test_gpu_joint_replay.py` instead asserts the version-2 contract and accuracy.

## Candidates

| Candidate | Commands within a 40 ms observation interval |
| --- | --- |
| `dense` | Former production cuRobo path: ten 4 ms position/velocity targets |
| `joint_replay` | The last 14D target of each block, through ordinary `robodojo_step` |
| `joint_repeat` | Repeated ordinary replay from a fresh matching episode |
| `dense_reference` | Recorded dense position/velocity samples, replayed without planning |
| `enpire_60hz` | Timestamp-interpolated positions at 60 Hz; zero velocity references; hold between updates |
| `enpire_25hz` | Same timestamp method at 25 Hz; no measured-state interpolation |

The latter three used a **test-only simulator launcher** and recorded MCP command
data. The test still observes and commands through MCP. The launcher substitutes
the inner executor only when a submitted action matches its reference row. No
new production MCP/RPC capability or runtime environment switch is introduced.
Native accounting, rendering, termination, and physics remain enabled.

The ENPIRE-style candidates reproduce the waypoint sampling/control semantics,
not the complete ENPIRE server. They use deterministic simulation elapsed time,
not wall-clock sleeps or network jitter. The endpoint is explicitly sent at the
last tick. They share a 15-step (0.6 s) post-motion hold with every candidate.
This deliberately controls for robot/gain/settling differences.

## Reference implementations inspected

- RoboDojo pinned at `ee67a1468510da7624a089164402359f2afc72c8`:
  `env/robot_manager/robot_manager.py:plan_ee` packages planned position and
  velocity samples. `control_robot` sends position and velocity targets to Isaac
  articulations. Ordinary evaluation actions instead use measured-state
  interpolation (eight intermediate samples followed by two endpoint holds).
- [ENPIRE](https://github.com/NVlabs/ENPIRE), inspected at
  `99ee90acf65b5b18957c8382ad580db999528be3`:
  `freespace_move.py:_execute_trajectory` sends timestamped waypoints;
  `native.py:_sample_keypoints` linearly interpolates them; the direct helper
  defaults to 60 Hz and commands zero desired velocities. The Portal server uses
  a configurable control period. These are not RoboDojo's interpolation rule.
- [Official cuRobo Isaac Sim guide](https://curobo.org/get_started/2b_isaacsim_examples.html)
  and the pinned [v0.7.8 motion-generation example](https://github.com/NVlabs/curobo/blob/v0.7.8/examples/isaac_sim/motion_gen_reacher.py):
  the example sends both `cmd_state.position` and `cmd_state.velocity` using
  `ArticulationAction`. Its nonreactive interpolation interval is 0.05 s and it
  advances three default simulation steps per trajectory command. Do not copy
  those timing constants into RoboDojo's 0.004 s physics loop. Our
  `dense_reference` candidate preserves its position/velocity command semantics
  at RoboDojo's timestep; it is **not** an execution of that standalone example.

Upstream low-level support, our dense path, and the official example use the same
kind of position/velocity reference. Their differing scheduling/bookkeeping is
not evidence of different endpoint accuracy; compare time-aligned commands.

## Dense-to-25 Hz conversion

cuRobo sample 0 is the initial state, not a command. Remove it. Partition the
remaining samples into groups of ten. Keep the final position in each group:
original indices 10, 20, 30, ..., N-1. A short final group holds the final joint
position and uses zero arm velocity for its padding ticks. Grippers use the same
normalized interpolation as execution; the inactive arm holds its initial state.
No averaging, measured-state substitution, or velocity-to-position conversion
is involved. The recorded `executed_action` is the last normalized 14D command
actually constructed for that block.

The historical unit tests checked both arms, off-by-one alignment, closed grippers, native command
values, padding, and timestamp sampling. Historical GPU tests additionally checked equality
with `trajectory_preview.benchmark_action_blocks[].projected_14d_action`.

## Measurements and limitations

The standard scenario covers a base sweep, a roughly 69-degree wrist rotation,
a combined translation/rotation, gripper changes, and an EEF-step segment.
The elevated scenario adds a 21.7 cm / 49-degree combined move from home, with
four execution candidates. The historical reports record:

- EEF translation and quaternion-geodesic orientation error to requested poses,
  both at motion end and after the common hold;
- time-aligned joint, gripper, EEF position/orientation differences from dense;
- repeatability and initial-state mismatch;
- retained 25 Hz frames, executed targets, and native logs on SSD.

`joint_replay_accuracy.json` is written after each completed candidate. A passing
diagnostic test means the executions and comparisons completed, **not** that all
candidates met an accuracy threshold. Inspect metrics before selecting a
production contract. This does not test object contact, grasp retention, full
task success, or arbitrary layouts. Endpoint agreement alone does not establish
equivalent paths or equally useful imitation-learning labels.

## Results: 2026-09-23

Primary selection criterion: **measured final EEF error against the requested
goal**, not agreement with the dense reference. Report both arrival and settled
errors; do not hide an inaccurate arrival by measuring only after a long hold.

The standard six-episode run completed in 675.77 seconds. Its artifacts are at
`runtime/integration-tests/preview-parity-2814cf7x/joint_replay_accuracy.json`.
The separate elevated run is at
`runtime/integration-tests/preview-parity-3t4ig5rz/joint_replay_accuracy.json`.
Both include full frame sequences/native logs, exact replay actions, and a
`reference.json` containing executed dense commands. These paths are local,
ignored SSD artifacts, not distributable simulator assets or policy inputs.

### Goal reaching: standard scenario

Entries are **millimetres / degrees at motion end**, before the common hold.

| Goal | Dense | Ordinary 25 Hz | ENPIRE-style 60 Hz | ENPIRE-style 25 Hz |
| --- | ---: | ---: | ---: | ---: |
| Left base sweep | 0.0295 / 0.0173 | 0.0236 / 0.0139 | 0.0293 / 0.0172 | 0.0284 / 0.0168 |
| Right wrist rotation | 0.0293 / 0.0173 | 0.0293 / 0.0173 | 0.0293 / 0.0173 | 0.0293 / 0.0173 |
| Left combined stress case | 14.901 / 4.950 | 14.878 / 4.949 | 14.894 / 4.948 | 14.866 / 4.941 |

After 0.6 seconds of holding, the stress case still has approximately 14.91 mm /
4.95 degrees error for every method. This is **not** a successful reach and must
not be treated as a clean free-space demonstration just because planning passed.

The planned stress-case endpoint, checked with robot URDF FK, is within about
0.0037 mm / 0.000035 degrees of its requested pose. At execution end the measured
J2 is -0.26415 rad versus a -0.35001 rad target (0.08585 rad tracking error),
and J1 differs by 0.00954 rad. Native soft joint limits do not explain it: the
test-only robot diagnostic reports approximately [-10, 10] rad for J1--J5.
The exact mechanical/contact cause remains unconfirmed. No collision disabling,
gain changes, teleportation, or task-state manipulation was used to make it pass.

### Planning tolerance is distinct from physical tracking

RoboDojo's `_build_motion_planner_cfg` does not override the installed
`MotionPlannerCfg.create` defaults: **0.005 m position / 0.05 rad orientation**.
Its separate `_build_ik_cfg` uses 0.001 m / 0.02 rad; those settings do not set the
motion planner's acceptance threshold. A planner `Success` therefore does not
promise sub-millimetre task execution.

For the elevated case the planner reports a 1.7459 mm / 0.05455-degree endpoint
residual. Dense execution achieves almost exactly that residual: this case is
planning-accuracy limited rather than showing the stress case's tracking miss.

### Goal reaching: elevated 21.7 cm / 49-degree move

The four-episode elevated comparison completed in 246.79 seconds.

| Execution | Arrival position error | Arrival orientation error | After 0.6 s hold (mm / degrees) |
| --- | ---: | ---: | ---: |
| Dense position/velocity | 1.7459 mm | 0.05454 | 1.7459 / 0.05451 |
| Ordinary 25 Hz | 1.7450 mm | 0.05682 | 1.7459 / 0.05457 |
| ENPIRE-style 60 Hz | 1.7458 mm | 0.05465 | 1.7459 / 0.05449 |
| ENPIRE-style 25 Hz | 1.7469 mm | 0.05100 | 1.7459 / 0.05447 |

There is no practically meaningful endpoint advantage for an ENPIRE-style
candidate in these measurements. Along this motion, maximum position deviation
from dense was 2.201 mm for ordinary replay, 9.156 mm for timestamped 60 Hz, and
16.128 mm for timestamped 25 Hz; these are secondary path/phase metrics, not
goal errors.

### Secondary criterion: path replay

Maximum time-aligned difference from dense in the standard scenario:

| Replay | EEF position | EEF orientation |
| --- | ---: | ---: |
| Recorded dense position/velocity | 0 mm | Numerical noise only |
| Ordinary 25 Hz | 2.015 mm | 0.925 degrees |
| ENPIRE-style 60 Hz | 8.194 mm | 3.532 degrees |
| ENPIRE-style 25 Hz | 14.856 mm | 6.750 degrees |

The two independent ordinary replays produced identical joint/EEF positions;
quaternion-angle noise was at most 4.3e-8 radians. Dense replay also reproduced
the original exactly to that precision. Timing-based position-only playback
has larger time-aligned lag here, but does not improve final goal accuracy.

### Recommendation and remaining limits

Ordinary **25 Hz absolute 14D joint targets** are the simplest promising policy
training/replay contract in these tests. ENPIRE-style playback provides no
demonstrated goal-reaching advantage on this robot. Dense position/velocity
replay is necessary if exact reproduction of the current dense path is required.

For an exactly replayable 25 Hz dataset, collect demonstrations using the same
25 Hz execution path as the policy. Merely extracting endpoints from existing
dense demonstrations does not make their intermediate observations equivalent.
The benchmark has **not** switched production free-space execution to 25 Hz.

Before selecting the final production contract: choose task-specific goal
tolerances, validate measured endpoints rather than trusting planner status,
investigate the stress-case tracking failure, and test contact/grasp retention.
No method is established as universally most accurate by these few trajectories.

Validation: ten completed real-environment benchmark episodes across two GPU
test invocations; 125 non-GPU regression tests passed. Both benchmark tests
closed their owned services, and no owned simulator process remained afterward.
Earlier harness failures (output-directory setup and missing mandatory EEF
gripper flag) were corrected. An exploratory larger backward target was rejected
by cuRobo and is not counted among the successful benchmark trajectories.
