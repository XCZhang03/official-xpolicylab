# Robot MCP guide (official profile)

Agents reach the simulator only through the host-owned MCP. They connect either
directly (as Codex tools) or from development Python
(`from api.runtime import Context`). Every session uses the `official` observation
profile, so the robot tools are exactly the ones an XPolicyLab policy gets:

| Tool | Contract |
| --- | --- |
| `robodojo_observe` | Cached post-action observation: RGB images and state. It does not step physics. |
| `robodojo_status` | Episode status, environment spec and cumulative step cost. It does not step physics. |
| `robodojo_step` | Execute 1..50 absolute 14-D joint rows at 25 Hz, then return the observation after the last row. |
| `robodojo_step_ee` | Execute 1..50 official native EEF rows (16-D: per arm link6 `x,y,z,qw,qx,qy,qz` + gripper) at 25 Hz; the environment's own cuRobo IK solves each row, as in official `EvalEnv.take_action`. |

In `services/mcp_contract.py`, `OFFICIAL_ROBOT_TOOLS` is the allow-list and the
`official` `ObservationProfile` does the filtering. The native service in
`services/robodojo/mcp_server.py` implements more tools, such as EEF steps, the
cuRobo move, pose math and pixel-to-position. None of them is listed or dispatched
in this contract, so the equivalent motion code lives in bundles, via
`robodojo_toolkit`.

## Observations

- **Cameras:** `cam_high`, `cam_left_wrist` and `cam_right_wrist`, as MCP PNG image
  blocks. Image replies carry `content` without a duplicate `structuredContent`,
  so Codex keeps the image blocks.
- **Filtered out:** depth, intrinsics, extrinsics and `camera_to_world_usd` are
  removed (`GEOMETRY_KEYS`), matching `env_cfg/arx_x5.yml` in the official
  evaluation.
- **`states`:** 14 values, `[left q1..q6, left gripper, right q1..q6, right gripper]`.
  As in official observations, the two gripper entries are the last *commanded*
  openings (0 closed, 1 open), not measured ones.
- **`eef_positions` / `eef_quaternions_wxyz`:** `[left, right]` X5 `link6` poses.
  Positions are metres relative to the environment origin; orientations are
  `wxyz` in environment axes.
- **Episode fields:** replies carry `episode_ended`, and, at the end, native
  `task_complete`. Native reward, score and intermediate success are never
  returned.

`robodojo_toolkit.parse_reply(reply)` returns `(meta, images)` from any robot reply.

## Actions and steps

- **One row, one step:** every row is one 25 Hz benchmark step. RoboDojo turns each
  row into ten 250 Hz joint-target updates.
- **Step accounting:** step limits count rows, not calls, so batching saves calls
  but not steps. Execution stops at native termination or at the episode step
  limit.
- **Replies:** a `robodojo_step` reply attaches the final frame and a
  `frame_sequence` manifest pointing to every published 25 Hz frame of the chunk.
- **Rejected requests:** a request rejected during validation reports
  `no_action_executed: true` and leaves the episode usable.
- **Failures after dispatch:** these (for example a lost response) set
  `control_uncertain: true` and block further motion. Inspect with
  `robodojo_status` / `robodojo_observe`; don't retry blindly.

## Motion in bundles

`packages/robodojo_toolkit` supplies the motion code, and the agent-facing
[motion-toolkit skill](../auto_research_agent/.agents/skills/motion-toolkit/SKILL.md)
shows its entry points:
- `shared_planner().plan(arm, states, target)`: the vendored RoboDojo cuRobo
  planner, on the official config, returning 25 Hz rows. It avoids self-collision
  and the table; it does not see task objects.
- `approach(...)`: bounded DLS IK rows.
- `set_gripper`, `hold`: row helpers.
- `pose_math`: pose composition and conversion.

The planner reaches its goal within 1 mm / 0.005 rad on the official path (see the
[official README](../official/xpolicylab/README.md)).

## Lifecycle tools (harness only)

The session frontend (`services/controller/frontend.py`) adds these tools during
exploration. Isolated rehearsal and formal runs receive only the four robot tools, through
the official AgentBundle bridge.

| Tool | Contract |
| --- | --- |
| `status` | Task, manual-success evidence, episode state and timeout, exploration episodes left, bundles with `rehearsal_qualified`, storage, artifact paths |
| `start_episode` | Close the previous exploration episode and start the next one, with a distinct seed |
| `evaluate` | Trusted evaluation of the current exploration episode |
| `finish` | Close exploration; this is not a submission |
| `register(source)` | Freeze `code/<project>` (must contain `controller.py`) |
| `rehearse(bundle)` | Run the frozen bundle alone in a fresh exploration episode, under the formal wall limit |
| `submit(bundle)` | One formal batch of an exactly rehearsed bundle; no retries |

With `exploration_envs > 1`, episode-scoped calls need an explicit `env_id`; see
[parallel exploration](parallel_exploration.md). The full isolation, timing and
scoring rules are in the [infrastructure contract](controller_backend.md).
