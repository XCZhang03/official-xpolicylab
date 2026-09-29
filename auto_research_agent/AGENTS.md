# RoboDojo auto-research (official interface)

Your goal is a frozen controller bundle that completes the configured task on the
official RoboDojo evaluation, with no live agent. Read `TASK.md` and satisfy every
condition for a score of 100. Manual success is preparation; the bundle's formal
batch success rate is the final result.

The robot interface is the official one. Observations are RGB (`cam_high`,
`cam_left_wrist`, `cam_right_wrist`) plus measured joint state and link6 poses.
Observations carry no depth or camera calibration. `cam_high` is fixed in the
environment. The wrist cameras have fixed mounts and intrinsics, and their world
poses follow the measured link6 poses. The RGB position calibration skill's
CALIBRATION.md derives all three models for you. The only robot tools are those of the
official evaluation: `robodojo_observe`, `robodojo_status`, `robodojo_step`
(absolute 25 Hz joint rows) and `robodojo_step_ee` (25 Hz link6 pose rows solved by
the environment's own IK). Motion planning, bounded IK and pose math come from the installed
`robodojo_toolkit` Python package; see
[motion-toolkit](.agents/skills/motion-toolkit/SKILL.md). One model tool is also
available, `gemini_generate` (Gemini 3.8 Flash, $10 per session): it generalizes to
domain randomization and can act as a function-calling decision agent, but is less
accurate per pixel; see [gemini](.agents/skills/gemini/SKILL.md). The bundle you
submit is exactly what an official XPolicyLab evaluation runs, so it may use only
these tools, the toolkit, and code or data inside the bundle.

## Start here

Read `TASK.md`, then any existing `memory/PROGRESS.md` and `memory/INDEX.md`, and
call `exploration_status` to learn which stage has been reached and what budget remains. Do not
start a new episode merely because the conversation resumed. If manual success is
not yet confirmed, begin with manual control; otherwise continue from the saved
evidence with autonomous control. Before submitting, confirm the bundle's rehearsal
outcome from `exploration_status.bundles` (`rehearsal_qualified`) and the `rehearse` result.
A note is not a replacement for server-confirmed evidence.

Read skills for the current stage when needed rather than the whole workspace first.
`MCP_SESSION.md` states this session's resolved interface.

## Workflow

1. **Solve the task yourself.** Read
   [manual-robot-control](.agents/skills/manual-robot-control/SKILL.md).
   - Read `task_source/README.md` first: it has the official task code, exactly how
     success and score are computed, and the objects' 3D geometry.
   - Use [in-context-action-learning](.agents/skills/in-context-action-learning/SKILL.md)
     when demonstrations are supplied.
   - Use [rgb-position-calibration](.agents/skills/rgb-position-calibration/SKILL.md)
     to localize objects without depth.
   - Use [wrist-camera-inspection](.agents/skills/wrist-camera-inspection/SKILL.md)
     for close-up views.

   Look with the MCP tools, move with short Python using the toolkit, and confirm
   success through `evaluate` or native `task_complete=true`.
2. **Turn it into a controller.** Read
   [autonomous-control](.agents/skills/autonomous-control/SKILL.md). Reproduce your
   successful decisions as observation-driven code in `controller.py:main(ctx)`.
   Test functions and full runs in exploration episodes across layouts. Use simple
   perception rules and toolkit motions where they hold across layouts, and
   [gemini](.agents/skills/gemini/SKILL.md) where appearance varies (unseen objects,
   clutter, domain randomization); refine its coarse outputs geometrically.
3. **Freeze and rehearse.** Evaluation runs on held-out layouts the bundle has
   never seen, so before registering a candidate final bundle run the
   [subagent audit](.agents/skills/subagent-audit/SKILL.md) for layout overfitting
   and test its findings on fresh layouts. `register(source="code/<project>")` freezes the
   directory; `rehearse(bundle=...)` runs it alone in a fresh container and
   episode. Inspect the evaluation and logs, not only the exit code. Any edit
   needs a new registration and rehearsal.
4. **Submit once.** After using the exploration budget and obtaining a successful
   rehearsal of the exact bundle, call `submit` with it. The formal batch
   (50 held-out episodes for new sessions) runs fresh containers and environments
   with no intervention or retries. Report successes divided by the fixed episode
   count; invalid or unconfirmed episodes count as failures. If a prerequisite
   cannot be met within budget, report the blocker; do not bypass it.

## Environment and budgets

- **Episodes:** `start_episode` begins an exploration episode with a new seed and
  the task's native step limit. Layouts differ between episodes, and objects move
  during interaction, so re-acquire the scene from fresh observations. Every
  executed row is one step against the limit.
- **Use the whole budget:** use all exploration episodes to pressure-test the
  controller across layouts, and reserve enough of them for rehearsals.
  Rehearsals consume exploration episodes.
- **Wall limit:** a rehearsal or formal episode has one wall limit including
  startup (`exploration_status.episode_timeout_seconds`); a timeout stops execution.
- **Python:** development Python uses `from api.runtime import Context`, then
  `with Context() as ctx:`. `ctx.call(tool, **arguments)` invokes MCP.
  - Scripts and your own tool calls share the live scene. Calls serialize
    individually, not per script, so stop a control loop before intervening,
    and check the scene afterwards.
  - Stop acting when an episode ends. After an ambiguous failure, inspect
    `exploration_status` and a fresh observation before retrying.
- **Finishing:** use `finish` to close exploration.

## Workspace and artifacts

- **Container:** Bash, Python, file and image tools run inside an offline
  development container with a GPU. `robodojo_toolkit` and its cuRobo planner
  are installed there. There is no internet, host shell, simulator mount or Docker
  socket; the only model API is the `gemini_generate` tool, routed by the host. Do not look for simulator internals.
- **Storage:** `/workspace` is persistent and writable. Workspace and agent CLI state
  share a storage cap (`exploration_status` or `df -h`).
- **Bundle layout:** keep the bundle under `/workspace/code/<project>/`, with
  `controller.py:main(ctx)` and every module or data file it needs. Use paths
  relative to `__file__`.
- **What isolated runs receive:** frozen, read-only source and current-run
  frames, never the live workspace or memory.
- **Outputs:** write generated files to `ctx.output_dir`. Isolated outputs
  return under `runtime/autonomous_controller/runs/<run-id>/exports/`, next to
  logs, traces and frames.
- **Inspecting runs:** look at selected images and logs as needed; do not load
  whole recordings into context.
- **Task source (`task_source/`, read-only, reference only):** the official
  RoboDojo source for this task only. It contains the task module, excerpts of the
  RoboDojo code it depends on (success and score checks, episode flow), its
  configs, and the assets of each of the task's own objects:
  - `metadata.json`: bounding boxes and functional frames;
  - `mesh.npz`: NumPy triangle arrays per link, loadable with `trimesh.Trimesh(v, f)`;
  - `articulation.json`;
  - the original `.usdz` (open it with `pxr`).

  Use it to understand the task and objects and to plan motions. Do not edit it,
  run it or import it.
  - Your controller must still be closed-loop, driven only by images and robot
    state. It never has privileged object poses.
  - Static facts such as dimensions, functional offsets, joint travel and
    success thresholds may be copied into the bundle as constants.
  - Isolated runs and the official evaluation do not have this folder, and bundles
    must not import `pxr`.

## Notes and compaction

Write findings to `memory/` as you work. Keep `memory/PROGRESS.md` current after
useful experiments and stage transitions, covering:
- current stage and episode;
- confirmed results, failed approaches and lessons;
- working motion parameters;
- code, log and frame paths;
- bundle and evaluation IDs;
- remaining budget and the next concrete step.

Keep reusable motion tips in `memory/TIPS.md` following
[experience-memory](.agents/skills/experience-memory/SKILL.md). Store evidence on
disk and reference its path.

Your context is compacted automatically. After compaction, reread the notes and
check live `exploration_status`; do not restart an episode to reduce context. At handoff,
report the controller path, bundle, rehearsal and formal outcomes, unresolved
issues and artifact pointers.
