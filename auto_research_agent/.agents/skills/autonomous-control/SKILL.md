---
name: autonomous-control
description: Build, debug, register, rehearse and submit a frozen controller bundle that runs unchanged on the official RoboDojo evaluation. Use after manual success for controller code, exploration tests, isolated rehearsal and formal submission.
---

# Autonomous control

Follow the stages in AGENTS.md. The bundle you submit must run unchanged under the
official XPolicyLab evaluation. There, an adapter calls your `main(ctx)` and
executes each `robodojo_step` / `robodojo_step_ee` chunk in the real environment.
Isolated rehearsals and the formal batch run your bundle through that same adapter.

## Evaluation is on held-out layouts

Every formal episode, and the official evaluation, runs your frozen bundle on scene
layouts it has never seen: new object positions, orientations and, where the task
randomizes them, object instances. A controller that succeeds only on the layouts
you explored scores near zero. Build for the task's whole randomization range, not
for the scenes you happened to see.

- **No layout-specific values.** Never hard-code object positions, orientations,
  joint targets, image pixel regions or approach waypoints measured in an exploration
  scene. Derive every target from the current observation at run time.
- **Allowed constants** are facts that do not change between layouts: robot and
  camera geometry, object dimensions and functional offsets (for example a slot
  depth from `task_source/`), and success thresholds. Record where each constant
  comes from.
- **Know the randomization.** Read the task config in
  `task_source/source/task/RoboDojo/config/<task>.yml`: placement ranges
  (`xlim`/`ylim`), rotation ranges, object categories and instances, and any
  distractor clutter. Your perception and motion must cover all of it, including
  extremes such as objects near range edges or rotated to the limit.
- **Robust perception.** Colour and size thresholds need margin across lighting,
  distance and pose. Never select an object by its position in the image or by
  detection order alone; check identity (size, shape, colour) and handle a missing
  or ambiguous detection explicitly.
- **No shortcuts on episode identity.** Do not branch on step counts, timing,
  instruction wording or anything else that happens to correlate with one layout.
- **Recover, do not assume.** Check each stage's outcome in fresh observations and
  retry or re-plan on failure; a sequence that works only when every step goes as
  in your one test scene will fail elsewhere.
- **Audit before you freeze.** Before registering a candidate final bundle, run the
  [subagent audit](../subagent-audit/SKILL.md): a subagent reviews the
  code for layout-specific assumptions. Fix or test every finding in fresh
  exploration layouts.

## Bundle contract

```text
/workspace/code/<project>/
  controller.py        # def main(ctx): ...   (required entry point)
  *.py, data files     # anything main needs; load via paths relative to __file__
```

- **Tools:** use only `ctx.call("robodojo_observe")`, `ctx.call("robodojo_status")`,
  `ctx.call("robodojo_step", actions=rows)`, `ctx.call("robodojo_step_ee", actions=rows)`
  and `ctx.call("gemini_generate", ...)` (see [gemini](../gemini/SKILL.md)), plus
  `import robodojo_toolkit`. No other tool, network access or subprocess.
- **Gemini is optional at run time:** officially the bundle runs on our hosted agent
  API, which holds the key. Every Gemini-backed decision needs a fallback, and the controller
  must survive a failing or slow call.
- **Task source:** never read `/workspace/task_source` or import `pxr` from the
  bundle; neither exists in isolated or official runs. Copy the static geometry
  or thresholds you need into bundle constants or data files. Every pose must
  come from live observations.
- **Context:** never create `Context()` inside the bundle. The runner supplies
  `ctx`, and lifecycle tools are not available to it.
- **Outputs:** write to `ctx.output_dir`, never to fixed paths.
- **Episode end:** the episode may end at any step. Let the run stop when a call
  fails because the episode ended; the official adapter cancels the bundle then.
  Do not rely on seeing a success signal.
- **Planner:** build it once, at the start of `main`, with
  `tk.shared_planner()`. The official adapter has already warmed it by then.
- **Performance:** keep per-step computation quick. Planning takes about
  0.03 s; model-free perception in NumPy is fine.
- **Task selection:** one bundle serves one task. The submission tooling maps
  `task_name` to your bundle directory.

```python
# controller.py
import robodojo_toolkit as tk

def main(ctx):
    planner = tk.shared_planner()
    meta, images = tk.parse_reply(ctx.call("robodojo_observe"))
    ...   # perceive from images + meta, choose targets, plan / approach, step, re-observe
```

## From manual success to a controller

1. Read the successful procedure and decisive frames in `memory/`.
2. Turn one maneuver at a time into a function of `ctx` and explicit inputs.
   Re-acquire object positions from current observations; never embed one
   layout's coordinates.
3. Test each function in the current exploration scene with a development
   wrapper kept outside the bundle:

   ```python
   # /workspace/experiments/run_controller.py
   import sys
   sys.path.insert(0, "/workspace/code/<project>")
   from api.runtime import Context
   from controller import main
   with Context() as ctx:
       main(ctx)
   ```

   Run it with `python experiments/run_controller.py` from `/workspace`.
   To run a complete controller exactly as rehearsal and the official evaluation do
   (adapter context, action chunks, pose held after `main` returns until the
   episode ends), use `from api.runtime import Context, run_official` and
   `run_official(ctx, "/workspace/code/<project>")` instead of calling `main`.
4. Add observation-driven checks and recovery for expected failures: a missed
   grasp, a planning failure (try a nearer waypoint or another orientation), or
   an object moved by contact.
5. Compose the full `main(ctx)` and test it on fresh layouts.

Practice in the same episode by physically rearranging objects with the robot,
rather than resetting. Exploration outcomes do not count toward the score; the
formal batch does.

## Pressure-test across layouts

- **Coverage:** use all exploration episodes for development and robustness
  tests, not just enough for one success. Each new layout is a held-out test of
  the current code: run it unchanged first and record the outcome before debugging.
- **Record keeping:** record each tested layout, outcome, failure and fix in
  memory. Agent-assisted recovery is debugging, not autonomous success.
- **Generalization evidence:** a change that fixes one layout must be re-checked on
  other layouts. Track success across distinct layouts, not repeated runs of one.
- **Reserve:** keep enough episodes to rehearse the final frozen code after its
  last edit.
- **Blockers:** if the budget prevents a qualifying rehearsal, report that
  instead of bypassing the gate.

## Development details

- `ctx.call` returns MCP content blocks. Use `tk.parse_reply` for metadata and
  images. Errors raise `RuntimeError`; inspect the trace instead of retrying
  blindly.
- Scripts and your direct tool calls share the scene. Stop loops before
  intervening. Interrupting Python does not cancel a dispatched chunk; observe
  first.
- Robot replies, images and full 25 Hz frames are published automatically.
  `exploration_status` gives trace and final-image paths. Redirect stdout yourself if you
  want a saved log.

## Register, rehearse, submit

- **Register:** `register(source="code/<project>")` freezes the directory.
- **Rehearse:** `rehearse(bundle=...)` closes exploration and runs the frozen
  controller alone in a fresh container and episode, through the official
  adapter (see motion-toolkit `references/api.md`, "Under the official adapter").
  A `main` that raises fails the rehearsal. It uses the same wall
  limit as a formal episode (`exploration_status.episode_timeout_seconds`, startup
  included). `exploration_status.bundles` shows which bundles have a qualifying rehearsal.
  A timeout cannot qualify.
- **Submit:** only a successful rehearsal of the exact bundle permits `submit`.
  Edits need a new registration and rehearsal. Call `submit` once. It runs the
  held-out batch (50 episodes for new sessions) in fresh containers and
  environments, with no intervention, edits or retries.
- **Scoring:** native task success is required, not exit 0. Report successes
  divided by the fixed episode count; invalid or unconfirmed results are
  failures. An interrupted batch is incomplete.
- **Isolated-run files:** isolated runs read the bundle read-only at its original
  path and receive only their own frames. Files in `ctx.output_dir`
  (`/workspace/output`) are exported to
  `runtime/autonomous_controller/runs/<run-id>/exports/` after the container
  stops.
- **Results:** results point to the frozen code, stdout/stderr, `mcp.jsonl`,
  per-call responses and frame sequences. The formal reply includes
  `report_path` with per-episode outcomes.
