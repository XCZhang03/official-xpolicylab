---
name: experience-memory
description: Robot-operation stage. Save successful RoboDojo action tips and retrieve them for future similar actions with matching observable conditions within the current trial.
---

# Experience memory

Memory under `memory/` is local to this trial and persists across exploration
episodes. Retrieve matching tips when controlling the robot or building automation.
At task start or after compaction, read `memory/INDEX.md` if present and retrieve
only matching learned skills. Open linked evidence when applicability or outcome
is uncertain; do not load the entire history. Keep learned procedures under
`memory/skills/`, separate from the provided tool-contract skills.
Full rehearsal/formal scripts cannot read the editable workspace: incorporate
needed procedures into the frozen controller before registration.

## Record successful tips immediately

- As soon as an action produces a visibly successful, useful result for reuse
  in the same task, add or update a concise row in `memory/TIPS.md`.
  A visible local result is enough to save the tip; episode-level confirmation
  can be added later.
- Record the observable preconditions, task phase, tool/action or reusable pose
  relation, visible effect, and exact manifest/frame evidence. State what made
  it work so later attempts avoid repeating exploration or a superseded failure.
- Update the matching tip when the same maneuver is refined. Apply it to future
  similar actions with matching observable conditions.
- A tip records the visible local effect. Automatic evaluation supplies any
  episode-level task confirmation.

## Select

- Select successful experience with future reuse value. Motion manifests
  already preserve the complete executed actions, states, and ordered frames
  at the published paths returned by MCP under `runtime/autonomous_controller/runs/`.
- Promote a tip to a learned skill when it demonstrates a reusable motion,
  contact strategy, recovery, or completed task procedure.
- Reference the exact manifest, step IDs, and decisive frame indices; keep PNGs,
  trajectories and complete manifests at their published paths.
- Record the episode mode from the execution context/result in every example:
  direct exploration (result `mode: interactive`), isolated rehearsal, or formal. Label evaluator-returned
  `task_complete: true` with that mode. Exploration `true` confirms a useful
  strategy; formal `true` is task-level confirmation and the primary scored
  outcome. Otherwise describe the visible local effect.

## Encode poses

- Record the executed action representation separately from the reusable pose
  recipe. EEF and cuRobo requests remain absolute environment-frame `link6` poses;
  only the learned recipe may be relative. Prefer `relative_start_eef` plus `delta_frame: "local"` for a motion
  expressed once in the starting EEF frame; derive it with `relative_pose`.
- Store `relative_environment` with `delta_frame: "environment"` for a delta
  in fixed environment axes.
- Save an absolute environment pose only for a genuinely fixed robot/table
  waypoint and label it `absolute_environment`. Use a reacquired relative target
  for randomized objects.
- A mixed recipe may use an absolute safe waypoint plus relative manipulation
  steps; store ordered `pose_recipe.segments`. Reconstruct relative recipes with
  `compose_pose` and preserve its normalized quaternion output. The starting-EEF
  frame is fixed for that recipe, not a continuously moving control frame.
- If the anchor is only a visible surface point, store the offset and how the
  point was obtained under `point_reacquisition`.
- If a procedure was adapted from a demonstration, retain its source and
  selected evidence under `demonstration_provenance`.

## Distill

Create or revise `memory/skills/<slug>/SKILL.md` and `evidence.json`.
Keep procedures short; keep episode-specific values and evidence links in
`evidence.json`.

Update `memory/INDEX.md` with the skill path, task/maneuver tags,
confirmation level, distinct evidence modes, and last episode. A learned skill
may guide current control, but current perception and fixed MCP contracts take
precedence. Revise the skill when later evidence contradicts it; retain
provenance in evidence.

Read [references/format.md](references/format.md) only when creating or
updating a learned skill.
