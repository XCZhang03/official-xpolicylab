# Learned memory format

Store learned skills only below `memory/skills/`. Use a lowercase
hyphenated slug.

## `TIPS.md`

Add a row immediately after a visibly successful and reusable local result:

```markdown
| Task phase | Preconditions | What worked | Visible effect | Evidence |
|---|---|---|---|---|
| grasp | gripper centered around handle | close, then lift in environment +Z | handle remained held | <returned frame manifest path> frame 3 |
```

Keep tips short, task-local, and evidence-linked. Update a matching row instead
of adding variants that repeat the same lesson.

## `INDEX.md`

Keep one compact table:

```markdown
| Skill | Tags | Confirmation | Modes | Last episode |
|---|---|---|---|---|
| [lift-after-grasp](skills/lift-after-grasp/SKILL.md) | grasp, lift | observed | exploration | <episode_id> |
```

`Confirmation` is `observed` for a clear local visual effect or
`evaluator-confirmed` when the procedure contributed to an episode accepted with
`task_complete: true`. `Modes` lists the distinct evidence modes as
`exploration`, `rehearsal`, `formal`, or a combination; only evaluator-confirmed formal
evidence is formal success.

## Generated `SKILL.md`

```markdown
---
name: learned-<slug>
description: <specific task or maneuver and when this experience applies>
---

# <name>

- Applies when: <observable conditions>
- Avoid when: <observable mismatch>
- Executed action: <tool and exact action representation>
- Pose recipe: <anchor, recipe representation, frame, delta>
- Actions: <compact ordered procedure>
- Expected effect: <visible result>
- Recovery: <evidence-backed alternative>
```

Do not copy global control limits, safety policy, or tool documentation into a
learned skill. Do not encode hidden state, native reward, inferred task success,
or an unobserved object center/pose. Record returned evaluation confirmation
with its episode mode in the evidence.

## `evidence.json`

```json
{
  "schema_version": 4,
  "skill": "learned-<slug>",
  "examples": [
    {
      "task": "<task>",
      "episode_id": "<episode_id>",
      "episode_mode": "exploration | rehearsal | formal (exploration = result mode interactive)",
      "confirmation": "observed | evaluator-confirmed",
      "manifest_path": "runtime/autonomous_controller/runs/<run>/frames/<sequence>/manifest.json",
      "step_ids": [1, 2],
      "frame_indices": [0, 1],
      "action_kind": "plan | approach | joint",
      "executed_action": {
        "tool": "robodojo_step",
        "representation": "absolute_joint_positions_gripper_14d_25hz",
        "joint_target_contract_version": 2,
        "rows": "transition.steps[*].executed_action"
      },
      "pose_recipe": {
        "representation": "mixed",
        "anchor": "<observable anchor semantics>",
        "segments": [
          {
            "representation": "absolute_environment",
            "anchor": "<fixed safe waypoint>",
            "position_m": [0.0, 0.0, 0.8],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]
          },
          {
            "representation": "relative_environment",
            "anchor": "<visible point>",
            "delta_frame": "environment",
            "position_m": [0.0, 0.0, 0.05],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]
          }
        ]
      },
      "point_reacquisition": {
        "method": "RGB visual alignment with measured robot poses",
        "visible_feature": "<feature description>",
        "preferred_views": ["cam_high"],
        "offset_m": [0.0, 0.0, 0.05]
      },
      "demonstration_provenance": {
        "source": "runtime/demonstrations/<id>/",
        "demonstration_id": "<id>",
        "kind": "completion_sequence | terminal_state",
        "demonstrator": "human | robot | unknown",
        "selected_frame_indices": [0, 10]
      },
      "observed_effect": "<concise visible change>"
    }
  ]
}
```

`episode_mode` is required for every example. Never relabel an exploration
confirmation as formal success when reusing it.

Use `absolute_joint_positions_gripper_14d_25hz` for the executed low-level
actions sent to `robodojo_step`, with `joint_target_contract_version: 2`.
The source is `transition.steps[*].executed_action`, not measured states or
the full planned sequence after partial execution. Pair commands with their
preceding `observation_step_id`. Keep requested EEF poses in `pose_recipe`;
they are goals, not the executed joint-action representation.
Do not treat older dense-execution endpoint records as version-2 replay data.

`relative_start_eef` means one transform expressed in the observed starting
`link6` frame and replayed as `T_target=T_current*T_delta`; it is not updated
continuously during the move. `relative_environment` means translation and
rotation deltas in fixed environment axes. For `absolute_environment`, omit
`delta_frame` and identify the fixed environment anchor.

For `mixed`, omit the single delta fields and store ordered `segments`, each
using one non-mixed representation. For point-relative recipes,
`point_reacquisition` records the visible feature, pose-independent offset,
RGB alignment procedure, and preferred views. When adapted from a
visual demonstration, `demonstration_provenance` records its workspace source,
visual form, demonstrator, and selected frames. It never records supplied action
or state data. Otherwise those fields may be `null`.
Preserve exact `robodojo_toolkit.pose_math` output; do not hand-edit quaternion components.
