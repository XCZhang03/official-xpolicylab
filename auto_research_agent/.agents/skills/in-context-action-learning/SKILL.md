---
name: in-context-action-learning
description: Robot-operation stage. Infer RoboDojo task goals and useful strategy from a supplied image sequence of a human or robot completing the task, or from a supplied terminal-state image. Use when visual examples should guide the current task.
---

# In-context action learning

Supplied demonstrations contain images only: either an ordered sequence of a
human or robot completing the task, or one terminal-state image. They contain
no actions, robot state, poses, calibration, or control timing. Use them for
visual intent and strategy, never as executable motion data.

The demonstration may use a different layout from the current episode. Use it
mainly to understand what the task requires and the intended final relationships,
not to copy exact positions or reproduce the image pixel-for-pixel. Fulfill every
rubric condition using the current layout and observations.
If a demo video is provided as multiple images, use the sequence to understand
the task phases, but do not imitate every demonstrated action. Choose simpler or
more convenient actions when they achieve the same required outcome in the
current layout.

## Retrieve

- Retrieve matching current-trial tips or learned experience for similar
  actions, then inspect supplied images under `runtime/demonstrations/`.
- View the images directly. For a sequence, preserve its supplied order and
  identify visible phases, contacts, object relations, and the final effect.
  Do not infer exact timing or interpolate missing phases.
- For a terminal image, infer only the desired visible end relationship; devise
  the intermediate robot behavior independently.

## Adapt

- Transfer semantic structure, such as approach side, contact order, grasp
  region, relative placement, and completion appearance. A human hand or a
  different robot does not define the current arm, pose, or trajectory.
- Ground all geometry again from current RoboDojo observations. Use the pixel
  and pose tools when needed; never estimate metric targets or action magnitudes
  from demonstration pixels alone.
- Construct new actions through the current MCP contracts.
- When several examples or adaptations remain plausible, use `action-search`
  to compare them from the same measured start state.

## Update from experience

- Associate the chosen visual strategy with its observed effect; do not repeat
  an ineffective interpretation from the same conditions.
- When starting a new exploration episode, retrieve the failed episode's decisive
  before/action/after evidence and revise the plan.
- Before registering the controller, turn relevant exploration evidence into
  executable logic. The script must reacquire current geometry rather than
  assuming the exploratory layout is identical. Rehearse before submitting.
- Keep this working set compact. Promote only reusable, evidence-backed results
  through the experience-memory skill.

When retaining demonstration evidence, record its kind (`completion_sequence`
or `terminal_state`), demonstrator (`human`, `robot`, or `unknown`), workspace
path, and selected frame indices using the experience-memory format.
