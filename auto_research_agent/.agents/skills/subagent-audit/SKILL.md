---
name: subagent-audit
description: Spawn a read-only reviewer subagent, on a cheaper model, that flags controller code likely to overfit the explored layouts and fail on held-out evaluation layouts. Use before registering a candidate final bundle, and again after substantial controller changes. Keep each auditor small and short-lived; retire it after its report.
---

# Subagent audit: overfitting to layouts

Formal and official evaluation run your bundle on layouts it has never seen (see
[autonomous-control](../autonomous-control/SKILL.md)). You wrote the controller
while looking at specific scenes, so you are the worst judge of which of its
assumptions only hold there. Hand the review to a subagent with fresh context.

## When

- Before every `register` of a bundle you may rehearse or submit.
- After any change made to fix one particular layout.
- When exploration results are good on some layouts and fail on others without a
  clear cause.

## Spawn the auditor

Use the native spawn tool's live schema. Do **not** fork your context: the auditor
should read the code without your beliefs about why it works.

**Spawn it on the cheaper model.** An audit is reading and reasoning over a few files,
and a cheaper model does it well for a fraction of the cost. You keep your own
model, and your own exploration subagents keep inheriting it.

- Claude Code: the `Agent` tool with `model: "sonnet"` (a fresh context by default).
- Codex: `spawn_agent` with `model: "gpt-6-sol"` and `fork_turns: "none"`.

This is the operator's explicit instruction for auditors. If the spawn tool rejects
the model, spawn with the default model instead and note it in memory.

Give it this brief, with the paths filled in:

```text
You are a read-only auditor. Find where this RoboDojo controller may overfit the
layouts it was developed on and fail on held-out layouts.

Read:
- controller code: /workspace/code/<project>/ (all files)
- task randomization: /workspace/task_source/source/task/RoboDojo/config/<task>.yml
  and /workspace/task_source/README.md (object sizes, frames, clutter)
- rubric: /workspace/TASK.md
- tested layouts and outcomes: /workspace/memory/ (PROGRESS.md, INDEX.md, notes)
- optionally a few frames listed in memory (initial/final images of tested runs)

Rules: do not call any MCP or robot tool, do not start or change episodes, do not
edit files outside /workspace/memory/audits/. Report only; the main agent decides.

Write /workspace/memory/audits/<UTC time>-overfit.md with findings ranked by risk.
For each: file:line, the assumption, why it can fail on a held-out layout (cite the
randomization range or instance variation), a concrete scene that would break it,
and the cheapest exploration test that would confirm or refute it.
```

Continue other work while it reads; do not let it touch the robot.

For a later review, fill in the delta brief below instead of the full reading list.

## Token budget: small context, short life

Every model call re-sends the auditor's whole conversation, and all agents share
one model-call cap for the session. Once the cap is exhausted the session ends
before the formal submission. A long-lived auditor that takes several follow-ups
can cost as much as hours of main-agent work, so keep each one small and brief.

- **One review, one auditor.** Spawn a fresh auditor (`fork_turns: "none"`) for
  each review and retire it once its report is written. Do not reuse a finished
  auditor with `followup_task` or `send_message` for the next registration or a
  new topic: its context only grows. The saved report carries what matters
  forward.
- **Give a delta brief after the first audit.** Point at the previous report and
  the reviewed controller hash, and list the changed files and functions (or the
  diff) plus the specific question. Do not ask for a re-read of the whole bundle,
  the task source and all of memory:

  ```text
  Read-only delta audit. Previous report: /workspace/memory/audits/<file>.md
  (reviewed SHA <sha>). Changed since: <files/functions or diff path>.
  Question: <the one decision this review should support>.
  Read only what the question needs; same rules and report format as before.
  ```
- **Put the whole question in the brief.** Do not chat with a running auditor
  or send it extra messages, because each message triggers more turns over the
  full context. If the question changes substantially, interrupt it and spawn a
  new one.
- **Wait, do not poll.** Use `wait_agent` with long timeouts (minutes), or keep
  working and read the report when it arrives. Run at most one auditor at a time
  unless the reviews are truly independent.
- **Shut down retired auditors.** When the report has been delivered, or the
  code it reviews has changed underneath it, the auditor is retired. If it is
  still running, call `interrupt_agent` on it. Never send it more work. Check
  `list_agents` before registering or spawning, and interrupt any stale auditor
  still active. If the live tool list offers a close or shutdown tool, use it.

Add this to every brief:

```text
Budget: keep your context small. Read with rg and bounded line ranges (sed -n,
head), never whole large files, logs or JSON dumps. View at most a few images.
Run offline checks as scripts that print short summaries. Write the full report
to the file; reply with at most 15 lines: verdict, top risks, report path. Then
stop. Do not wait for or ask follow-up questions.
```

## What the auditor checks

- **Literal layout values:** numeric positions, orientations, joint angles, pixel
  coordinates or crop boxes that match one explored scene rather than a derived or
  physical constant. Each allowed constant should state its source.
- **Coverage of the randomization:** placement and rotation ranges, object
  instances or models, and distractors in the task config, compared with what
  perception and motion handle. Flag untested regions: range edges, extreme
  rotations, the other arm's side, objects close together.
- **Perception fragility:** colour or size thresholds tuned to one scene; selecting
  objects by image position or detection order; no handling of zero, several or
  occluded detections; clutter that could match a detector.
- **Motion assumptions:** fixed approach heights or waypoints valid for one object
  pose; a single arm or grasp orientation where reachability depends on placement;
  planner failure paths without a fallback.
- **Sequencing assumptions:** success of every stage assumed without an
  observation check; fixed step or time budgets per stage; branches keyed to step
  counts or other episode identity.
- **Evidence:** how many distinct layouts each stage was actually verified on,
  according to memory; conclusions drawn from one scene.

## Act on the report

1. Read the report and triage each finding: fix it, test it, or record why it is
   not a risk.
2. Prefer fixing by deriving the value from observations or by widening coverage,
   not by adding another special case.
3. Test each fix and each "test it" item on fresh exploration layouts, running the
   code unchanged first. Record outcomes next to the finding.
4. Re-run the audit after substantial changes. Register and rehearse only when the
   remaining findings are resolved or knowingly accepted in memory.
