# RoboDojo local patches

`robodojo.patch` contains the existing robot-control adaptations.

`robodojo-pr48-drive-reset.patch` separately backports the proposed upstream
[PR #48](https://github.com/RoboDojo-Benchmark/RoboDojo/pull/48), commit `487e425`.
It was still unmerged when adopted. It restores authored spring drive targets
into PhysX, not just USD, for articulations configured with `reset_drive: true`.
It also initializes their positions from limit-clipped authored targets and
converts angular USD targets from degrees to radians. Missing/ambiguous target
mappings leave the previous behavior intact. Task rewards, thresholds, layouts,
and button assets are unchanged.

Both patches are checksum-pinned in `dependencies.lock` and applied idempotently
by `scripts/bootstrap_sources.sh`. The second patch is already applied to this
workspace's RoboDojo checkout. New simulator processes use it; old results are
not changed. Label comparisons against older results as different simulator
revisions. Revisit this backport when updating the upstream source pin.

Validation:

- `pytest local_tests/unit/test_articulation_drive_reset.py` checks target mapping.
- With `scripts/robodojo_env.sh` sourced, run
  `python scripts/smoke_button_reset.py --task swap_blocks --output <fresh-SSD-directory>`.
  Repeat with `--layout-seed 0` (default: 7), and for `press_by_number`, which
  shares the button asset. Use a fresh process/output directory for each layout,
  just as production does; upstream in-process multi-layout reloads can fail.
  Require `BUTTON_RESET_PASS` in the log and inspect `button-reset-probe.json`;
  Isaac shutdown can mask a Python exception in the process exit code.

The native probe checks three drive-actuated depress/release cycles per layout,
using unchanged native press-transition thresholds. It is an
operator-only physics diagnostic, not a robot-contact test or task-success
evaluation, and adds no agent-facing state manipulation interface.

Local verification (2026-09-27): both tasks passed on layouts 7 and 0 in fresh
simulators. All 12 depress/release cycles registered one native press transition;
normalized button height was 1.0 at reset/release and approximately 0.000743
while depressed. This is not a full task success-rate measurement.
Evidence is under `runtime/integration-tests/button-reset-` directories:
`TN7RvE` (Swap Blocks 7), `LCjc6z` (Swap Blocks 0), `tKYUAc` (Press by Number 7),
and `fXibni` (Press by Number 0).

Full recorded-action replay (2026-09-27): replayed the previously diagnosed
Swap Blocks exploration episode 4 from session `20260926T131236Z_1bb7f5c30c`,
layout 4 / evaluation collection 0, using `scripts/replay_swap_button_episode.py`.
The original 694-action trace had zero registered presses and nine pending checks.
With PR48, presses registered at steps 161, 379, and 588; all ordered checks passed
and native success/score 1.0 occurred at step 653/700. All 653 executed joint
targets exactly matched the original prefix; remaining actions were not executed
after native completion. Initial robot state matched. No scoring rules changed.
This is one diagnostic replay, not an averaged formal evaluation; the old session
records were not rewritten or granted rehearsal qualification.

Evidence: `runtime/diagnostics/swap-pr48-replay-gsyGGI/replay/`, containing
`replay-report.json`, `button-trace.jsonl`, `sim/evaluation_outcome.json`, and
`sim/sensors.mp4`. Generated artifacts use the existing SSD-backed runtime link.
