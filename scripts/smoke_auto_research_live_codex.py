"""Operator-only, bounded paid Codex/MCP smoke; artifacts stay on SSD."""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess

from harness.codex_cli.auto_research import deploy, launch
from services.controller.config import Configuration, Limits
from scripts.smoke_common import add_gpu_arguments, gpu_uuid, integration_root


PROMPT = """Run a bounded infrastructure smoke, NOT a task-solving attempt. The operator
explicitly authorizes programming before manual task success for this infrastructure test.
Never bypass or forge the real successful-rehearsal requirement for submit. There are only
12 model requests available: batch independent inspection and safe tool calls using exec.
Discover the complete MCP inventory and exercise EVERY discovered MCP tool once if safely
possible. Maintain /workspace/memory/smoke-coverage.json with each tool, outcome, evidence,
and exact error; finish with a concise coverage report. Missing coverage is not a pass.

Read AGENTS.md, TASK.md and relevant skills. Use native Bash/Python/apply_patch/image tools
inside this offline container. Verify /.dockerenv, no Docker socket, no host home or
simulator package or provider credential, internet TCP denied, and exactly one CUDA GPU.
Check df on /workspace and /codex-home: both share a 2 GiB persistent filesystem cap.

Call status, then start_episode and inspect the returned image(s) using image() forwarding.
Read robot status/observe and keep actual current measured joint and EEF pose values.
Use only a current-pose hold or a very small (<1 cm) safe free-space displacement.
Exercise robodojo_pixel_to_position on a valid visible pixel, robodojo_pose_math,
robodojo_step (current joints), robodojo_step_eef (current absolute EEF poses),
robodojo_fk_preview, robodojo_free_space_move preview_only=true include_trajectory=true,
robodojo_execute_motion_plan. Observe/read-only calls should not stale the cached plan;
no mutating calls between planning and executing that plan. Inspect goal_reached separately.
Avoid malformed mutating calls: they poison the episode. At most four exploration episodes.

Make ONE real gemini_generate image request, passing an image content block returned
by MCP (type, mimeType, data), asking to describe visible objects in one sentence,
max_tokens=128. Do not print base64 into context; forward the block programmatically. Inspect returned
usage/cost/budget; maximum two API calls, no uncontrolled retry or external endpoints.

Run ordinary Python inside this existing container, using from api.runtime import Context
and with Context() as ctx to connect while the agent MCP connection stays open. Observe,
execute a measured-joint hold through local RecordingContext (joint, image_size=64),
finish a segment, and save logs. Inspect the current scene via direct agent MCP. Run Python
again to record a second segment, concatenate locally, and inspect HDF5 keys/shapes plus
preview PNG with view_image. Verify the episode ID stays unchanged across script/agent
handoff. No special operate or recorder MCP tools exist. Test discarding a local recorder
without resetting. Call evaluate for trusted real task state.

Write a valid minimal controller.py main(ctx) that observes, holds current joints for one
25 Hz step, prints a marker, and returns; no actual task solving. Register the bundle and
rehearse it end to end in a fresh exploration episode. Inspect artifacts/final images.
Call submit on that exact bundle to verify REJECTION because rehearsal did not achieve
task success. Do not forge success, modify private state or consume a formal attempt.
Check status shows formal_reserved=false. Call finish and final status to verify cleanup.
If any tool fails, record exact error, continue safe unrelated checks; one bounded retry
with corrected documented arguments is okay except provider failures. Do not train policies
or attempt a successful robot task. Leave all code/logs and coverage JSON in workspace.
"""


FOCUSED_PROMPT = """Run ONLY this targeted infrastructure regression, not a task-solving attempt.
The operator explicitly permits programming before manual success for this smoke. Preserve
the exact successful-rehearsal requirement; never forge success or run positive formal.
Only SIX model requests are available: batch calls, keep outputs short, do not repeat the
broad smoke suite. Read TASK.md and the pose-math skill plus its exact reference if needed;
other skills are unnecessary for this minimal controller. Discover exact MCP names.

1. Call start_episode, then robodojo_observe and inspect its RGB image. Use measured left
EEF pose in a VALID robodojo_pose_math request: operation='format_target',
target_kind='step_eef', gripper_closed=false (REQUIRED even with gripper_opening),
pose={position_m:[x,y,z],rotation:{representation:'quaternion_wxyz',value:[w,x,y,z]}}.
Verify target returned successfully; do not move the robot.
2. Call evaluate TWICE at the same unchanged step. Both must succeed and return the same
trusted task_complete=false result without resetting or moving.
3. Write /workspace/code/focused/controller.py containing exactly a main(ctx) function
that calls ctx.call('robodojo_observe'), prints 'FOCUSED_REHEARSAL_OBSERVE_OK' and returns.
No policy package, training, actions, APIs or extra dependencies needed. Register this
directory and call rehearse with the returned bundle ID. The bug under test was duplicate
same-step evaluation during evaluate->rehearse. Rehearsal MUST now actually start a NEW
episode, execute the controller, return returncode=0, task_complete=false, final images
and stdout/MCP artifacts. View one final image and read the stdout marker using native tools.
4. Submit the exact bundle; expect rejection because its actual rehearsal was unsuccessful.
Check status formal_reserved=false, exploration_remaining=0 after the two episodes.
Call finish, then status; active_episode must be null. Do not repeat failed provider calls.
Write /workspace/memory/focused-regression.json with per-call outcomes/evidence, images,
evaluation identity/step, actual rehearsal result, rejection and cleanup. Final report must
distinguish tooling regression success from task success (task success is NOT expected).
"""


MINIMAL_CONTROLLER = 'def main(ctx):\n    ctx.call("robodojo_observe")\n    print("REHEARSAL_OK")\n'
POLICY_PROMPT = """Run a bounded hands-on infrastructure smoke, not a task-solving attempt.
The operator explicitly permits programming before manual success for THIS test.
There are at most TWELVE model requests and approximately TEN minutes. Batch related
reads/actions with exec, keep outputs short, and finish a report even if something fails.
No Gemini calls, register, rehearse, submit, downloads, installs, simulator access, or
existing pytest/smoke scripts. Author and use normal Python yourself inside /workspace.

Read AGENTS.md and the autonomous-control and policies skills, including data/actions
and training references. Check /.dockerenv, Python package versions, CUDA device and
/workspace capacity. Copy the programmable policy package and licenses into code/policy_smoke.
Discover the native robot MCP names and exact schemas. Start ONE exploration episode,
inspect the real image and measured joints. Only send holds at actual current measured
joint targets; do not execute learned predictions, free-space plans, or task motions.

Write and run ordinary Python importing api.runtime.Context in the existing container.
The agent's MCP connection must remain open. Use the documented local RecordingContext
to collect at least two current-joint hold chunks, saving genuine current-episode RGB,
state and joint action data. First run Python for one segment, inspect the current scene
through direct agent MCP, then run Python again for a contiguous second segment. Prove
the same episode continues and step count advances. Concatenate local segments, inspect
the HDF5 keys/shapes, and produce/view a PNG from the actual saved training image.
The agent can choose action numbers directly from observations; no synthetic demos.

Use your editable robot_policy package to train tiny ACT and DP models on this real
recording, CUDA, 2 optimizer steps each, batch_size=1, validation_fraction=0, horizon=4,
hidden=32, layers=1, latent=4, diffusion_steps=2. Keep data images small (64px) and use
the matching number of cameras. Reload both checkpoints using documented inference
interfaces and produce finite predicted joint chunks of the correct shape, but NEVER
execute these untrained outputs. Inspect training metrics/status and monitor HTML.
This proves plumbing only, not learned task success. If time remains, check one local
pretrained backbone loads; ACT+DP and real recording take priority over extras.

Use native shell/Python and file editing tools; use the image viewer on a saved PNG.
Do not hide failures or edit framework/skills to make tests pass. Record exact errors,
API/instruction friction and missing checks in memory/policy-smoke-report.json plus a
short memory/policy-smoke-report.md. Preserve your source, logs, dataset, checkpoints,
metrics and preview PNG under code/policy_smoke. Finally call finish, then status,
verify no active episode. End with an evidence-based report, not a generic assurance.
"""
MINIMAL_PROMPT = """Run ONLY the minimal infrastructure regression below. The operator explicitly
permits programming before manual success for this smoke. Do not solve the task or forge
success. Only FOUR model requests available, so batch tool calls. No start_episode,
evaluate, Gemini, pose tools, training, broad tool suite or skill rereads are needed.
Use native file editing to write /workspace/code/minimal/controller.py EXACTLY:

""" + MINIMAL_CONTROLLER + """
The SDK call signature is call(tool, **arguments): do NOT add a positional {}.
1. register source='code/minimal', then rehearse the returned bundle. This should start
the one available native exploration episode and return reason='exit', returncode=0,
task_complete=false, with saved stdout and inline final images. Forward one inline image
with image(block), not by embedding base64 in shell arguments. Read published stdout
using native shell and confirm REHEARSAL_OK. Do not re-register or change code afterward.
2. submit the exact bundle: expect rejection because the rehearsal did not solve the task.
Call finish and status, confirm active_episode=null and formal_reserved=false. Do not
attempt positive formal or retry any failed provider request.
Write memory/minimal-regression.json with actual tool results, marker and image evidence,
rehearsal return code/task result, expected submit rejection, final status. Return concise
PASS/FAIL for this infrastructure regression, never claim task-solving success.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--focused-lifecycle', action='store_true')
    group.add_argument('--minimal-rehearsal', action='store_true')
    group.add_argument('--policy-hands-on', action='store_true')
    add_gpu_arguments(parser)
    args = parser.parse_args()
    if args.minimal_rehearsal:
        # Validate the exact supplied code against the real SDK with a local MCP fixture.
        from io import StringIO
        from auto_research_agent.api.runtime import Context
        response = {'jsonrpc': '2.0', 'id': 1, 'result': {'content': []}}
        output = StringIO()
        context = Context(StringIO(json.dumps(response)+'\n'), output)
        namespace = {}
        exec(compile(MINIMAL_CONTROLLER, 'minimal-controller.py', 'exec'), namespace)
        namespace['main'](context)
        assert json.loads(output.getvalue())['params'] == {'name': 'robodojo_observe', 'arguments': {}}
    root = integration_root() / (
        'live-codex-mcp-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    image = subprocess.check_output(['docker', 'image', 'inspect', 'robodojo-auto-research:dev',
                                     '--format', '{{.Id}}'], text=True).strip()
    limits = Limits(wall_seconds=300, memory_mb=8192, cpus=4, pids=512,
                    scratch_mb=512, artifact_bytes=128*1024**2)
    config = Configuration(root=root, image=image, task='make_kong',
        exploration_seeds=(0,) if args.minimal_rehearsal else ((0, 1) if args.focused_lifecycle else (0, 1, 2, 3)),
        formal_seed=1 if args.minimal_rehearsal else (2 if args.focused_lifecycle else 4), eval_seed=0,
        sim_gpu=gpu_uuid(args.sim_gpu),
        controller_gpu=gpu_uuid(args.research_gpu),
        training_gpu=gpu_uuid(args.research_gpu),
        development=limits, formal=Limits(**{**asdict(limits), 'wall_seconds': 60}) if args.policy_hands_on else limits,
        training=limits,
        workspace_mb=256 if args.minimal_rehearsal else 2048)
    # No Gemini credential for smoke modes that must not issue paid API calls.
    key = (None if (args.focused_lifecycle or args.minimal_rehearsal or args.policy_hands_on)
           else Path.home()/'.codex/secrets/openrouter_api_key')
    if key is None:
        os.environ.pop('OPENROUTER_API_KEY', None)
    deployed = deploy(config, provider_home=Path.home()/'.codex', key_file=key)
    prompt = POLICY_PROMPT if args.policy_hands_on else (MINIMAL_PROMPT if args.minimal_rehearsal else (FOCUSED_PROMPT if args.focused_lifecycle else PROMPT))
    (root/'smoke-prompt.txt').write_text(prompt)
    print(json.dumps({'root': str(root), 'configuration': asdict(config)}, default=str), flush=True)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    result = launch(config, deployed, key_file=key, prompt=prompt, capture=True,
                    max_model_calls=4 if args.minimal_rehearsal else (6 if args.focused_lifecycle else 12))
    (root/'codex-stdout.jsonl').write_text(result.stdout)
    (root/'codex-stderr.log').write_text(result.stderr)
    print(json.dumps({'returncode': result.returncode, 'root': str(root)}), flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
