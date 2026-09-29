---
name: gemini
description: Use Gemini 3.8 Flash (gemini_generate) inside controllers and interactively, for perception that must survive domain randomization and as a general-purpose reasoning agent with function calling. Covers when to use it, how to combine it with precise geometry, and how to pressure-test controllers that depend on it.
---

# Gemini

`gemini_generate` is a vision-language model you can call from a running controller
with `ctx.call("gemini_generate", ...)`, or interactively as a tool. You write the
prompts and parse the answers; the router adds no task logic. Request format, limits
and budget: [references/api.md](references/api.md).

## Why use it

- **It generalizes where hand-tuned perception breaks.** Colour thresholds, fixed
  crops and size filters are tuned on the scenes you saw. Held-out and `_random`
  layouts change object instances, add clutter and vary table, floor, lighting and
  background (see `TASK.md`). Gemini recognizes objects by what they are, so it keeps
  working under this domain randomization when a threshold silently fails.
- **It is less accurate per pixel.** Pixel coordinates and boxes from Gemini are
  coarse, often off by several to tens of pixels, and can vary between calls. Never
  send a Gemini pixel straight to a grasp or insertion.
- **It can serve as a general-purpose agent.** With function calling (`tools`),
  Gemini can decide *which* step to do next, choose between candidate objects or
  strategies, read an instruction variant, judge whether a stage succeeded from an
  image, and plan recovery. Your code defines the functions, validates every call and
  executes it; robot motion still goes through the toolkit and the step tools.

## Combine it with precise geometry

Use Gemini for the decisions that must generalize, and classical perception plus
closed-loop motion for precision:

1. **Identify and localize coarsely.** Ask for the target's identity and a rough
   point or box in one camera image, with a JSON schema.
2. **Refine locally.** Around Gemini's region, use colour, edges, known object size
   from `task_source/`, the calibrated camera model and the table plane to compute
   the precise position. Or move the wrist camera close and refine there.
   Points for the same named point on the target object in two or more views can
   be triangulated. First test whether Gemini marks the same physical point
   across views accurately enough for that
   ([RGB position calibration](../rgb-position-calibration/SKILL.md#triangulate-from-several-views)).
3. **Act closed loop.** Plan to an approach pose, then servo or step with
   measured feedback (motion-toolkit).
4. **Verify.** Ask Gemini (or check geometrically) whether the stage succeeded before
   moving on, and branch to recovery if not.

Prefer this split over either extreme: pure thresholds overfit the explored scenes,
and pure Gemini control is imprecise and slow.

## Make calls reliable

- **Structured output:** use `response_format` with a JSON schema and
  `temperature: 0`. Parse strictly, and check that the answer is complete
  (`finish_reason == "stop"`, no refusal).
- **Check plausibility:** reject coordinates outside the image or the table, a
  detected object whose size does not match `task_source/`, or contradictions with
  the previous observation. Re-ask or fall back rather than act on a doubtful answer.
- **Handle failure:** a call can fail (timeout, HTTP error, budget) with
  `RuntimeError`. Every Gemini use needs a fallback path, and a bounded retry at most.
- **Latency:** calls take seconds. Call at decision points, not every step. Under the
  official adapter the environment holds the pose while the bundle waits, and a wait
  over 90 s between chunks costs held steps.
- **Budget:** $10 per session across exploration, rehearsal and the formal batch.
  Downscale or crop images, keep prompts short, and count calls per episode × formal
  episodes before committing to a design.
- **Availability:** officially, the bundle runs on our hosted agent API, which holds
  the key. See references/api.md. Keep a non-Gemini
  fallback for every decision, so a bundle still runs, even if less robustly,
  without it.

## Pressure-test controllers that use Gemini

A Gemini-backed controller has failure modes a threshold controller does not. Test
them explicitly in exploration before registering:

- **Answer variance:** send the same image several times and measure the spread of
  the returned points or decisions. Design margins and refinement around the worst
  case, not the typical one.
- **Domain randomization:** test on layouts with different objects, clutter, table
  and lighting. On `_random` tasks, compare Gemini and threshold detection on the
  same frames and keep whichever holds up.
- **Failure injection:** in a development wrapper, make the Gemini call raise, time
  out, return malformed JSON or an implausible point, and confirm the controller
  falls back or recovers instead of acting blindly.
- **Budget and latency:** log calls, tokens and seconds per episode; confirm the
  formal batch fits the budget and that no chunk waits long enough to cost steps.
- Record these results in memory and include Gemini assumptions in the
  [subagent audit](../subagent-audit/SKILL.md).
