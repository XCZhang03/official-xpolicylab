# Gemini MCP API

## Request and response

Use `ctx.call("gemini_generate", **request)` inside Python, or the same arguments
with the interactive MCP tool. This is the supported subset of OpenRouter's
`POST /api/v1/chat/completions`, not a Google SDK wrapper. No SDK installation,
API key or network access is needed inside the container.

| Field | Contract |
| --- | --- |
| `messages` | Required, 1–64 messages; roles `system`, `user`, `assistant`, `tool`. |
| `max_tokens` | Output budget, 1–16384; default 2048. Reasoning also consumes tokens. |
| `temperature` | Optional number, 0–2. |
| `response_format` | `{"type":"text"}`, `{"type":"json_object"}`, or `json_schema` below. |
| `tools` | Up to 64 client function declarations; optional `description`, `parameters` JSON Schema and boolean `strict`. Names must be unique, 1–64 letters/digits/underscores/hyphens. |
| `tool_choice` | `auto`, `none`, `required`, or `{"type":"function","function":{"name":"add"}}`. A named choice must refer to a declared function; `auto`/`required` need nonempty `tools`. |
| `reasoning` | Optional OpenRouter object: `effort` **or** `max_tokens`, plus optional booleans `enabled`/`exclude`. `effort="low"` has been live-tested. Other settings depend on provider support. A reasoning token budget must fit within request `max_tokens`. |

The host fixes model `google/gemini-3.8-flash`, endpoint, non-streaming behavior
and `provider.require_parameters=true`. Unsupported parameters fail instead of
being silently dropped. No model/provider override, plugins, server tools, web
search, remote files or arbitrary HTTP fields are accepted. `parallel_tool_calls`
is not exposed: current endpoints reject it under strict parameter routing.
Multiple function calls can still appear in one response; process every call.

Unpack the MCP text block once:

```python
import json

def generate(ctx, **request):
    reply = ctx.call("gemini_generate", **request)
    result = json.loads(reply["content"][0]["text"])
    print("Gemini usage:", result["usage"], "budget:", result["budget"])
    return result
```

Result fields are `model`, `message`, `finish_reason`, `usage`, `cost_usd`, and
`budget`. `message` is the complete provider assistant message: text/null content,
optional `tool_calls`, refusal and reasoning metadata. `usage` retains provider
token/cost details. `cost_usd` is null if the provider omitted it. The budget
includes call counts, charged tokens, unresolved reservations and remaining dollars.
Call/token counters are informational, not additional session or phase caps.
Each generation, including a tool-result follow-up, consumes another API call.

### Where the tool is available

- **Development, rehearsal and the harness formal batch:** the host routes every call
  and enforces the session cap below.
- **Official evaluation:** the bundle runs on our hosted agent API
  (`official/xpolicylab/serve_endpoint.sh`). Its workers hold the key and its own cap (`gemini_budget_usd` in the
  adapter's `deploy.yml`), and the result's `budget` reports that cap instead. A
  server without a key answers every call with `RuntimeError` ("not configured"), so
  a bundle must still complete, or degrade gracefully, when Gemini is unavailable.

### Session spending cap

Gemini spending is limited to **$10 total per agent session**, shared by direct
agent calls, Python scripts, exploration, rehearsal and formal submission.
Episode resets and service restarts do not replenish it. Your own model usage is
separate. The host enforces this; scripts cannot change the limit.

`budget` and `exploration_status.api_budget` include:

| Field | Meaning |
| --- | --- |
| `session_cost_limit_usd` | Fixed total limit, 10. |
| `session_cost_reported_usd` | Cumulative confirmed provider cost across phases. |
| `session_cost_reserved_usd` | Pending or unresolved conservative cost reservations. |
| `session_cost_remaining_usd` | Amount available for another reservation. |
| `session_cost_blocked` | Fail-closed billing state; no further Gemini calls accepted. |

The host reserves a maximum cost **before** sending each request, applies fixed
provider price ceilings and disables provider fallbacks. Valid reported cost
releases the unused reservation. Missing/invalid cost or a timeout keeps the
reservation; it is not treated as free usage. Exhaustion rejects Gemini calls,
not otherwise-available robot control or local Python.

A large request can be rejected even with a positive remaining balance. Reduce
`max_tokens`, trim completed history or resize images so the conservative
reservation fits. Per-request size and output-token limits still apply.

An MCP/provider failure raises `RuntimeError` through `ctx.call`; failed billable
requests retain conservative token/dollar reservations if usage cannot be confirmed.
There is no automatic retry. Inspect the trace/budget before deciding to retry.
For text/JSON consumption, check `finish_reason == "stop"`, nonempty content and
no refusal. A `length` result is incomplete; do not execute partial decisions.

## Structured output

`json_object` requests JSON syntax; specify the desired object in your prompt.
Use `json_schema` to specify field shapes. This example can be combined with image
blocks from the skill's visual prompt:

```python
schema = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "visible": {"type": "boolean"},
    },
    "required": ["label", "visible"],
    "additionalProperties": False,
}
result = generate(ctx,
    messages=[{"role": "user", "content": "Report label='red block' and visible=true."}],
    response_format={"type": "json_schema", "json_schema": {
        "name": "object_detection", "strict": True, "schema": schema,
    }},
    max_tokens=1024,
)
assert result["finish_reason"] == "stop" and not result["message"].get("refusal")
detection = json.loads(result["message"]["content"])
assert set(detection) == {"label", "visible"}
assert isinstance(detection["label"], str) and type(detection["visible"]) is bool
```

The provider applies the schema; the router does not validate generated answers
against it. Validate types, ranges, coordinate frames and task meaning in your
script before using a response for motion. Valid JSON is not accurate perception.
For positions, state the camera, units and coordinate frame in both prompt and
field descriptions; do not confuse image pixels with robot coordinates.

Each response/function parameter schema is limited to 16 KiB. Local `$ref`
references such as `#/$defs/Point` are allowed; external references are rejected.
Schema dialect support is provider-dependent, including strict tool schemas.

## Client function calling

Functions are your script's interface to Gemini, not additional MCP tools. Gemini
requests a function by name and JSON-string arguments; your code decides whether
and how to run it. The following complete two-generation example uses only local
arithmetic. Substitute tested perception/motion/policy functions as appropriate:

```python
tools = [{"type": "function", "function": {
    "name": "add",
    "description": "Add two integers locally.",
    "parameters": {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    },
}}]
messages = [{"role": "user", "content": "Use add to compute 2+3, then report the result."}]
first = generate(ctx, messages=messages, tools=tools, tool_choice="required",
                 reasoning={"effort": "low"}, max_tokens=1024)
assert first["finish_reason"] == "tool_calls" and first["message"].get("tool_calls")
messages.append(first["message"])  # Preserve ALL fields/signatures unchanged.
for call in first["message"]["tool_calls"]:
    assert call["type"] == "function" and call["function"]["name"] == "add"
    args = json.loads(call["function"]["arguments"])
    assert set(args) == {"a", "b"} and all(type(v) is int for v in args.values())
    value = {"answer": args["a"] + args["b"]}
    messages.append({"role": "tool", "tool_call_id": call["id"],
                     "content": json.dumps(value)})
final = generate(ctx, messages=messages, tools=tools, tool_choice="none", max_tokens=1024)
assert final["finish_reason"] == "stop" and not final["message"].get("refusal")
print(final["message"]["content"])
```

Use `auto` for model-selected text/function replies, `required` to request at least
one function, or a named choice to select a specific function. Use `none` for the
final answer; that follow-up can also set `response_format` for structured JSON.
Repeat generations for a longer closed loop with an explicit iteration/budget
bound. Always append every tool result with its matching `tool_call_id`.

Keep an explicit dispatch map of permitted Python functions and validate their
arguments. Never `eval` model output or dispatch arbitrary function/MCP names.
For robot functions, call native MCP through `ctx`, examine feedback and stop at
episode end or ambiguous motion failure. Multiple proposed calls do not authorize
concurrent robot actions; execute them in the order your controller requires.

## History, images and limits

Requests are stateless: maintain `messages` in Python and send the needed history
each time. Append `result["message"]` intact, particularly `reasoning_details`
(Gemini thought signatures). Do not regenerate, trim or reorder those fields
within a tool round trip. Plain text history uses `role` and string `content`.
Assistant function requests can have null/omitted content. Tool results require
string content, usually compact JSON; attach new images in user image messages.

Images accept the MCP block `{type:"image", mimeType:"image/png", data:base64}`
or OpenRouter-style `{type:"image_url", image_url:{url:"data:image/png;base64,..."}}`.
JPEG uses `image/jpeg`. Both forms undergo identical byte/image validation;
`image_url` does not permit HTTP(S), file paths or remote fetching. Encode local
or in-memory images with Python; use the skill's example. Camera observation
image blocks can be passed directly. No observation IDs are needed.

Limits: 32 content blocks per message; 8 images/request; 4 MiB/image, 8 MiB combined;
at most 4096×4096 pixels per image; PNG/JPEG, no animation. Combined non-image
text, schemas, tool arguments/results and reasoning history: 256 KiB. Entire
request: 12 MiB. Preserve only relevant completed turns to keep within limits.

## Compatibility evidence

On 2026-09-24, eight cases passed both direct OpenRouter requests and the production
MCP gateway: JSON mode, JSON Schema, required/named/auto tool selection, a
tool-result follow-up with JSON Schema, two function calls in one response, and
image-to-schema output. This verifies transport/format behavior, not robot-task
reasoning or perception accuracy. `parallel_tool_calls` returned HTTP 404 under
strict parameter routing and is intentionally not exposed.

This local reference contains the supported API subset, examples and limits;
no external documentation is needed. It reflects the endpoint tests above.
