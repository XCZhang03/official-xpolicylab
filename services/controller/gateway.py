"""MCP-only capability gateway. A session is bound by its trusted transport."""
from __future__ import annotations

import json
import secrets
import threading
import uuid

from .billing import reserve_cost
from .errors import public_failure
from .timing import Timings

from services.mcp_contract import Contract, ROBOT_TOOLS
# One MCP JSON-RPC line (base64 images included) on every controller transport.
MAX_REQUEST_BYTES = 12 * 1024 * 1024
MUTATING = frozenset({"robodojo_step", "robodojo_step_ee", "robodojo_step_eef", "robodojo_free_space_move", "robodojo_execute_motion_plan"})
PRIVATE = {"seed", "eval_seed", "formal_seed", "layout_id", "sim_gpu", "sim_port", "metadata", "environment_spec", "demonstration_context"}


GEMINI_TOOL = {
            "name": "gemini_generate", "description": "OpenRouter-style chat completion via Gemini 3.8 Flash: text/images, JSON schema output and client function calling. Returns the complete assistant message plus usage/cost/API budget; fixed $10 Gemini total per agent session, shared across episodes/phases. Call/token totals are informational, not additional usage caps. Per-request size/output limits still apply. Preserve assistant messages including reasoning_details when sending tool results. Tool calls are data only: your script validates and executes functions; the router never executes them. Images: {type: image, mimeType: image/png or image/jpeg, data: base64}, or image_url with an inline data URL. No paths, remote URLs or server tools. See gemini skill references/api.md.",
            "inputSchema": {"type": "object", "additionalProperties": False,
                "required": ["messages"], "properties": {
                    "messages": {"type": "array", "minItems": 1, "maxItems": 64,
                        "description": "Chat messages: system/user/assistant/tool. Tool results use tool_call_id and string content; preserve complete assistant messages.",
                        "items": {"type": "object"}},
                    "max_tokens": {"type": "integer", "minimum": 1, "maximum": 16384},
                    "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                    "response_format": {"type": "object", "description": "text, json_object, or json_schema with {name, strict, schema}."},
                    "tools": {"type": "array", "maxItems": 64, "items": {"type": "object"},
                        "description": "Client function declarations: {type: function, function: {name, description, parameters, strict?}}."},
                    "tool_choice": {"oneOf": [{"type": "string", "enum": ["auto", "none", "required"]}, {"type": "object"}],
                        "description": "Or {type: function, function: {name: declared_name}}. Multiple calls may be returned."},
                    "reasoning": {"type": "object", "description": "OpenRouter reasoning options: effort or max_tokens, optionally enabled/exclude. Provider support varies."},
                }},
        }


def sanitized(value):
    if isinstance(value, dict):
        return {k: sanitized(v) for k, v in value.items() if k not in PRIVATE and not any(
            part in k for part in ("path", "directory", "root", "artifact_storage")
        )}
    if isinstance(value, list):
        return [sanitized(v) for v in value]
    return value


def ended(value):
    if isinstance(value, dict):
        return value.get("episode_ended") is True or any(ended(v) for v in value.values())
    return isinstance(value, list) and any(ended(v) for v in value)


def complete(value):
    if isinstance(value, dict):
        return value.get("task_complete") is True or any(complete(v) for v in value.values())
    return isinstance(value, list) and any(complete(v) for v in value)


class Gateway:
    def __init__(self, backend, gemini=None, *, audit, charge=None, refund=None, budget=lambda: None,
                 on_success=lambda: None, publish_sequence=lambda value: None, contract=None):
        self.contract = contract or Contract.from_config(getattr(backend, "config", None), isolated=True)
        self.backend, self.gemini = backend, gemini
        self.audit, self.charge, self.refund = audit, charge, refund
        self.budget = budget
        self.on_success = on_success
        self.publish_sequence = publish_sequence
        self.call_id = self.request_id = None
        self._lock = threading.RLock()
        self._token = None
        self.terminal = False
        self.control_uncertain = False
        self.timings = Timings()

    def acquire(self):
        with self._lock:
            if self._token is not None:
                raise RuntimeError("Another controller owns the episode")
            self._token = secrets.token_hex(32)
            self.terminal = False
            self.control_uncertain = False
            return self._token

    def revoke(self):
        with self._lock:
            self._token = None

    def transfer(self, token):
        """Atomic handoff within the SAME episode; preserve terminal state."""
        with self._lock:
            if not self._token or not secrets.compare_digest(token, self._token):
                raise PermissionError("Expired control capability")
            self._token = secrets.token_hex(32)
            return self._token

    def definitions(self):
        return self.contract.select([t for t in self.backend.tools() if t["name"] in ROBOT_TOOLS] + [GEMINI_TOOL])

    def handle(self, token, request):
        params = request.get('params', {}) if isinstance(request, dict) else {}
        name = params.get('name') if isinstance(params, dict) else None
        if not isinstance(name, str) or (name not in ROBOT_TOOLS and name != 'gemini_generate'):
            name = 'protocol_or_rejected'
        if (name == 'robodojo_free_space_move' and isinstance(params.get('arguments'), dict)
                and params['arguments'].get('preview_only') is True):
            name = 'robodojo_free_space_move_preview'
        with self._lock, self.timings.measure(name):
            return self._handle(token, request)

    def _handle(self, token, request):
        """Handle one actual MCP JSON-RPC message, with no client-selected identity."""
        with self._lock:
            ident = request.get("id") if isinstance(request, dict) else None
            self.call_id, self.request_id = uuid.uuid4().hex, ident
            self.operation, self.stage = None, 'request'
            self.motion_dispatched = False
            try:
                if not self._token or not secrets.compare_digest(token, self._token):
                    raise PermissionError("Expired control capability")
                if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
                    raise ValueError("Invalid JSON-RPC message")
                method = request.get("method")
                if method == "notifications/initialized":
                    return None
                if method == "initialize":
                    result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "robodojo-controller", "version": "1.0"}}
                elif method == "tools/list":
                    result = {"tools": self.definitions()}
                elif method == "tools/call":
                    params = request["params"]
                    result = self.call(params["name"], params.get("arguments", {}))
                else:
                    raise PermissionError("Unsupported MCP method")
                return {"jsonrpc": "2.0", "id": ident, "result": result}
            except Exception as exc:
                if self.motion_dispatched:
                    self.terminal = self.control_uncertain = True
                failure = public_failure(exc, operation=self.operation, stage=self.stage,
                                         uncertain=self.control_uncertain)
                try:
                    self._audit({"event": "mcp_error", **failure})
                except Exception:
                    failure['audit_recording_failed'] = True
                try:
                    budget = self.budget()
                except Exception:
                    budget = None
                return {"jsonrpc": "2.0", "id": ident, "error": {
                    "code": -32000, "message": failure['reason'],
                    "data": {**failure, "call_id": self.call_id, "budget": budget}}}

    def _audit(self, event):
        self.audit({"call_id": self.call_id, "request_id": self.request_id, **event})

    def call(self, name, arguments):
        self.operation = name
        self.contract.require_robot(name, arguments)
        if self.terminal and name not in {"gemini_generate", "robodojo_status", "robodojo_observe"}:
            raise PermissionError('Control is uncertain after an earlier execution failure; further motion is blocked'
                                  if self.control_uncertain else 'Episode has ended; further motion is blocked')
        if not isinstance(arguments, dict):
            raise ValueError("Arguments must be an object")
        self._audit({"event": "mcp_request", "tool": name, "arguments": arguments})
        if name == "gemini_generate":
            self.stage = 'api_validation'
            if self.gemini is None:
                raise RuntimeError("Gemini is not configured")
            payload, reservation = self.gemini.prepare(arguments)
            cost_reservation = reserve_cost(reservation, payload['max_tokens'])
            self.stage = 'api_budget'
            self.charge(reservation, cost_reservation)  # Durable reservations before billable I/O.
            self.stage = 'provider_request'
            value, used = self.gemini.execute(payload, reservation)
            self.stage = 'api_accounting'
            self.refund(reservation - used, value["usage"], cost_reservation)
            value["budget"] = self.budget()
            images = []
        elif name in ROBOT_TOOLS:
            self.stage = 'validation'
            validator = getattr(self.backend, 'validate', None)
            if validator:
                validator(name, arguments)  # Never poison an episode for preflight rejection.
            mutating = name in MUTATING and not (name == 'robodojo_free_space_move' and arguments.get('preview_only') is True)
            self.stage = 'native_execution'
            self.motion_dispatched = mutating
            try:
                value, images = self.backend.call(name, arguments)
            except BaseException:
                if mutating:
                    self.terminal = True
                    self.control_uncertain = True
                raise
            try:
                self.stage = 'artifact_publication'
                sequence = self.publish_sequence(value)
            except BaseException:
                if mutating:
                    self.terminal = True
                    self.control_uncertain = True
                raise
            value = self.contract.observation.clean(sanitized(value))
            if sequence is not None:
                value["frame_sequence"] = self.contract.observation.clean(sequence)
            self.terminal = self.terminal or ended(value)
            if complete(value):
                self.on_success()
        else:
            raise PermissionError("Tool is outside controller capabilities")
        result = {"content": [{"type": "text", "text": json.dumps(value, allow_nan=False)}, *images]}
        self.stage = 'response_recording'
        self._audit({"event": "mcp_result", "tool": name, "result": result})
        return result
