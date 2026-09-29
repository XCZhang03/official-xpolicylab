"""Credential-holding OpenRouter router for gemini_generate, with no task/perception prompt logic.

The single implementation of the Gemini tool: the official AgentBundle bridge uses it
on a self-hosted remote policy server, and the harness gateway loads this same file
(services/controller/gemini.py) on the host. The key never enters a bundle.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import stat
import urllib.request

# OpenRouter prompt/completion caps are USD per million tokens; request/image caps are
# per unit. The provider refuses requests it cannot serve within these prices.
MAX_PRICE = {'prompt': 0.75, 'completion': 3.75, 'request': 0, 'image': 0.00000075}

MODEL = "google/gemini-3.8-flash"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_TOTAL_BYTES = 8 * 1024 * 1024
MAX_REQUEST_BYTES = 12 * 1024 * 1024
MAX_TEXT_BYTES = 256 * 1024  # Includes schemas, tool arguments/results and reasoning history.


def json_size(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode())


def validate_schema(schema):
    """Bound schemas without resolving references or imposing a provider's dialect."""
    if not isinstance(schema, dict) or json_size(schema) > 16384:
        raise ValueError('Schema must be an object of at most 16 KiB')
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {'$ref', '$dynamicRef', '$recursiveRef'} and (
                        not isinstance(child, str) or not child.startswith('#')):
                    raise ValueError('Only local schema references are allowed')
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(schema)


def function_name(value):
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value)


def validate_tool_calls(calls):
    if not isinstance(calls, list) or len(calls) > 64:
        raise ValueError('Invalid assistant tool_calls')
    for call in calls:
        if (not isinstance(call, dict) or set(call) - {'id', 'type', 'function', 'index', 'extra_content'}
                or call.get('type') != 'function' or not isinstance(call.get('id'), str) or not call['id']):
            raise ValueError('Invalid assistant function call')
        function = call.get('function')
        if (not isinstance(function, dict) or set(function) != {'name', 'arguments'}
                or not function_name(function['name']) or not isinstance(function['arguments'], str)):
            raise ValueError('Function arguments must be a JSON string')


def request_options(arguments):
    """OpenRouter inference fields only; function definitions are inert client data."""
    options = {}
    tools = arguments.get('tools', [])
    if not isinstance(tools, list) or len(tools) > 64:
        raise ValueError('Provide at most 64 function definitions')
    names = set()
    for tool in tools:
        if not isinstance(tool, dict) or set(tool) != {'type', 'function'} or tool['type'] != 'function':
            raise ValueError('Only client function tools are supported')
        function = tool['function']
        if (not isinstance(function, dict) or set(function)-{'name', 'description', 'parameters', 'strict'}
                or not function_name(function.get('name')) or function['name'] in names):
            raise ValueError('Invalid or duplicate function definition')
        names.add(function['name'])
        if 'parameters' in function:
            validate_schema(function['parameters'])
        if 'description' in function and not isinstance(function['description'], str):
            raise ValueError('Invalid function description')
        if 'strict' in function and type(function['strict']) is not bool:
            raise ValueError('Invalid function strict flag')
    if 'tools' in arguments:
        options['tools'] = tools
    if 'tool_choice' in arguments:
        choice = arguments['tool_choice']
        if isinstance(choice, str):
            if choice not in {'auto', 'none', 'required'} or (choice != 'none' and not tools):
                raise ValueError('Invalid tool_choice or missing tools')
        elif (not isinstance(choice, dict) or set(choice) != {'type', 'function'}
                or choice['type'] != 'function' or not isinstance(choice['function'], dict)
                or set(choice['function']) != {'name'}
                or not function_name(choice['function']['name']) or choice['function']['name'] not in names):
            raise ValueError('tool_choice must name a declared function')
        options['tool_choice'] = choice
    if 'response_format' in arguments:
        fmt = arguments['response_format']
        if not isinstance(fmt, dict) or fmt.get('type') not in {'text', 'json_object', 'json_schema'}:
            raise ValueError('Invalid response format')
        if fmt['type'] == 'json_schema':
            spec = fmt.get('json_schema')
            if (set(fmt) != {'type', 'json_schema'} or not isinstance(spec, dict)
                    or set(spec)-{'name', 'description', 'strict', 'schema'} or not function_name(spec.get('name'))):
                raise ValueError('Invalid response json_schema')
            validate_schema(spec.get('schema'))
            if 'strict' in spec and type(spec['strict']) is not bool:
                raise ValueError('Invalid schema strict flag')
            if 'description' in spec and not isinstance(spec['description'], str):
                raise ValueError('Invalid schema description')
        elif set(fmt) != {'type'}:
            raise ValueError('Unexpected response format fields')
        options['response_format'] = fmt
    if 'reasoning' in arguments:
        reasoning = arguments['reasoning']
        if not isinstance(reasoning, dict) or set(reasoning)-{'effort', 'max_tokens', 'enabled', 'exclude'}:
            raise ValueError('Invalid reasoning configuration')
        if 'effort' in reasoning and (not isinstance(reasoning['effort'], str)
                or reasoning['effort'] not in {'none', 'minimal', 'low', 'medium', 'high', 'xhigh'}):
            raise ValueError('Invalid reasoning effort')
        if 'max_tokens' in reasoning and (type(reasoning['max_tokens']) is not int
                or not 0 <= reasoning['max_tokens'] <= arguments.get('max_tokens', 2048)
                or 'effort' in reasoning):
            raise ValueError('Use effort or a reasoning token budget within max_tokens')
        if any(type(reasoning[key]) is not bool for key in ('enabled', 'exclude') if key in reasoning):
            raise ValueError('Invalid reasoning flag')
        options['reasoning'] = reasoning
    return options


def validate_image(image):
    """Validate supplied bytes without opening paths or fetching URLs."""
    from PIL import Image
    data, mime = image.get('data'), image.get('mimeType')
    formats = {'image/png': 'PNG', 'image/jpeg': 'JPEG'}
    if not isinstance(mime, str) or mime not in formats or not isinstance(data, str):
        raise ValueError('Provide a base64 PNG or JPEG image')
    if len(data) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValueError('Image exceeds 4 MiB')
    raw = base64.b64decode(data, validate=True)
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError('Image exceeds 4 MiB or is empty')
    with Image.open(io.BytesIO(raw)) as frame:
        if (frame.format != formats[mime] or frame.width * frame.height > 4096 * 4096
                or getattr(frame, 'n_frames', 1) != 1):
            raise ValueError('Unsupported image format, dimensions or animation')
        pixels = frame.width * frame.height
        frame.verify()
    with Image.open(io.BytesIO(raw)) as frame:
        frame.load()  # Reject incomplete compressed pixel data too.
    return raw, mime, pixels


def load_key(path=None):
    """Read an operator-selected private key file, or the supervisor environment.

    Never discover credentials in agent bundles or persist them in trial state.
    """
    if path is None:
        return os.environ.get("OPENROUTER_API_KEY") or None
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "r") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("OpenRouter key file must be a private, owner-readable regular file")
        key = stream.read(16385).strip()
    if not key or len(key) > 16384 or any(c.isspace() for c in key):
        raise ValueError("Invalid OpenRouter key file")
    return key


class GeminiRouter:
    def __init__(self, key: str, *, send=None):
        if not key:
            raise ValueError("OPENROUTER_API_KEY is required")
        self._key = key
        self._send = send or self._http

    def _http(self, payload):
        request = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(), headers={
            "Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
        })
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        with urllib.request.build_opener(NoRedirect).open(request, timeout=60) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("Provider response too large")
        return json.loads(raw)

    def prepare(self, arguments):
        """Validate inline images and preserve author prompts.

        No external URLs, server tools, provider override, key or arbitrary HTTP fields
        are accepted. The caller may supply any prompt and parse any response.
        """
        allowed = {"messages", "max_tokens", "temperature", "response_format", "tools", "tool_choice", "reasoning"}
        if not isinstance(arguments, dict) or set(arguments) - allowed:
            raise ValueError("Unsupported Gemini parameter")
        if json_size(arguments) > MAX_REQUEST_BYTES:
            raise ValueError('Gemini request exceeds 12 MiB')
        messages = arguments.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 64:
            raise ValueError("Provide 1..64 messages")
        maximum = arguments.get("max_tokens", 2048)
        if type(maximum) is not int or not 1 <= maximum <= 16384:
            raise ValueError("max_tokens must be 1..16384")
        payload = {"model": MODEL, "max_tokens": maximum, "messages": [],
                   "provider": {"require_parameters": True, "allow_fallbacks": False,
                                "max_price": dict(MAX_PRICE)}, "stream": False}
        image_reserve = 0
        image_count = 0
        image_bytes = 0
        options = request_options(arguments)
        text_bytes = json_size(options)
        for message in messages:
            if not isinstance(message, dict) or message.get('role') not in {'system', 'user', 'assistant', 'tool'}:
                raise ValueError("Invalid message")
            role = message['role']
            fields = {'role', 'content', 'name'}
            if role == 'assistant':
                fields |= {'tool_calls', 'reasoning', 'reasoning_details', 'refusal', 'annotations'}
                if 'tool_calls' in message:
                    validate_tool_calls(message['tool_calls'])
                if any(message.get(key) is not None and not isinstance(message[key], str)
                       for key in ('reasoning', 'refusal')):
                    raise ValueError('Invalid assistant metadata')
                if any(message.get(key) is not None and not isinstance(message[key], list)
                       for key in ('reasoning_details', 'annotations')):
                    raise ValueError('Invalid assistant metadata')
            if role == 'tool':
                fields.add('tool_call_id')
                if not isinstance(message.get('tool_call_id'), str) or not message['tool_call_id']:
                    raise ValueError('Tool results require tool_call_id')
                if not isinstance(message.get('content'), str):
                    raise ValueError('Tool result content must be a string')
            if set(message)-fields or ('name' in message and not function_name(message['name'])):
                raise ValueError('Invalid message fields')
            content = message.get('content')
            text_bytes += json_size({k: v for k, v in message.items() if k != 'content'})
            if content is None and role == 'assistant':
                payload['messages'].append(dict(message))
                continue
            if isinstance(content, str):
                text_bytes += len(content.encode())
                payload["messages"].append(dict(message))
                continue
            if not isinstance(content, list) or not 1 <= len(content) <= 32:
                raise ValueError("Invalid content")
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError("Invalid content block")
                if part.get("type") == "text" and set(part) == {"type", "text"} and isinstance(part["text"], str):
                    text_bytes += len(part["text"].encode())
                    parts.append(dict(part))
                elif part.get("type") in {"image", "image_url"}:
                    image_count += 1
                    if image_count > 8:
                        raise ValueError("At most eight images per request")
                    if part['type'] == 'image_url':
                        spec = part.get('image_url')
                        if set(part) != {'type', 'image_url'} or not isinstance(spec, dict) or set(spec) != {'url'}:
                            raise ValueError('Provide an inline image data URL')
                        url = spec['url']
                        match = re.fullmatch(r'data:(image/png|image/jpeg);base64,([A-Za-z0-9+/=]+)', url) if isinstance(url, str) else None
                        if not match:
                            raise ValueError('Only inline PNG/JPEG data URLs are accepted')
                        image = {'mimeType': match[1], 'data': match[2]}
                    else:
                        if set(part) != {'type', 'data', 'mimeType'}:
                            raise ValueError('Provide PNG/JPEG image bytes')
                        image = part
                    raw, mime, pixels = validate_image(image)
                    image_bytes += len(raw)
                    if image_bytes > MAX_IMAGE_TOTAL_BYTES:
                        raise ValueError('Images exceed 8 MiB total')
                    # Conservative reservation, not claimed tokenizer accuracy.
                    image_reserve += max(65536, pixels * 4)
                    parts.append({"type": "image_url", "image_url": {"url":
                        f"data:{mime};base64," + base64.b64encode(raw).decode('ascii')}})
                else:
                    raise ValueError("Only text and inline images are allowed")
            payload["messages"].append({**message, "content": parts})
        if text_bytes > MAX_TEXT_BYTES:
            raise ValueError("Text, schemas and history exceed 256 KiB")
        if "temperature" in arguments:
            temperature = arguments["temperature"]
            if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
                raise ValueError("Invalid temperature")
            payload["temperature"] = temperature
        payload.update(options)
        reservation = text_bytes + image_reserve + maximum + 32768
        return payload, reservation

    def execute(self, payload, reservation):
        response = self._send(payload)
        if response.get("model") != MODEL:
            raise ValueError("Provider returned an unexpected model")
        usage = response.get("usage", {})
        tokens = usage.get("total_tokens")
        if type(tokens) is not int or not 0 <= tokens <= reservation:
            raise ValueError("Missing or excessive provider usage; reservation retained")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("Invalid provider choices")
        message = choices[0].get("message", {})
        if not isinstance(message, dict):
            raise ValueError('Invalid provider message')
        # The response is data for the controller, never host code or actions.
        return {"model": MODEL, "message": message, "finish_reason": choices[0].get("finish_reason"),
                "usage": usage, "cost_usd": usage.get("cost")}, tokens


def _nanodollars(cost):
    from decimal import Decimal, ROUND_CEILING
    if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
        raise ValueError('Missing or invalid provider cost')
    return int((Decimal(str(cost)) * 1_000_000_000).to_integral_value(rounding=ROUND_CEILING))


def reserve_nanodollars(token_reservation, max_output_tokens):
    """Worst-case request cost under MAX_PRICE (input bound, output limit, images)."""
    prompt = token_reservation - max_output_tokens
    return (prompt * _nanodollars(MAX_PRICE['prompt']) // 1_000_000
            + max_output_tokens * _nanodollars(MAX_PRICE['completion']) // 1_000_000
            + 8 * _nanodollars(MAX_PRICE['image']))


class BudgetedGemini:
    """gemini_generate for a self-hosted official policy server: router plus a dollar cap.

    Each call reserves its worst-case cost before the request and settles to the
    reported cost afterwards; once the cap is reached, calls fail without contacting
    the provider. Thread-safe; one instance per policy-server process.
    """

    def __init__(self, router, *, limit_usd=10.0):
        import threading
        self.router, self.limit = router, _nanodollars(float(limit_usd))
        self.reported = self.reserved = 0
        self.calls = 0
        self._lock = threading.Lock()

    def budget(self):
        with self._lock:
            return {'session_cost_limit_usd': self.limit / 1e9, 'session_cost_reported_usd': self.reported / 1e9,
                    'session_cost_reserved_usd': self.reserved / 1e9, 'calls': self.calls,
                    'session_cost_remaining_usd': max(0, self.limit - self.reported - self.reserved) / 1e9}

    def __call__(self, arguments):
        payload, reservation = self.router.prepare(arguments)
        cost = reserve_nanodollars(reservation, payload['max_tokens'])
        with self._lock:
            if self.reported + self.reserved + cost > self.limit:
                raise RuntimeError('Gemini budget of this policy server is exhausted')
            self.reserved += cost
            self.calls += 1
        reported = cost  # A failed or unaccounted request keeps its worst-case charge.
        try:
            value, _ = self.router.execute(payload, reservation)
            reported = _nanodollars(value.get('cost_usd'))
        finally:
            with self._lock:
                self.reserved -= cost
                self.reported += reported
        value['budget'] = self.budget()
        return {'content': [{'type': 'text', 'text': json.dumps(value, allow_nan=False)}]}


def official_service(model_cfg):
    """BudgetedGemini from deploy.yml settings and the server's environment, or None.

    The key is read from the environment variable named by ``gemini_key_env`` (default
    OPENROUTER_API_KEY) or the private file named by ``gemini_key_file``. It is never
    part of the checkpoint.
    """
    if not model_cfg.get('gemini_enabled', True):
        return None
    path = model_cfg.get('gemini_key_file') or None
    key = load_key(path) if path else os.environ.get(model_cfg.get('gemini_key_env') or 'OPENROUTER_API_KEY')
    if not key:
        return None
    return BudgetedGemini(GeminiRouter(key), limit_usd=float(model_cfg.get('gemini_budget_usd', 10.0)))
