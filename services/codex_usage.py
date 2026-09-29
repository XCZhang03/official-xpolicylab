"""Incremental, operator-only token accounting from Codex rollout JSONL."""
import json
from pathlib import Path

KEYS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
        "output_tokens", "reasoning_output_tokens", "total_tokens")


def _usage(value):
    if not isinstance(value, dict):
        return None
    result = {key: value[key] for key in KEYS if isinstance(value.get(key), int)
              and not isinstance(value[key], bool) and value[key] >= 0}
    if not result:
        return None
    if "total_tokens" not in result:
        result["total_tokens"] = result.get("input_tokens", 0) + result.get("output_tokens", 0)
    return result


class CodexUsageReader:
    def __init__(self):
        self.identity = None
        self.offset = 0
        self.totals = None
        self.session = None
        self.fallback = None

    def read(self, path: Path):
        stat = path.stat()
        identity = (str(path), stat.st_dev, stat.st_ino)
        if identity != self.identity or stat.st_size < self.offset:
            self.__init__()
            self.identity = identity
        with path.open("rb") as stream:
            stream.seek(self.offset)
            while True:
                line = stream.readline()
                # A concurrent writer may not have finished its last record yet.
                if not line or not line.endswith(b"\n"):
                    break
                self.offset = stream.tell()
                if b'"token_usage_record"' not in line and b'"token_count"' not in line:
                    continue
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                payload = record.get("payload") or {}
                if record.get("type") == "token_usage_record":
                    usage = _usage(payload.get("usage"))
                    if usage is not None:
                        if self.totals is None:
                            self.totals = dict.fromkeys(KEYS, 0)
                        for key, value in usage.items():
                            self.totals[key] += value
                    self.session = _usage(payload.get("thread_token_usage")) or self.session
                elif record.get("type") == "event_msg" and payload.get("type") == "token_count":
                    info = payload.get("info") or {}
                    self.fallback = _usage(info.get("total_token_usage")) or self.fallback
        # Cumulative token_count records must not be summed or added to detailed
        # usage records. Fresh trial sessions permit a cumulative fallback.
        return {"token_usage": self.totals or self.fallback,
                "total_session_usage": self.session or self.fallback or self.totals}
