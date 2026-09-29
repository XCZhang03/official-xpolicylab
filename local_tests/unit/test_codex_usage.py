import json
from services.codex_usage import CodexUsageReader


def record(kind, payload):
    return json.dumps({"type": kind, "payload": payload}) + "\n"


def test_incremental_usage_no_double_count_and_partial_lines(tmp_path):
    path = tmp_path / "session.jsonl"
    reader = CodexUsageReader()
    path.write_text(record("token_usage_record", {"usage": {"input_tokens": 20, "output_tokens": 3},
                                               "thread_token_usage": {"total_tokens": 23}}))
    assert reader.read(path)["token_usage"]["total_tokens"] == 23
    assert reader.read(path)["token_usage"]["total_tokens"] == 23
    next_record = record("token_usage_record", {"usage": {"total_tokens": 7}, "thread_token_usage": {"total_tokens": 30}})
    with path.open("a") as stream:
        stream.write(next_record[:-1])
    assert reader.read(path)["token_usage"]["total_tokens"] == 23
    with path.open("a") as stream:
        stream.write("\n")
    result = reader.read(path)
    assert result["token_usage"]["total_tokens"] == 30
    assert result["total_session_usage"]["total_tokens"] == 30


def test_cumulative_fallback_is_not_summed_or_added_to_detailed_records(tmp_path):
    path = tmp_path / "session.jsonl"
    fallback = lambda n: record("event_msg", {"type": "token_count", "info": {"total_token_usage": {"total_tokens": n}}})
    path.write_text(fallback(20) + fallback(40))
    reader = CodexUsageReader()
    assert reader.read(path)["token_usage"]["total_tokens"] == 40
    with path.open("a") as stream:
        stream.write(record("token_usage_record", {"usage": {"total_tokens": 40}, "thread_token_usage": {"total_tokens": 40}}))
    assert reader.read(path)["token_usage"]["total_tokens"] == 40


def test_missing_usage_is_not_zero_and_truncation_resets(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(record("event_msg", {"type": "token_count", "info": None}))
    reader = CodexUsageReader()
    assert reader.read(path)["token_usage"] is None
    path.write_text("")
    assert reader.read(path)["total_session_usage"] is None
