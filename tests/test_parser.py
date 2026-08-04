import json

import analyze_chat as ac

SAMPLE = str(ac.Path(__file__).resolve().parents[1] / "sample_data")


def test_numeric_key_orders_files_correctly():
    files = sorted(ac.Path(SAMPLE).glob("message_*.json"), key=ac.numeric_key)
    numbers = [ac.numeric_key(f) for f in files]
    assert numbers == sorted(numbers)


def test_find_thread_dirs_finds_sample():
    dirs = ac.find_thread_dirs(SAMPLE)
    assert len(dirs) == 1
    assert ac.Path(dirs[0]).resolve() == ac.Path(SAMPLE).resolve()


def test_decode_messenger_text_fixes_double_encoded_emoji():
    raw = "\\u00f0\\u009f\\u0091\\u008d"
    assert ac.decode_messenger_text(raw) == "\U0001f44d"


def test_decode_messenger_text_passes_plain_text_through():
    assert ac.decode_messenger_text("hello") == "hello"
    assert ac.decode_messenger_text("\u0645\u0631\u062d\u0628\u0627") == "\u0645\u0631\u062d\u0628\u0627"
    assert ac.decode_messenger_text(None) is None


def test_normalize_messages_sorts_and_decodes():
    raw = [
        {"sender_name": "A", "timestamp_ms": 2000, "content": "later"},
        {"sender_name": "B", "timestamp_ms": 1000, "content": "\\u00f0\\u009f\\u0091\\u008d"},
    ]
    msgs = ac.normalize_messages(raw)
    assert [m["ts_ms"] for m in msgs] == [1000, 2000]
    assert msgs[0]["content"] == "\U0001f44d"


def test_normalize_messages_handles_seconds_timestamp_fallback():
    raw = [{"sender_name": "A", "timestamp": 1483272000, "content": "hi"}]
    msgs = ac.normalize_messages(raw)
    assert msgs[0]["ts_ms"] == 1483272000000


def test_normalize_messages_reads_reaction_actor_and_reactor():
    raw = [
        {"sender_name": "A", "timestamp_ms": 1000, "content": "x",
         "reactions": [{"reaction": "\u2764", "actor": "B"}]},
        {"sender_name": "A", "timestamp_ms": 2000, "content": "y",
         "reactions": [{"reaction": "\u2764", "reactor": "C"}]},
    ]
    msgs = ac.normalize_messages(raw)
    assert msgs[0]["reactions"] == [("B", "\u2764")]
    assert msgs[1]["reactions"] == [("C", "\u2764")]


def test_longest_streak():
    from datetime import date
    assert ac.longest_streak([date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]) == 3
    assert ac.longest_streak([date(2020, 1, 1), date(2020, 1, 3)]) == 1
    assert ac.longest_streak([]) == 0


def test_anonymize_map_deterministic():
    msgs = ac.normalize_messages(
        [{"sender_name": n, "timestamp_ms": i * 1000, "content": "x"} for i, n in
         enumerate(["Bob", "Alice", "Bob", "Alice", "Alice"])]
    )
    mapping = ac.anonymize_map(msgs)
    assert mapping["Alice"].startswith("Person ")
    assert mapping["Bob"].startswith("Person ")
    assert len(set(mapping.values())) == 2
    assert mapping["Alice"] == "Person A"  # most messages first


def test_slug_ascii():
    assert ac._slug("Saturday Squad") == "saturday_squad"
    assert ac._slug("") == "thread"
