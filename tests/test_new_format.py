import json
from datetime import datetime
from pathlib import Path

import pytest

import analyze_chat as ac

SAMPLE = str(Path(__file__).resolve().parents[1] / "sample_data")


def _raw_msg(**kw):
    base = {
        "id": "1", "sender_name": "Alice", "timestamp_ms": 1609459200000,
        "content": "hello", "type": "Generic",
    }
    base.update(kw)
    return base


def test_new_format_normalization():
    raw = [
        _raw_msg(id="1", sender_name="Alice", content=None,
                 gifs=[{"uri": "gifs/a.gif"}]),
        _raw_msg(id="2", sender_name="Bob", content=None,
                 videos=[{"uri": "videos/v.mp4"}]),
        _raw_msg(id="3", sender_name="Charlie", content=None,
                 audio_files=[{"uri": "audio/s.opus"}]),
        _raw_msg(id="4", sender_name="Dana", content=None,
                 files=[{"uri": "files/doc.pdf", "name": "doc.pdf"}]),
        _raw_msg(id="5", sender_name="Alice", content=None,
                 polls=[{"question": "Where should we eat?"}]),
        _raw_msg(id="6", sender_name="Bob", content="old text",
                 is_taken_down=True),
    ]
    msgs = ac.normalize_messages(raw)
    by_id = {m["id"]: m for m in msgs}
    assert by_id["1"]["has_gif"] and by_id["1"]["gif_uris"] == ["gifs/a.gif"]
    assert by_id["2"]["has_video"] and by_id["2"]["video_uris"] == ["videos/v.mp4"]
    assert by_id["3"]["has_audio"] and by_id["3"]["audio_uris"] == ["audio/s.opus"]
    assert by_id["4"]["has_file"] and by_id["4"]["file_uris"] == ["files/doc.pdf"]
    assert by_id["4"]["file_names"] == ["doc.pdf"]
    assert by_id["5"]["poll_question"] == "Where should we eat?"
    assert by_id["5"]["content"] == "Where should we eat?"
    assert by_id["6"]["is_taken_down"]
    assert all(m["has_media"] for m in (by_id[str(i)] for i in range(1, 5)))
    assert not by_id["5"]["has_media"]
    assert not by_id["6"]["has_media"]


def test_media_leaderboard_counts_new_kinds():
    raw = [
        _raw_msg(id="1", sender_name="Alice", gifs=[{"uri": "a.gif"}]),
        _raw_msg(id="2", sender_name="Alice", videos=[{"uri": "v.mp4"}]),
        _raw_msg(id="3", sender_name="Bob", files=[{"uri": "f.pdf"}]),
        _raw_msg(id="4", sender_name="Bob", audio_files=[{"uri": "s.opus"}]),
        _raw_msg(id="5", sender_name="Bob", sticker="sticker.png"),
    ]
    media = ac.media_leaderboard(ac.normalize_messages(raw))
    assert media["gifs"]["Alice"] == 1
    assert media["videos"]["Alice"] == 1
    assert media["files"]["Bob"] == 1
    assert media["audio"]["Bob"] == 1
    assert media["stickers"]["Bob"] == 1


def test_load_thread_dedupes_ids(tmp_path):
    d = tmp_path / "thread"
    d.mkdir()
    dup = {"id": "dup1", "sender_name": "Alice", "timestamp_ms": 1609459200000,
           "content": "twice"}
    (d / "message_1.json").write_text(json.dumps(
        {"title": "T", "participants": [{"name": "Alice"}],
         "messages": [dup, dict(dup)]}), encoding="utf-8")
    (d / "message_2.json").write_text(json.dumps(
        {"title": "T", "participants": [{"name": "Alice"}],
         "messages": [dict(dup), {"id": "b", "sender_name": "Bob",
                                  "timestamp_ms": 1609459200001,
                                  "content": "other"}]}), encoding="utf-8")
    title, participants, msgs = ac.load_thread(d)
    assert title == "T"
    assert len(msgs) == 2
    assert [m["id"] for m in msgs] == ["dup1", "b"]


def test_load_thread_dedupes_idless(tmp_path):
    d = tmp_path / "thread"
    d.mkdir()
    (d / "message_1.json").write_text(json.dumps(
        {"title": "T", "messages": [
            {"sender_name": "Alice", "timestamp_ms": 1, "content": "x"},
            {"sender_name": "Alice", "timestamp_ms": 1, "content": "x"},
        ]}), encoding="utf-8")
    _, _, msgs = ac.load_thread(d)
    assert len(msgs) == 1


def test_check_flag_reports_and_exits_zero(tmp_path):
    thread = tmp_path / "thread"
    thread.mkdir()
    (thread / "message_1.json").write_text(json.dumps(
        {"title": "thread", "messages": [
            {"id": "1", "sender_name": "Alice", "timestamp_ms": 1609459200000,
             "content": "hi", "type": "MysteryType", "mystery_field": True,
             "photos": [{"uri": "photos/gone.jpg"}]},
            {"id": "1", "sender_name": "Alice", "timestamp_ms": 1609459200000,
             "content": "hi", "type": "MysteryType", "mystery_field": True,
             "photos": [{"uri": "photos/gone.jpg"}]},
            {"id": "2", "sender_name": "Bob", "timestamp_ms": 1609459200000,
             "type": "Generic"},
        ]}), encoding="utf-8")
    out = tmp_path / "out"
    code = ac.main(["--input", str(tmp_path), "--output", str(out),
                    "--check", "--json"])
    assert code == 0
    report = json.loads((out / "thread" / "check.json").read_text(encoding="utf-8"))
    assert report["messages"] == 3
    assert report["unknown_types"] == ["MysteryType"]
    assert "mystery_field" in report["unknown_keys"]
    assert report["duplicates"] == 1
    assert report["empty_messages"] == 1
    assert report["missing_media"][0]["uri"] == "photos/gone.jpg"


def test_year_review_pages_generated(tmp_path):
    code = ac.main(["--input", SAMPLE, "--output", str(tmp_path), "--json"])
    assert code == 0
    out = tmp_path / "saturday_squad"
    assert (out / "year_in_review.html").exists()
    year_files = sorted(out.glob("year_*.html"))
    assert len(year_files) >= 2
    page = (out / "year_2017.html").read_text(encoding="utf-8")
    assert "2017 in review" in page
    assert "data:image/png;base64," in page
    index = (out / "year_in_review.html").read_text(encoding="utf-8")
    assert "year_2017.html" in index


def test_reader_serves_year_page(tmp_path):
    import socket
    import threading
    import urllib.request

    import chat_ui
    import analyze_chat as ac

    from server_utils import wait_for_server

    out = tmp_path / "out"
    code = ac.main(["--input", SAMPLE, "--output", str(out)])
    assert code == 0

    msgs = []
    for i in range(1, 5):
        data = json.load(open(f"{SAMPLE}/message_{i}.json", encoding="utf-8"))
        msgs.extend(data["messages"])
    threads = [{"slug": "saturday_squad", "title": "Saturday Squad",
                "thread_dir": SAMPLE, "msgs": ac.normalize_messages(msgs)}]
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = threading.Thread(target=chat_ui.run_server, args=(threads, port, out),
                              daemon=True)
    server.start()
    wait_for_server(port)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/t/saturday_squad/year/2017") as r:
        body = r.read().decode("utf-8")
    assert "2017 in review" in body
    with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/t/saturday_squad/year_in_review.html") as r:
        index = r.read().decode("utf-8")
    assert "year_2017.html" in index


def test_normalized_fields_flow_to_reader_json(tmp_path):
    import chat_ui
    raw = [
        _raw_msg(id="1", sender_name="Alice", gifs=[{"uri": "gifs/a.gif"}]),
        _raw_msg(id="2", sender_name="Bob", videos=[{"uri": "videos/v.mp4"}]),
        _raw_msg(id="3", sender_name="Bob", content="bye", is_taken_down=True),
    ]
    msgs = ac.normalize_messages(raw)
    thread = chat_ui.ThreadIndex("t", "T", tmp_path, msgs)
    j = thread.to_json(0)
    assert j["has_gif"] and j["gif_uris"] == ["gifs/a.gif"]
    j = thread.to_json(2)
    assert j["is_taken_down"]
