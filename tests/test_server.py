import json
import socket
import threading
from pathlib import Path

from chat_ui import run_server

from server_utils import wait_for_server

import analyze_chat as ac

SAMPLE = str(Path(__file__).resolve().parents[1] / "sample_data")


def _raw_messages():
    msgs = []
    for i in range(1, 5):
        data = json.load(open(f"{SAMPLE}/message_{i}.json", encoding="utf-8"))
        msgs.extend(data["messages"])
    return msgs


def _serve(tmp_path, build_index=True):
    """Start the reader on a free port against sample_data; return (port, get)."""
    import urllib.request

    threads = [{"slug": "saturday_squad", "title": "Saturday Squad",
                "thread_dir": SAMPLE,
                "msgs": ac.normalize_messages(_raw_messages())}]
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    threading.Thread(target=run_server, args=(threads, port, tmp_path),
                     kwargs={"build_index": build_index}, daemon=True).start()
    wait_for_server(port)

    def get(path):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
            return r.read().decode("utf-8")

    return port, get


def test_server_endpoints(tmp_path):
    from urllib.parse import quote

    msgs = _raw_messages()
    _, get = _serve(tmp_path)

    landing = get("/")
    assert "chat-flashback" in landing and "saturday_squad" in landing
    viewer = get("/t/saturday_squad/")
    assert "__SLUG__" not in viewer and "search" in viewer
    meta = json.loads(get("/t/saturday_squad/api/thread"))
    assert meta["total"] == 94
    assert meta["members"][0]["name"] in ("Alice", "Bob")
    data = json.loads(get("/t/saturday_squad/api/messages?limit=20"))
    assert len(data["messages"]) == 20
    assert data["next_before"] is not None
    data = json.loads(get("/t/saturday_squad/api/messages?after=0&limit=10"))
    assert data["messages"][0]["ts"] < data["messages"][-1]["ts"]
    data = json.loads(get("/t/saturday_squad/api/messages?q=" + quote("shawarma")))
    assert data["search"] and data["total_matches"] >= 4
    data = json.loads(get("/t/saturday_squad/api/messages?member=Alice&limit=50"))
    assert data["messages"] and all(m["sender"] == "Alice" for m in data["messages"])
    from collections import Counter
    from datetime import datetime as _dt

    dates = Counter(_dt.fromtimestamp(m["timestamp_ms"] / 1000).strftime("%Y-%m-%d")
                    for m in msgs if m.get("timestamp_ms"))
    query_date = dates.most_common(1)[0][0]
    day = json.loads(get(f"/t/saturday_squad/api/day?date={query_date}"))
    assert day["total"] >= 3
    assert "years" in day
    rnd = json.loads(get("/t/saturday_squad/api/random"))
    assert "sender" in rnd["message"]
    rex = json.loads(get("/t/saturday_squad/api/messages?q=%5Cb2018%5Cb&re=1&limit=5"))
    assert rex["search"] and rex["total_matches"] >= 1
    bad_re = json.loads(get("/t/saturday_squad/api/messages?q=%5B&re=1&limit=5"))
    assert bad_re["search"] is True


def test_word_endpoint_returns_a_profile(tmp_path):
    _, get = _serve(tmp_path)
    body = json.loads(get("/t/saturday_squad/api/word?q=shawarma"))
    assert body["word"] == "shawarma"
    assert body["uses"] == 5
    assert body["per_member"]
    assert body["examples"]["first"]["index"] is not None


def test_word_endpoint_404s_on_an_unknown_word(tmp_path):
    import urllib.error
    _, get = _serve(tmp_path)
    try:
        get("/t/saturday_squad/api/word?q=zzzznotaword")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
    else:
        raise AssertionError("expected 404")


def test_suggest_endpoint_completes_a_prefix(tmp_path):
    _, get = _serve(tmp_path)
    assert "bro" in json.loads(get("/t/saturday_squad/api/suggest?q=br"))["words"]


def test_word_endpoint_resolves_a_phrase(tmp_path):
    from urllib.parse import quote
    _, get = _serve(tmp_path)
    body = json.loads(get("/t/saturday_squad/api/word?q=" + quote("shawarma spot")))
    assert body["word"] == "shawarma spot"
    assert body["is_phrase"] is True
    assert body["uses"] == 1
    assert "shawarma" in body["examples"]["first"]["content"]


def test_viewer_ships_the_word_panel(tmp_path):
    _, get = _serve(tmp_path)
    page = get("/t/saturday_squad/")
    assert 'id="wordpanel"' in page
    assert 'id="wordq"' in page


def test_word_endpoint_reports_a_disabled_index(tmp_path):
    import urllib.error
    _, get = _serve(tmp_path, build_index=False)
    try:
        get("/t/saturday_squad/api/word?q=shawarma")
    except urllib.error.HTTPError as exc:
        assert exc.code == 503
    else:
        raise AssertionError("expected 503")


def test_word_hits_list_every_mention_oldest_first(tmp_path):
    _, get = _serve(tmp_path)
    body = json.loads(get("/t/saturday_squad/api/word/hits?q=shawarma"))
    assert body["total"] == 5
    assert len(body["messages"]) == 5
    assert body["next_offset"] is None
    stamps = [m["ts"] for m in body["messages"]]
    assert stamps == sorted(stamps)
    assert all("shawarma" in m["content"].lower() for m in body["messages"])


def test_word_hits_page_through_a_word(tmp_path):
    _, get = _serve(tmp_path)
    first = json.loads(get("/t/saturday_squad/api/word/hits?q=shawarma&limit=2"))
    assert len(first["messages"]) == 2 and first["next_offset"] == 2
    second = json.loads(
        get("/t/saturday_squad/api/word/hits?q=shawarma&limit=2&offset=2"))
    assert second["messages"][0]["ts"] > first["messages"][-1]["ts"]


def test_word_hits_404s_on_an_unknown_word(tmp_path):
    import urllib.error
    _, get = _serve(tmp_path)
    try:
        get("/t/saturday_squad/api/word/hits?q=zzzznotaword")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
    else:
        raise AssertionError("expected 404")


def test_the_viewer_groups_each_day_so_headers_cannot_stack(tmp_path):
    """Sticky siblings all stop at the same offset and pile up; each day needs
    to be its own containing block for one header to push out the last."""
    _, get = _serve(tmp_path)
    page = get("/t/saturday_squad/")
    assert "daygroup" in page
    assert ".daygroup{position:relative}" in page
