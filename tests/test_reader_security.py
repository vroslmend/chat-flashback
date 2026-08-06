"""Reader regression tests: escaping, media serving, and paging behaviour."""
import json
import socket
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta

import pytest

import analyze_chat as ac
import chat_ui
from server_utils import wait_for_server

BASE = datetime(2020, 6, 15, 23, 30)
EVIL_TITLE = "</script><script>alert(1)</script>'+alert(2)+'"


@pytest.fixture(scope="module")
def reader(tmp_path_factory):
    """A reader serving a thread whose title and attachments are hostile."""
    root = tmp_path_factory.mktemp("thread")
    (root / "photos").mkdir()
    (root / "files").mkdir()
    (root / "photos" / "p.jpg").write_bytes(b"\xff\xd8\xff" + b"A" * 5000)
    (root / "files" / "evil.html").write_text("<script>alert(3)</script>", encoding="utf-8")
    (root.parent / "secret.txt").write_text("secret", encoding="utf-8")

    raw = [{"sender_name": "Alice", "id": str(i), "content": f"message {i} shawarma",
            "timestamp_ms": int((BASE + timedelta(minutes=i)).timestamp() * 1000)}
           for i in range(50)]
    raw.append({"sender_name": "Bob", "id": "p", "content": None,
                "timestamp_ms": int(BASE.timestamp() * 1000),
                "photos": [{"uri": "photos/p.jpg"}]})
    threads = [{"slug": "t", "title": EVIL_TITLE, "thread_dir": root,
                "msgs": ac.normalize_messages(raw)}]

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    threading.Thread(target=chat_ui.run_server, args=(threads, port, root),
                     daemon=True).start()
    wait_for_server(port)
    return f"http://127.0.0.1:{port}"


def fetch(base, path, headers=None):
    req = urllib.request.Request(base + path, headers=headers or {})
    return urllib.request.urlopen(req)


def get_text(base, path):
    return fetch(base, path).read().decode("utf-8")


def get_json(base, path):
    return json.loads(get_text(base, path))


# --------------------------------------------------------------------------- #
# escaping                                                                     #
# --------------------------------------------------------------------------- #

def test_viewer_escapes_the_export_title(reader):
    page = get_text(reader, "/t/t/")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;/script&gt;" in page


def test_viewer_does_not_put_the_title_in_javascript(reader):
    """HTML escaping does not make a value safe inside a script block, so the
    title must not reach the JS at all."""
    page = get_text(reader, "/t/t/")
    script = page.split("<script>")[-1]
    assert "alert(2)" not in script
    assert "TITLE=" not in script


def test_landing_page_escapes_the_title(reader):
    assert "<script>alert(1)</script>" not in get_text(reader, "/")


# --------------------------------------------------------------------------- #
# media serving                                                                #
# --------------------------------------------------------------------------- #

def test_media_supports_range_requests(reader):
    r = fetch(reader, "/t/t/media/photos/p.jpg", {"Range": "bytes=10-19"})
    assert r.status == 206
    assert len(r.read()) == 10
    assert r.headers["Content-Range"] == "bytes 10-19/5003"
    assert r.headers["Accept-Ranges"] == "bytes"


def test_media_full_request_returns_whole_file(reader):
    r = fetch(reader, "/t/t/media/photos/p.jpg")
    assert r.status == 200
    assert len(r.read()) == 5003


def test_media_sets_nosniff(reader):
    r = fetch(reader, "/t/t/media/photos/p.jpg")
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_images_render_inline_but_html_downloads(reader):
    img = fetch(reader, "/t/t/media/photos/p.jpg")
    assert img.headers.get("Content-Disposition") is None
    page = fetch(reader, "/t/t/media/files/evil.html")
    assert "attachment" in page.headers.get("Content-Disposition", "")


@pytest.mark.parametrize("path", [
    "/t/t/media/../../secret.txt",
    "/t/t/media/..%2f..%2fsecret.txt",
    "/t/t/media/photos/../../../secret.txt",
])
def test_media_refuses_paths_outside_the_thread(reader, path):
    with pytest.raises(urllib.error.HTTPError) as exc:
        fetch(reader, path)
    assert exc.value.code == 404


# --------------------------------------------------------------------------- #
# paging and search                                                            #
# --------------------------------------------------------------------------- #

def test_search_reports_the_true_match_count(reader):
    data = get_json(reader, "/t/t/api/messages?q=shawarma&limit=10")
    assert data["total_matches"] == 50
    assert data["shown"] == 10
    assert data["truncated"] is True


def test_search_untruncated_is_not_flagged(reader):
    data = get_json(reader, "/t/t/api/messages?q=shawarma&limit=100")
    assert data["total_matches"] == 50
    assert data["truncated"] is False


def test_forward_cursor_returns_oldest_first(reader):
    data = get_json(reader, "/t/t/api/messages?after=0&limit=10")
    ts = [m["ts"] for m in data["messages"]]
    assert ts == sorted(ts)


def test_default_feed_returns_newest_first(reader):
    data = get_json(reader, "/t/t/api/messages?limit=10")
    ts = [m["ts"] for m in data["messages"]]
    assert ts == sorted(ts, reverse=True)


def test_viewer_honours_the_order_selector(reader):
    """The selector used to be inert: nothing read it, so it always paged back
    from the newest message."""
    assert "orderEl.value==='oldest'" in get_text(reader, "/t/t/")


# --------------------------------------------------------------------------- #
# on this day                                                                  #
# --------------------------------------------------------------------------- #

def test_on_this_day_uses_the_message_local_date(reader):
    day = get_json(reader, "/t/t/api/day?date=2020-06-15")
    assert day["total"] > 0
    assert day["years"] == [2020]


def test_on_this_day_is_empty_for_another_date(reader):
    assert get_json(reader, "/t/t/api/day?date=2020-03-02")["total"] == 0


def test_day_endpoint_matches_tz_shifted_dates():
    """day() must group by the same local date the reports use, not by a
    window rebuilt in the system timezone."""
    raw = [{"sender_name": "A", "content": "late night", "id": "1",
            "timestamp_ms": int(datetime(2020, 6, 15, 23, 30).timestamp() * 1000)}]
    for tz, expect_15 in (("+09:00", False), ("-09:00", True)):
        msgs = ac.normalize_messages(raw, tz=tz)
        index = chat_ui.ThreadIndex("t", "T", ".", msgs)
        local = msgs[0]["dt"]
        assert index.day(local.month, local.day)["total"] == 1
        assert (index.day(6, 15)["total"] == 1) is expect_15
