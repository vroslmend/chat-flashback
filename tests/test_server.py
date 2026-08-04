import json
import socket
import threading
from pathlib import Path

from chat_ui import run_server

from server_utils import wait_for_server

import analyze_chat as ac

SAMPLE = str(Path(__file__).resolve().parents[1] / "sample_data")


def test_server_endpoints(tmp_path):
    import json as _json

    msgs = []
    for i in range(1, 5):
        data = _json.load(open(f"{SAMPLE}/message_{i}.json", encoding="utf-8"))
        msgs.extend(data["messages"])
    norm = ac.normalize_messages(msgs)
    threads = [{"slug": "saturday_squad", "title": "Saturday Squad",
                "thread_dir": SAMPLE, "msgs": norm}]
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = threading.Thread(target=run_server, args=(threads, port, tmp_path), daemon=True)
    server.start()
    wait_for_server(port)

    import urllib.request
    from urllib.parse import quote

    def get(path):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
            return r.read().decode("utf-8")

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
    day = json.loads(get("/t/saturday_squad/api/day?date=2025-08-09"))
    assert day["total"] >= 3
    assert "years" in day
    rnd = json.loads(get("/t/saturday_squad/api/random"))
    assert "sender" in rnd["message"]
    rex = json.loads(get("/t/saturday_squad/api/messages?q=%5Cb2018%5Cb&re=1&limit=5"))
    assert rex["search"] and rex["total_matches"] >= 1
    bad_re = json.loads(get("/t/saturday_squad/api/messages?q=%5B&re=1&limit=5"))
    assert bad_re["search"] is True
