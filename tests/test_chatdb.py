"""The reader's SQLite store.

The store replaced an in-memory list, so the tests that matter are the ones
that would let it quietly disagree with that list: does paging reach every
message exactly once, does search still mean what it meant, and does a stale
file get rebuilt instead of read.
"""
import re
from datetime import timedelta

import analyze_chat as ac
import chat_ui
import chatdb
from test_correctness import BASE, mk


def sample(n=40):
    msgs = [mk(["Alice", "Bob", "Carol"][i % 3], BASE + timedelta(minutes=7 * i),
               f"message number {i} about youtube.com and cricket")
            for i in range(n)]
    msgs[3]["reactions"] = [("Bob", "love")]
    return msgs


def store_for(msgs, tmp_path, fingerprint="fp"):
    store = chatdb.MessageStore(tmp_path / "t.sqlite3", fingerprint)
    if not store.ready:
        store.build(msgs, title="Test Thread")
    return store


def test_a_reopened_store_needs_no_messages(tmp_path):
    msgs = sample()
    store_for(msgs, tmp_path).close()
    again = chatdb.MessageStore(tmp_path / "t.sqlite3", "fp")
    assert again.ready
    assert again.title == "Test Thread"
    assert again.total == len(msgs)
    # The point of the store: a working reader with nothing parsed.
    reader = chat_ui.ThreadIndex("t", again.title, ".", msgs=None,
                                 build_index=False, store=again)
    assert reader.page(limit=5)["messages"]


def test_a_changed_export_is_rebuilt_not_reused(tmp_path):
    store_for(sample(), tmp_path).close()
    stale = chatdb.MessageStore(tmp_path / "t.sqlite3", "a-different-fingerprint")
    assert not stale.ready


def test_an_interrupted_build_is_not_mistaken_for_a_finished_one(tmp_path):
    store = store_for(sample(), tmp_path)
    store._conn.execute("DELETE FROM meta WHERE key = 'complete'")
    store._conn.commit()
    store.close()
    assert not chatdb.MessageStore(tmp_path / "t.sqlite3", "fp").ready


def test_search_still_finds_a_substring_inside_a_word(tmp_path):
    """A full-text index would tokenize `youtube.com` and lose this; the reader
    has always matched substrings and must keep doing so."""
    msgs = sample()
    reader = chat_ui.ThreadIndex("t", "T", ".", msgs, build_index=False,
                                 store=store_for(msgs, tmp_path))
    expected = sum(1 for m in msgs if "tube" in m["content"].lower())
    assert expected > 0
    assert reader.page(q="tube")["total_matches"] == expected


def test_search_is_case_insensitive_and_counts_every_match(tmp_path):
    msgs = sample()
    reader = chat_ui.ThreadIndex("t", "T", ".", msgs, build_index=False,
                                 store=store_for(msgs, tmp_path))
    assert (reader.page(q="CRICKET")["total_matches"]
            == reader.page(q="cricket")["total_matches"] == len(msgs))


def test_regex_search_reaches_the_store(tmp_path):
    msgs = sample()
    reader = chat_ui.ThreadIndex("t", "T", ".", msgs, build_index=False,
                                 store=store_for(msgs, tmp_path))
    hits = reader.page(q=r"number \d+ about", limit=100, regex=True)
    assert hits["total_matches"] == len(msgs)


def test_paging_backwards_reaches_every_message_once(tmp_path):
    msgs = sample()
    reader = chat_ui.ThreadIndex("t", "T", ".", msgs, build_index=False,
                                 store=store_for(msgs, tmp_path))
    seen, cursor = [], None
    while True:
        page = reader.page(before=cursor, limit=6)
        seen += [m["ts"] for m in page["messages"]]
        # Newest first is the order the feed renders in.
        assert page["messages"] == sorted(page["messages"], key=lambda m: -m["ts"])
        cursor = page["next_before"]
        if cursor is None:
            break
    assert sorted(seen) == sorted(m["ts_ms"] for m in msgs)


def test_paging_forwards_reaches_every_message_once(tmp_path):
    msgs = sample()
    reader = chat_ui.ThreadIndex("t", "T", ".", msgs, build_index=False,
                                 store=store_for(msgs, tmp_path))
    seen, cursor = [], 0
    while True:
        page = reader.page(after=cursor, limit=6)
        seen += [m["ts"] for m in page["messages"]]
        if page["next_after"] is None:
            break
        cursor = page["next_after"]
    assert sorted(seen) == sorted(m["ts_ms"] for m in msgs)


def test_member_filter_pages_only_that_member(tmp_path):
    msgs = sample()
    reader = chat_ui.ThreadIndex("t", "T", ".", msgs, build_index=False,
                                 store=store_for(msgs, tmp_path))
    page = reader.page(member="Alice", limit=500)
    assert {m["sender"] for m in page["messages"]} == {"Alice"}
    assert len(page["messages"]) == sum(1 for m in msgs if m["sender"] == "Alice")


def test_an_unbuilt_store_without_messages_is_an_error(tmp_path):
    empty = chatdb.MessageStore(tmp_path / "fresh.sqlite3", "fp")
    try:
        chat_ui.ThreadIndex("t", "T", ".", msgs=None, build_index=False, store=empty)
    except ValueError:
        return
    raise AssertionError("expected an unbuilt store with no messages to be refused")
