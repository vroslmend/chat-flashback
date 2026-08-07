"""SQLite store for a parsed thread.

The reader used to hold every message in memory and re-parse the export's JSON
on every start. On a 1.79M-message chat that is a slow startup and a large
resident process before anybody has read a word, and search was a substring scan
over the whole list.

This module writes the parsed messages once into a SQLite file beside the
report, keyed on the same thread fingerprint `--incremental` uses, and answers
the reader's questions with queries. A second start of an unchanged export
reuses the file and does no parsing at all.

Nothing here knows about HTML or HTTP. It takes normalized messages and returns
plain rows, so the reader and the tests see the same data.
"""
import json
import sqlite3
import threading

# Bump when the schema or the payload shape changes: an older file is thrown
# away and rebuilt rather than read with the wrong assumptions.
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE messages (
    idx        INTEGER PRIMARY KEY,
    msg_id     TEXT,
    ts_ms      INTEGER NOT NULL,
    sender     TEXT NOT NULL,
    content    TEXT,
    reply_to   TEXT,
    month      INTEGER NOT NULL,
    day        INTEGER NOT NULL,
    year       INTEGER NOT NULL,
    reacted    INTEGER NOT NULL,
    length     INTEGER NOT NULL,
    payload    TEXT NOT NULL
);
CREATE INDEX messages_ts ON messages(ts_ms);
CREATE INDEX messages_sender_ts ON messages(sender, ts_ms);
CREATE INDEX messages_monthday ON messages(month, day);
CREATE INDEX messages_msg_id ON messages(msg_id);
"""

# Columns the reader queries on. Everything else about a message — media uris,
# the flags, reactions — is only ever read whole, so it rides along as one JSON
# payload instead of six join tables nothing would ever query separately.
_PAYLOAD_SKIP = ("dt", "ts_ms", "sender", "content", "reply_to", "id")


def _payload(m):
    out = {k: v for k, v in m.items() if k not in _PAYLOAD_SKIP}
    # Tuples survive a JSON round trip as lists; the reader treats reactions as
    # pairs either way, so normalize here rather than at every read.
    out["reactions"] = [list(r) for r in m.get("reactions") or []]
    return out


class MessageStore:
    """One thread's messages on disk, and the queries the reader asks of them."""

    def __init__(self, path, fingerprint):
        self.path = path
        self.fingerprint = fingerprint
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.ready = self._matches_fingerprint()

    # -- building ---------------------------------------------------------- #

    def _matches_fingerprint(self):
        try:
            rows = dict(self._conn.execute("SELECT key, value FROM meta").fetchall())
        except sqlite3.Error:
            return False
        return (rows.get("fingerprint") == self.fingerprint
                and rows.get("schema") == str(SCHEMA_VERSION)
                and rows.get("complete") == "1")

    @property
    def title(self):
        """The thread's own title, stored so a reused file does not have to
        re-read the export just to learn what the chat is called.

        Asked of a file that does not exist yet, which is the first run.
        """
        try:
            row = self._one("SELECT value FROM meta WHERE key = 'title'")
        except sqlite3.Error:
            return None
        return row["value"] if row is not None else None

    def build(self, msgs, title=None):
        """Replace the file's contents with these messages.

        `complete` is written last, so a run interrupted halfway leaves a file
        that fails the fingerprint check and gets rebuilt rather than one that
        looks valid and is missing its tail.
        """
        with self._lock:
            cur = self._conn
            cur.executescript(
                "DROP TABLE IF EXISTS messages; DROP TABLE IF EXISTS meta;")
            cur.executescript(_SCHEMA)
            cur.executemany(
                "INSERT INTO messages (idx, msg_id, ts_ms, sender, content, reply_to,"
                " month, day, year, reacted, length, payload)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [(i, m.get("id"), m["ts_ms"], m["sender"], m["content"],
                  m.get("reply_to"), m["dt"].month, m["dt"].day, m["dt"].year,
                  1 if m.get("reactions") else 0, len(m["content"] or ""),
                  json.dumps(_payload(m), ensure_ascii=False))
                 for i, m in enumerate(msgs)])
            cur.executemany("INSERT INTO meta (key, value) VALUES (?,?)",
                            [("fingerprint", self.fingerprint),
                             ("schema", str(SCHEMA_VERSION)),
                             ("title", title or ""),
                             ("complete", "1")])
            cur.commit()
        self.ready = True

    # -- reading ----------------------------------------------------------- #

    def _all(self, sql, params=()):
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _one(self, sql, params=()):
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    @property
    def total(self):
        return self._one("SELECT COUNT(*) AS n FROM messages")["n"]

    def span(self):
        row = self._one("SELECT MIN(ts_ms) AS a, MAX(ts_ms) AS b FROM messages")
        return (row["a"], row["b"])

    def members(self):
        return [(r["sender"], r["n"]) for r in self._all(
            "SELECT sender, COUNT(*) AS n FROM messages"
            " GROUP BY sender ORDER BY n DESC, sender")]

    def has_replies(self):
        return self._one(
            "SELECT 1 AS x FROM messages WHERE msg_id IS NOT NULL LIMIT 1") is not None

    def row(self, idx):
        return self._one("SELECT * FROM messages WHERE idx = ?", (idx,))

    def rows(self, idxs):
        """Several messages by index, returned in the order asked for."""
        if not idxs:
            return []
        marks = ",".join("?" * len(idxs))
        found = {r["idx"]: r for r in
                 self._all(f"SELECT * FROM messages WHERE idx IN ({marks})", tuple(idxs))}
        return [found[i] for i in idxs if i in found]

    def by_msg_id(self, msg_id):
        return self._one("SELECT sender, content FROM messages WHERE msg_id = ?",
                         (msg_id,))

    def page(self, before=None, after=None, member=None, limit=400):
        """One screen of the feed, walking backwards unless asked to go forward.

        Returns the rows plus the cursor for the next call, mirroring what the
        in-memory reader returned so the handler and the front end are unchanged.
        """
        where, params = [], []
        if member:
            where.append("sender = ?")
            params.append(member)
        if after is not None:
            where.append("ts_ms > ?")
            params.append(after)
            order = "ASC"
        else:
            if before is not None:
                where.append("ts_ms < ?")
                params.append(before)
            order = "DESC"
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        rows = self._all(
            f"SELECT * FROM messages{clause} ORDER BY ts_ms {order}, idx {order}"
            f" LIMIT ?", tuple(params) + (limit + 1,))
        more = len(rows) > limit
        rows = rows[:limit]
        if order == "ASC":
            return rows, {"next_before": None,
                          "next_after": rows[-1]["ts_ms"] if (rows and more) else None}
        # Newest first, the order the feed renders in. The cursor is the oldest
        # message on the page, which is where the next one picks up.
        return rows, {"next_before": rows[-1]["ts_ms"] if (rows and more) else None,
                      "next_after": None}

    def search(self, q, member=None, limit=400):
        """Substring search, matching what the reader has always done.

        Deliberately not FTS5. Full-text indexes match whole tokens, so
        searching `tube` stops finding `youtube.com` -- on a 500k-row bench that
        is 0 hits where a substring scan finds 358,453. Nobody would report that
        as a bug; they would just conclude the search box is broken. The scan
        costs about 60 ms per 500k messages, which is not the reader's problem.
        """
        needle = q.lower()
        return self._search_scan(lambda c: needle in c.lower(), member, limit)

    def _search_scan(self, predicate, member, limit):
        """Streamed scan, for substring and regex alike.

        Rows are consumed as they arrive rather than collected, so searching a
        chat far larger than memory stays possible.
        """
        clause = " WHERE sender = ?" if member else ""
        params = (member,) if member else ()
        hits, total = [], 0
        with self._lock:
            cur = self._conn.execute(
                f"SELECT * FROM messages{clause} ORDER BY ts_ms", params)
            for row in cur:
                if predicate(row["content"] or ""):
                    total += 1
                    if len(hits) < limit:
                        hits.append(row)
        return hits, total

    def search_regex(self, pattern, member=None, limit=400):
        return self._search_scan(lambda c: pattern.search(c) is not None, member, limit)

    def day(self, month, day, limit=400):
        rows = self._all(
            "SELECT * FROM messages WHERE month = ? AND day = ? ORDER BY ts_ms LIMIT ?",
            (month, day, limit))
        total = self._one(
            "SELECT COUNT(*) AS n FROM messages WHERE month = ? AND day = ?",
            (month, day))["n"]
        years = [r["year"] for r in self._all(
            "SELECT DISTINCT year FROM messages WHERE month = ? AND day = ? ORDER BY year",
            (month, day))]
        return rows, total, years

    def random_row(self):
        """A message worth resurfacing: one that drew reactions, else a long one."""
        for clause in ("WHERE reacted = 1", "WHERE length > 40", ""):
            row = self._one(
                f"SELECT * FROM messages {clause} ORDER BY RANDOM() LIMIT 1")
            if row is not None:
                return row
        return None

    def close(self):
        with self._lock:
            self._conn.close()
