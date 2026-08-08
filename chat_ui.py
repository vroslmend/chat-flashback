"""chat-flashback reader: a local web UI to browse a parsed chat.

Started with `python analyze_chat.py --input <thread> --serve`. Binds to
127.0.0.1 only and serves a Messenger-style reader: day-grouped feed, member
filters, full-text search, reply threading, media, and sentiment tinting.
"""

import json
import mimetypes
import random
import re
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import analyze_chat as ac
import chatdb
import chatpage
from wordindex import WordIndex

_PAGE_SIZE = 400

# A generated page, and nothing else. The analyzer only ever writes names built
# out of these characters, so anything else is not ours and cannot address a
# parent directory.
_PAGE_NAME = re.compile(r"^[A-Za-z0-9_]+\.html$")
_SIDE_FILES = {"summary.md": "text/markdown; charset=utf-8",
               "summary.json": "application/json; charset=utf-8"}


def _snippet(text, n=90):
    if not text:
        return "(no text)"
    text = text.replace("\n", " ")
    return text[:n] + ("..." if len(text) > n else "")


class ThreadIndex:
    """The reader's view of one thread, answered out of a `MessageStore`.

    `msgs` is only needed to fill an empty store and to build the word index.
    Once the store is populated and the word explorer is off, the reader runs
    without the parsed messages in memory at all.
    """

    def __init__(self, slug, title, thread_dir, msgs=None, build_index=True, store=None):
        self.slug = slug
        self.title = title
        self.thread_dir = Path(thread_dir)
        if store is None:
            # No file given: an in-memory database, so callers that just hand
            # over a list of messages keep working unchanged.
            store = chatdb.MessageStore(":memory:", "memory")
        self.store = store
        if not self.store.ready:
            if msgs is None:
                raise ValueError("an unbuilt store needs messages to fill it")
            self.store.build(msgs)
        self.total = self.store.total
        self.colors = {}
        for i, name in enumerate(sorted(n for n, _ in self.store.members())):
            self.colors[name] = ac.PALETTE[i % len(ac.PALETTE)]
        self._sent_cache = {}
        # The word explorer's inverted index. Built here rather than lazily so
        # the cost lands at startup, where it is announced, instead of on the
        # first search.
        self.words = WordIndex(msgs) if (build_index and msgs is not None) else None

    def to_json(self, idx):
        row = self.store.row(idx)
        return self._row_json(row) if row is not None else None

    def _row_json(self, row):
        p = json.loads(row["payload"])
        j = {
            "ts": row["ts_ms"], "sender": row["sender"],
            "color": self.colors.get(row["sender"], ac.PALETTE[0]),
            "content": row["content"], "mtype": p.get("mtype"),
            "reactions": [{"actor": a, "reaction": r} for a, r in p.get("reactions", [])],
            "has_photo": p.get("has_photo", False), "photo_uris": p.get("photo_uris", []),
            "has_sticker": p.get("has_sticker", False),
            "has_gif": p.get("has_gif", False), "gif_uris": p.get("gif_uris", []),
            "has_video": p.get("has_video", False), "video_uris": p.get("video_uris", []),
            "has_audio": p.get("has_audio", False), "audio_uris": p.get("audio_uris", []),
            "has_file": p.get("has_file", False), "file_uris": p.get("file_uris", []),
            "file_names": p.get("file_names", []),
            "is_taken_down": p.get("is_taken_down", False),
            "link": p.get("link"),
            "is_unsent": p.get("is_unsent", False), "reply_to": None, "sentiment": None,
        }
        if row["reply_to"] is not None:
            parent = self.store.by_msg_id(row["reply_to"])
            if parent is not None:
                j["reply_to"] = {"sender": parent["sender"],
                                 "snippet": _snippet(parent["content"])}
        if ac._VADER is not None and row["content"]:
            c = self._sent_cache.get(row["content"])
            if c is None:
                c = ac._VADER.polarity_scores(row["content"])["compound"]
                self._sent_cache[row["content"]] = c
                if len(self._sent_cache) > 50_000:
                    self._sent_cache.clear()
            j["sentiment"] = c
        return j

    def meta(self):
        start, end = self.store.span()
        members = [{"name": n, "count": c, "color": self.colors[n]}
                   for n, c in self.store.members()]
        return {
            "title": self.title,
            "slug": self.slug,
            "total": self.total,
            "start": start,
            "end": end,
            "members": members,
            "sentiment_available": ac._VADER is not None,
            "has_replies": self.store.has_replies(),
        }

    def page(self, before=None, after=None, member=None, q=None, limit=_PAGE_SIZE, regex=False):
        if q:
            return self._search(q, member, limit, regex)
        rows, cursor = self.store.page(before=before, after=after, member=member,
                                       limit=limit)
        return {"messages": [self._row_json(r) for r in rows],
                "next_before": cursor["next_before"],
                "next_after": cursor["next_after"],
                "search": False}

    def _search(self, q, member, limit, regex=False):
        pattern = None
        if regex:
            try:
                pattern = re.compile(q, re.IGNORECASE)
            except re.error:
                pattern = None
        if pattern is not None:
            rows, total = self.store.search_regex(pattern, member, limit)
        else:
            rows, total = self.store.search(q, member, limit)
        return {"messages": [self._row_json(r) for r in rows],
                "next_before": None, "next_after": None,
                "search": True, "total_matches": total,
                "shown": len(rows), "truncated": total > len(rows)}

    def day(self, month, day, limit=_PAGE_SIZE):
        rows, total, years = self.store.day(month, day, limit)
        return {"messages": [self._row_json(r) for r in rows],
                "total": total, "years": years}

    def random_memory(self):
        row = self.store.random_row()
        return {"message": self._row_json(row) if row is not None else None}

    def resolve_media(self, rel):
        # Same resolver the analyzer uses, so the reader and --check agree on
        # which attachments exist and neither can drift out of the thread dir.
        return ac.resolve_media_path(self.thread_dir, rel)


def _day(ts_ms):
    """The date of a stored timestamp, which the store keeps as epoch millis."""
    return datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")


def _thread_pages(output_dir, slug):
    """Which generated pages exist for this thread, right now.

    Read off disk rather than guessed, so the reader's navigation offers exactly
    what a previous run wrote -- including nothing, when the export has only ever
    been served and never analyzed.
    """
    where = Path(output_dir) / slug
    return {href for href, _ in chatpage.NAV if (where / href).is_file()}


def _read_int(params, key):
    v = params.get(key)
    if not v:
        return None
    try:
        return int(float(v[0]))
    except (TypeError, ValueError):
        return None


def make_handler(threads, output_dir):

    class Handler(BaseHTTPRequestHandler):
        server_version = "chat-flashback/1.0"

        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="text/html; charset=utf-8"):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _send_media(self, path, ctype):
            """Stream a media file, honouring a single Range request.

            Streamed in chunks rather than read whole so a large video does not
            have to fit in memory, and Range is answered so players can seek.
            Anything that is not an image, video or audio is forced to download
            instead of rendering: an export can contain .html or .svg
            attachments, which would otherwise run as script on this origin.
            """
            size = path.stat().st_size
            start, end = 0, size - 1
            partial = False
            rng = self.headers.get("Range", "")
            m = re.match(r"^bytes=(\d*)-(\d*)$", rng.strip()) if rng else None
            if m and size:
                lo, hi = m.group(1), m.group(2)
                if lo:
                    start = min(int(lo), size - 1)
                    end = min(int(hi), size - 1) if hi else size - 1
                elif hi:                      # suffix range: last N bytes
                    start = max(0, size - int(hi))
                if start <= end:
                    partial = True
                else:
                    start, end = 0, size - 1
            length = end - start + 1
            inline = ctype.split("/")[0] in ("image", "video", "audio")
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if not inline:
                self.send_header("Content-Disposition",
                                 f'attachment; filename="{path.name}"')
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

        def do_GET(self):
            parts = urlsplit(self.path)
            path = unquote(parts.path)
            query = parse_qs(parts.query)
            if path == "/":
                return self._landing()
            if path == "/favicon.ico":
                return self._send(404, "")
            rest = path.split("/")[1:]
            if rest and rest[0] == "t" and len(rest) >= 2:
                slug = rest[1]
                thread = threads.get(slug)
                if thread is None:
                    return self._send(404, "not found")
                sub = [s for s in rest[2:] if s]
                if not sub:
                    # Which pages exist is read off disk on every request, so
                    # the reader's nav picks up a report written while it runs.
                    return self._send(200, viewer_html(slug, thread.title,
                                                       _thread_pages(output_dir, slug)))
                if sub[0] == "api":
                    if len(sub) >= 2 and sub[1] == "messages":
                        return self._messages(thread, query)
                    if len(sub) >= 2 and sub[1] == "day":
                        return self._day(thread, query)
                    if len(sub) >= 2 and sub[1] == "random":
                        return self._json(thread.random_memory())
                    if len(sub) >= 3 and sub[1] == "word" and sub[2] == "hits":
                        return self._word_hits(thread, query)
                    if len(sub) >= 2 and sub[1] == "word":
                        return self._word(thread, query)
                    if len(sub) >= 2 and sub[1] == "suggest":
                        return self._word_suggest(thread, query)
                    return self._json(thread.meta())
                # Every page the analyzer writes, not a hand-kept list of three.
                # The old routing meant most of the report's own navigation --
                # the narrative pages, the member pages, the year pages the
                # index links to -- answered 404 while being served.
                if len(sub) == 1 and _PAGE_NAME.match(sub[0]):
                    return self._page(slug, sub[0])
                if len(sub) == 1 and sub[0] in _SIDE_FILES:
                    side = output_dir / slug / sub[0]
                    if side.is_file():
                        return self._send(200, side.read_bytes(), _SIDE_FILES[sub[0]])
                    return self._send(404, "not written")
                # Kept so links from an older report still resolve.
                if sub[0] == "year" and len(sub) >= 2:
                    return self._page(slug, f"year_{sub[1]}.html")
                if sub[0] == "media":
                    rel = "/".join(sub[1:])
                    media = thread.resolve_media(rel)
                    if media is None:
                        return self._send(404, "not found")
                    ctype = mimetypes.guess_type(media.name)[0] or "application/octet-stream"
                    return self._send_media(media, ctype)
            return self._send(404, "not found")

        def _page(self, slug, name):
            """A generated page, with the reader announced to it.

            The file on disk has no reader link and no "open in reader" links:
            opened from the filesystem there is nothing to link to. Swapping the
            hook in on the way out lights both up without keeping two versions
            of the same page.
            """
            page = (output_dir / slug / name).resolve()
            root = (output_dir / slug).resolve()
            if page.parent != root or not page.is_file():
                return self._send(404, "not found")
            body = page.read_text(encoding="utf-8", errors="replace")
            body = body.replace(chatpage.READER_HOOK, chatpage.reader_hook_script("./"))
            return self._send(200, body, "text/html; charset=utf-8")

        def _landing(self):
            cards = []
            for t in threads.values():
                start, end = t.store.span()
                span = f"{_day(start)} to {_day(end)}" if start and end else ""
                pages = _thread_pages(output_dir, t.slug)
                links = "".join(
                    f'<a href="/t/{t.slug}/{href}">{html(label)}</a>'
                    for href, label in chatpage.NAV if href in pages)
                cards.append(
                    f'<li><a href="/t/{t.slug}/"><b>{html(t.title)}</b>'
                    f'<span>{t.total:,} messages</span>'
                    f'<span>{html(span)}</span></a>'
                    + (f'<p class="muted">{links}</p>' if links else
                       '<p class="muted">No report written yet &mdash; run without '
                       '--serve first.</p>')
                    + "</li>")
            main = ("<h1>chat-flashback</h1>"
                    '<p class="lede">Open a chat to read it, or jump straight to its '
                    "report. This server listens on your machine only.</p>"
                    f'<ul class="grid-links">{"".join(cards)}</ul>')
            # No thread context at this level, so the nav is only the brand.
            return self._send(200, chatpage.document(
                doc_title="chat-flashback", brand="chat-flashback", written=set(),
                current=None, main=main, brand_href="/"))

        def _messages(self, thread, query):
            before = _read_int(query, "before")
            after = _read_int(query, "after")
            member = (query.get("member") or [None])[0]
            q = (query.get("q") or [None])[0]
            regex = (query.get("re") or [""])[0] == "1"
            limit = _read_int(query, "limit") or _PAGE_SIZE
            return self._json(thread.page(before=before, after=after,
                                          member=member or None, q=q or None,
                                          limit=limit, regex=regex))

        def _day(self, thread, query):
            date = (query.get("date") or [""])[0]
            try:
                month, day = int(date[5:7]), int(date[8:10])
            except (ValueError, IndexError):
                return self._send(400, "date must be YYYY-MM-DD")
            return self._json(thread.day(month, day))

        def _word(self, thread, query):
            if thread.words is None:
                return self._json({"error": "index disabled"}, 503)
            q = (query.get("q") or [""])[0]
            fold = (query.get("variants") or ["0"])[0] == "1"
            profile = thread.words.profile(q, fold_variants=fold)
            if profile is None:
                return self._json({"error": "not found", "word": q}, 404)
            return self._json(profile)

        def _word_hits(self, thread, query):
            """Every message holding the word, oldest first, a page at a time."""
            if thread.words is None:
                return self._json({"error": "index disabled"}, 503)
            q = (query.get("q") or [""])[0]
            fold = (query.get("variants") or ["0"])[0] == "1"
            idxs = thread.words.matches(q, fold_variants=fold)
            if not idxs:
                return self._json({"error": "not found", "word": q}, 404)
            offset = max(0, _read_int(query, "offset") or 0)
            limit = min(_read_int(query, "limit") or _PAGE_SIZE, _PAGE_SIZE)
            window = idxs[offset:offset + limit]
            nxt = offset + limit
            return self._json({"word": q, "total": len(idxs), "offset": offset,
                               "next_offset": nxt if nxt < len(idxs) else None,
                               "messages": [thread.to_json(i) for i in window]})

        def _word_suggest(self, thread, query):
            if thread.words is None:
                return self._json({"words": []})
            q = (query.get("q") or [""])[0]
            return self._json({"words": thread.words.suggest(q)})

    return Handler


def html(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def viewer_html(slug, title, written=frozenset()):
    """The reader, wearing the same frame as every generated page.

    The title comes from the export, so it is untrusted text and is escaped.
    The slug is safe by construction (`ac._slug` keeps only alphanumerics and
    underscores).
    """
    return (_VIEWER
            .replace("__TOKENS__", chatpage.TOKENS)
            .replace("__BASE_CSS__", chatpage.BASE_CSS)
            .replace("__READER_CSS__", _READER_CSS)
            .replace("__THEME_BOOT__", chatpage.THEME_BOOT)
            .replace("__THEME_JS__", chatpage.THEME_JS)
            .replace("__TOPBAR__", chatpage.topbar(f"{title} flashback", written,
                                                   "reader", served=True))
            .replace("__SLUG__", slug)
            .replace("__TITLE__", html(title)))


_READER_CSS = """
/* The controls sit under the shared bar and are sticky in their own right, so
   the two together are the only thing pinned to the top. */
.ctlwrap{position:sticky;top:var(--topbar-h);z-index:25;background:var(--bg2);
border-bottom:1px solid var(--border)}
/* Same width as the feed below it, so the search box lines up with the
   messages instead of floating off to one side. */
.ctl{display:flex;flex-wrap:wrap;gap:var(--s2);align-items:center;
padding:var(--s2) var(--s4);max-width:820px;margin:0 auto}
.ctl input,.ctl select,.ctl button{font:inherit;font-size:var(--t-sm);
border:1px solid var(--border);border-radius:var(--radius-sm);
background:var(--bg);color:var(--fg);padding:6px 9px}
/* A label that breaks in half is worse than a row that wraps. */
.ctl button{cursor:pointer;white-space:nowrap}
.ctl button:hover{border-color:var(--accent)}
.ctl button.on{background:var(--accent);color:var(--accent-fg);border-color:var(--accent)}
#q{flex:1 1 auto;min-width:120px}
#re{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.more{display:flex;gap:var(--s2);align-items:center}
#moretoggle{display:none}
.status{max-width:820px;margin:0 auto;padding:0 var(--s4) var(--s2);
font-size:var(--t-sm);color:var(--fg2);min-height:20px}
@media(max-width:760px){
  #moretoggle{display:block}
  .more{display:none;width:100%;flex-wrap:wrap}
  .more.open{display:flex}
}
main{max-width:820px;margin:0 auto;padding:0 var(--s4) 140px}
/* Each day is its own containing block. Sticky siblings all stop at the same
   offset and pile up on top of each other, so the headers have to be scoped to
   the messages they belong to for one to push the last one out. */
.daygroup{position:relative}
/* Offset measured from the bars themselves. It used to be a hardcoded 52px,
   which put every day header underneath the toolbar the moment it wrapped. */
.day{position:sticky;top:var(--sticky-h,92px);z-index:10;display:flex;align-items:center;
gap:var(--s3);margin:var(--s5) 0 var(--s2);padding:var(--s1) 0;font-size:var(--t-sm);
color:var(--fg2);background:var(--bg)}
.day::before,.day::after{content:"";flex:1;height:1px;background:var(--border)}
.msg{display:flex;gap:var(--s3);padding:3px var(--s3);border-radius:var(--radius);
border-left:3px solid transparent}
.msg:hover{background:var(--bg2)}
/* A quiet edge rather than a full-row wash, which read as a hover state. */
.msg.pos{border-left-color:var(--pos)}
.msg.neg{border-left-color:var(--neg)}
.dot{width:30px;height:30px;border-radius:50%;flex:0 0 30px;display:flex;
align-items:center;justify-content:center;font-weight:700;font-size:var(--t-sm);color:#111}
/* A run from one sender shows the avatar and name once; the rest keep the
   column and reveal their time on hover. */
.msg.cont .gutter{flex:0 0 30px;font-size:var(--t-xs);color:var(--fg2);
text-align:right;opacity:0;padding-top:3px}
.msg.cont:hover .gutter{opacity:1}
.msg.cont{margin-top:0}
.msg.head{margin-top:var(--s2)}
.body{flex:1;min-width:0}
.sender{font-size:var(--t-sm);font-weight:700;margin-bottom:1px}
.text{word-wrap:break-word;white-space:pre-wrap}
.time{font-size:var(--t-xs);color:var(--fg2);margin-left:6px;font-weight:400}
.quote{border-left:3px solid var(--border);padding:2px 8px;margin:2px 0 4px;
font-size:var(--t-sm);color:var(--fg2);background:var(--bg2);border-radius:4px}
.badge{display:inline-block;font-size:var(--t-xs);padding:1px 6px;border-radius:8px;
background:var(--bg3);color:var(--fg2);margin-left:6px}
.unsent{text-decoration:line-through;opacity:.6}
.media{margin:4px 0}
.media img{max-width:220px;max-height:220px;border-radius:var(--radius);
border:1px solid var(--border);display:block}
/* An export usually ships without most of its media. The file is gone, not the
   message, so the bubble says so instead of rendering as a blank gap. */
.media.gone{font-size:var(--t-sm);color:var(--fg2);border:1px dashed var(--border);
border-radius:var(--radius-sm);padding:3px 8px;display:inline-block}
.nocontent{font-size:var(--t-sm);color:var(--fg2);font-style:italic}
.media video{max-width:280px;max-height:220px;border-radius:var(--radius);
border:1px solid var(--border);display:block}
.media audio{width:280px;max-width:100%}
.media a{word-break:break-all;font-size:var(--t-sm)}
.reactions{display:flex;gap:var(--s1);flex-wrap:wrap;margin-top:3px}
.ra{background:var(--bg2);border:1px solid var(--border);border-radius:12px;
padding:0 8px;font-size:var(--t-sm)}
.link a{word-break:break-all;font-size:var(--t-sm)}
.call{font-size:var(--t-sm);color:var(--fg2)}
.loader{text-align:center;color:var(--fg2);padding:var(--s5);font-size:var(--t-sm)}
.empty{text-align:center;color:var(--fg2);padding:var(--s7) var(--s4)}
#wordpanel{display:none;max-width:820px;margin:0 auto;padding:var(--s4) var(--s4) 0}
#wordpanel.open{display:block}
#wordq{width:100%;padding:8px 10px;border:1px solid var(--border);
border-radius:var(--radius-sm);background:var(--bg2);color:var(--fg);font:inherit}
#wordpanel label{display:inline-block;margin:var(--s2) 0;font-size:var(--t-sm);color:var(--fg2)}
#wordout h3{font-size:var(--t-md);margin:var(--s3) 0 2px}
#wordout p{font-size:var(--t-sm);margin:var(--s1) 0}
#wordout table{border-collapse:collapse;font-size:var(--t-sm);margin:var(--s2) 0;width:auto}
#wordout td{padding:2px 14px 2px 0;white-space:nowrap;border:0}
#wordout td:nth-child(2),#wordout td:nth-child(3){color:var(--fg2)}
.wordex{border-left:3px solid var(--border);background:var(--bg2);border-radius:4px;
padding:5px 9px;margin:6px 0;font-size:var(--t-sm)}
.wordex b{display:block;font-size:var(--t-xs);color:var(--fg2);font-weight:600;margin-bottom:2px}
.wordex button{margin-top:5px;padding:2px 9px;font:inherit;font-size:var(--t-xs);
border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--bg);
color:var(--fg);cursor:pointer}
.wordall{margin:var(--s1) 0 var(--s2);padding:5px 12px;font:inherit;font-size:var(--t-sm);
border:1px solid var(--accent);border-radius:var(--radius-sm);background:var(--accent);
color:var(--accent-fg);cursor:pointer}
body[data-wordmode] .msg{cursor:pointer}
"""

_VIEWER = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ | chat-flashback</title>
__THEME_BOOT__
<style>__TOKENS____BASE_CSS____READER_CSS__</style>
</head><body>
__TOPBAR__
<div class="ctlwrap">
<div class="ctl">
<input id="q" placeholder="Search messages..." aria-label="Search messages"/>
<button id="re" type="button" title="Match as a regular expression" aria-label="Regular expression search">.*</button>
<select id="member" aria-label="Filter by member"><option value="">Everyone</option></select>
<select id="order" aria-label="Sort order">
<option value="newest">Newest first</option>
<option value="oldest">Oldest first</option>
</select>
<button id="moretoggle" type="button" aria-expanded="false">More</button>
<div class="more" id="more">
<input id="jump" type="date" aria-label="Jump to a date"/>
<button id="oday" type="button" title="Show this day in every year">On this day</button>
<button id="wordbtn" type="button" title="Look up a word">Words</button>
<button id="surprise" type="button" title="A random message">Surprise me</button>
</div>
</div>
<div class="status" id="count"></div>
</div>
<section id="wordpanel">
<input id="wordq" placeholder="Look up a word or a phrase" autocomplete="off" list="wordsug"/>
<datalist id="wordsug"></datalist>
<label><input type="checkbox" id="wordfold"/> count spellings together</label>
<div id="wordout"></div>
</section>
<main id="feed"></main>
<div class="loader" id="loader">Loading...</div>
<script>__THEME_JS__</script>
<script>
var SLUG='__SLUG__';
var feed=document.getElementById('feed'), loader=document.getElementById('loader');
var qEl=document.getElementById('q'), memberEl=document.getElementById('member');
var orderEl=document.getElementById('order'), jumpEl=document.getElementById('jump');
var countEl=document.getElementById('count');
var reBtn=document.getElementById('re');
var odayBtn=document.getElementById('oday'), surpriseBtn=document.getElementById('surprise');
var wordBtn=document.getElementById('wordbtn'), wordPanel=document.getElementById('wordpanel');
var wq=document.getElementById('wordq'), wout=document.getElementById('wordout');
var wfold=document.getElementById('wordfold'), wsug=document.getElementById('wordsug'), wtimer=null;
var moreBtn=document.getElementById('moretoggle'), moreBox=document.getElementById('more');
var state={before:null,after:null,q:'',member:'',loading:false,done:false,searching:false,
mode:'feed',regex:false,wordq:'',wordfold:false,offset:0,day:'',anchor:null};
var lastDay='', lastSender='', lastTs=0;
var curGroup=null;
var THREAD=null;
var applying=false;
/* The sticky day headers start where the bars end. Measured rather than
   assumed: the toolbar is a different height on a phone than on a desktop. */
function measure(){var h=0;
document.querySelectorAll('.cf-top,.ctlwrap').forEach(function(e){h+=e.offsetHeight;});
document.documentElement.style.setProperty('--sticky-h',h+'px');}
if(window.ResizeObserver){var ro=new ResizeObserver(measure);
document.querySelectorAll('.cf-top,.ctlwrap').forEach(function(e){ro.observe(e);});}
window.addEventListener('resize',measure);measure();
function clearFeed(){feed.innerHTML='';lastDay='';lastSender='';lastTs=0;curGroup=null;
document.body.removeAttribute('data-wordmode');}
function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function clock(ts){var d=new Date(ts);return d.toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'});}
function dayKey(ts){var d=new Date(ts);return d.getFullYear()+'-'+(d.getMonth()+1)+'-'+d.getDate();}
function dayLabel(ts){var d=new Date(ts),t=new Date();var key=dayKey(ts);
var today=dayKey(t.getTime());var yest=dayKey(t.getTime()-864e5);
if(key===today)return 'Today';if(key===yest)return 'Yesterday';return d.toLocaleDateString(undefined,{weekday:'long',month:'long',day:'numeric',year:'numeric'});}
function yearsAgo(ts){var d=new Date(ts),t=new Date();if(d.getMonth()!==t.getMonth()||d.getDate()!==t.getDate())return null;var y=t.getFullYear()-d.getFullYear();return y>0?y:null;}
function sentClass(s){if(s==null)return '';if(s>0.15)return ' pos';if(s<-0.15)return ' neg';return '';}
/* Most exports ship without most of their media, and a broken <img> renders as
   nothing at all: the message looks blank while its reactions sit underneath. */
function mediaGone(el){var box=el.parentNode;box.className='media gone';
box.textContent=(el.getAttribute('data-kind')||'file')+' missing from the export';}
function mediaHTML(m){
var out='';
if(m.photo_uris&&m.photo_uris.length){m.photo_uris.slice(0,3).forEach(function(u){out+='<div class="media"><img loading="lazy" data-kind="photo" onerror="mediaGone(this)" src="/t/'+SLUG+'/media/'+esc(u)+'" alt=""/></div>';});}
if(m.has_gif&&m.gif_uris.length){m.gif_uris.slice(0,2).forEach(function(u){out+='<div class="media"><img loading="lazy" data-kind="gif" onerror="mediaGone(this)" src="/t/'+SLUG+'/media/'+esc(u)+'" alt=""/></div>';});}
if(m.has_video&&m.video_uris.length){m.video_uris.slice(0,2).forEach(function(u){out+='<div class="media"><video controls preload="metadata" data-kind="video" onerror="mediaGone(this)" src="/t/'+SLUG+'/media/'+esc(u)+'"></video></div>';});}
if(m.has_audio&&m.audio_uris.length){m.audio_uris.slice(0,2).forEach(function(u){out+='<div class="media"><audio controls preload="metadata" data-kind="voice message" onerror="mediaGone(this)" src="/t/'+SLUG+'/media/'+esc(u)+'"></audio></div>';});}
if(m.has_file&&m.file_uris.length){m.file_uris.slice(0,3).forEach(function(u){out+='<div class="media"><a href="/t/'+SLUG+'/media/'+esc(u)+'" target="_blank">'+esc(u.split('/').pop())+'</a></div>';});}
if(m.has_file&&!m.file_uris.length){m.file_names.slice(0,3).forEach(function(n){out+='<div class="media">[file: '+esc(n)+']</div>';});}
if(m.has_sticker)out+='<div class="media sticker">[sticker]</div>';
if(m.link)out+='<div class="link"><a href="'+esc(m.link)+'" target="_blank" rel="noopener noreferrer">'+esc(m.link)+'</a></div>';
if(m.mtype==='Call')out+='<div class="call">[call]</div>';
if(m.is_unsent)out+='<span class="unsent">(unsent)</span>';
if(m.is_taken_down)out+='<span class="unsent">(removed)</span>';
return out;}
/* `cont` marks a message that continues the run above it: same sender, same
   day, minutes apart. The date lives in the day header, so the message only
   ever shows a clock time. */
function msgHTML(m,cont){
var ya=yearsAgo(m.ts);
var quote=m.reply_to?'<div class="quote">'+esc(m.reply_to.sender)+': '+esc(m.reply_to.snippet)+'</div>':'';
var reactions=m.reactions.map(function(r){return '<span class="ra">'+esc(r.reaction)+' '+esc(r.actor)+'</span>';}).join('');
var badge=ya?'<span class="badge">'+ya+'y ago</span>':'';
var mhtml=mediaHTML(m);
/* Roughly one message in a hundred arrives with no text, no media and no type.
   Rendering it as an empty bubble looks like the reader dropped it. */
var body=m.content?'<div class="text">'+esc(m.content)+'</div>':
(mhtml?'':'<div class="nocontent">(this message came through empty)</div>');
var inner=quote+body+mhtml+(reactions?'<div class="reactions">'+reactions+'</div>':'');
var head=cont?'<div class="gutter">'+clock(m.ts)+'</div>':
'<div class="dot" style="background:'+esc(m.color)+'">'+esc(m.sender[0]).toUpperCase()+'</div>';
var name=cont?'':'<div class="sender">'+esc(m.sender)+'<span class="time">'+clock(m.ts)+
'</span>'+badge+'</div>';
return '<div class="msg '+(cont?'cont':'head')+sentClass(m.sentiment)+'" data-ts="'+m.ts+'">'+
head+'<div class="body">'+name+inner+'</div></div>';}
function dayGroup(label){var sec=document.createElement('section');sec.className='daygroup';
var d=document.createElement('div');d.className='day';d.textContent=label;
sec.appendChild(d);feed.appendChild(sec);return sec;}
function appendMsgs(msgs){msgs.forEach(function(m){var k=dayKey(m.ts);
if(k!==lastDay||!curGroup){curGroup=dayGroup(dayLabel(m.ts));lastDay=k;lastSender='';}
/* Five minutes: long enough to hold a burst together, short enough that a
   reply hours later still gets its own header. */
var cont=(m.sender===lastSender)&&Math.abs(m.ts-lastTs)<3e5;
curGroup.insertAdjacentHTML('beforeend',msgHTML(m,cont));
lastSender=m.sender;lastTs=m.ts;});}
function apiURL(){var u='/t/'+SLUG+'/api/messages?limit=400';
if(state.q){u+='&q='+encodeURIComponent(state.q);}
if(state.regex)u+='&re=1';
if(state.member)u+='&member='+encodeURIComponent(state.member);
if(state.before!=null)u+='&before='+state.before;
if(state.after!=null)u+='&after='+state.after;
return u;}
/* Nothing loads before the thread meta arrives. The lazy-load observer used to
   fire first and race the initial page, appending the same history twice --
   which is how a 94-message chat reported "188 of 94 messages loaded". */
function loadMore(){if(!THREAD||state.loading||state.done)return;
if(state.mode==='word')return loadWordPage();
state.loading=true;loader.textContent='Loading...';
fetch(apiURL()).then(function(r){return r.json();}).then(function(data){
state.loading=false;
if(state.searching){clearFeed();
countEl.textContent=data.truncated?('showing '+data.shown+' of '+data.total_matches+' match(es)'):(data.total_matches+' match(es)');
appendMsgs(data.messages);if(data.total_matches<1){feed.innerHTML='<div class="empty">No matches.</div>';}state.done=true;loader.style.display='none';return;}
appendMsgs(data.messages);
if(data.next_before!=null){state.before=data.next_before;}else if(data.next_after!=null){state.after=data.next_after;}else{state.done=true;}
countEl.textContent=feed.querySelectorAll('.msg').length+' of '+THREAD.total.toLocaleString()+' messages loaded';
if(state.done){loader.textContent='End of history.';}}).catch(function(){state.loading=false;loader.textContent='Error loading.';});}
function reset(mode){state.loading=false;state.done=false;state.before=null;state.after=null;
state.mode='feed';state.anchor=null;state.day='';clearFeed();loader.style.display='';
if(mode==='search'){state.searching=true;}else{state.searching=false;
/* Oldest-first walks forward from the start of history; leaving both cursors
   null always paginates backwards from the newest message. */
if(orderEl.value==='oldest')state.after=0;}
loadMore();}
function onThisDay(dateStr){state.mode='day';state.day=dateStr;state.done=false;state.loading=false;
clearFeed();loader.style.display='';
fetch('/t/'+SLUG+'/api/day?date='+encodeURIComponent(dateStr)).then(function(r){return r.json();}).then(function(data){
loader.style.display='none';state.done=true;
var byYear={};data.messages.forEach(function(m){var y=new Date(m.ts).getFullYear();(byYear[y]=byYear[y]||[]).push(m);});
var years=Object.keys(byYear).sort();
countEl.textContent=data.total+' message(s) on '+dateStr.slice(5)+' across '+(years.length||0)+' year(s)';
if(years.length===0){feed.innerHTML='<div class="empty">Nothing happened on this day.</div>';return;}
years.forEach(function(y){var sec=dayGroup(y+' ('+byYear[y].length+' message'+(byYear[y].length===1?'':'s')+')');
var ls='',lt=0;
byYear[y].forEach(function(m){var cont=(m.sender===ls)&&Math.abs(m.ts-lt)<3e5;
sec.insertAdjacentHTML('beforeend',msgHTML(m,cont));ls=m.sender;lt=m.ts;});});
}).catch(function(){loader.style.display='none';state.done=true;feed.innerHTML='<div class="empty">Error loading.</div>';});}
function randomMemory(){state.mode='random';state.done=true;state.loading=false;clearFeed();loader.style.display='none';
fetch('/t/'+SLUG+'/api/random').then(function(r){return r.json();}).then(function(d){
countEl.textContent='A random message from the chat';
feed.insertAdjacentHTML('beforeend',msgHTML(d.message,false));
feed.insertAdjacentHTML('beforeend','<div style="text-align:center;margin-top:16px"><button class="cf-btn" onclick="randomMemory()">Another</button></div>');
}).catch(function(){feed.innerHTML='<div class="empty">Error loading.</div>';});}
/* Word mode: the feed becomes every message holding the word, oldest first,
   and clicking one leaves the list to read the conversation around it. */
function showWordMentions(word,folded){state.mode='word';state.wordq=word;
state.wordfold=folded;state.offset=0;state.done=false;state.loading=false;
state.searching=false;clearFeed();document.body.setAttribute('data-wordmode','1');
loader.style.display='';countEl.textContent='Loading mentions of "'+word+'"...';
window.scrollTo(0,0);writeHash(true);loadWordPage();}
function loadWordPage(){state.loading=true;loader.textContent='Loading...';
fetch('/t/'+SLUG+'/api/word/hits?q='+encodeURIComponent(state.wordq)+
'&variants='+(state.wordfold?'1':'0')+'&offset='+state.offset)
.then(function(r){return r.status===200?r.json():null;}).then(function(d){
state.loading=false;
if(!d){state.done=true;loader.style.display='none';
feed.innerHTML='<div class="empty">Never said in this chat.</div>';return;}
appendMsgs(d.messages);
countEl.textContent=Math.min(d.offset+d.messages.length,d.total).toLocaleString()+
' of '+d.total.toLocaleString()+' mentions of "'+d.word+'" \\u2014 click one to open it in the chat';
if(d.next_offset!=null){state.offset=d.next_offset;}
else{state.done=true;loader.textContent='End of mentions.';}
}).catch(function(){state.loading=false;loader.textContent='Error loading.';});}
/* One handler on the feed rather than one per message: in word mode a click
   anywhere on a message is a request for its context. */
feed.addEventListener('click',function(ev){if(state.mode!=='word')return;
var el=ev.target.closest?ev.target.closest('.msg'):null;
if(el&&el.dataset.ts)jumpToTs(parseInt(el.dataset.ts,10));});
/* Everything below builds nodes and sets textContent. The profile carries raw
   export text, and unlike the feed it does not go through msgHTML, so nothing
   here may use innerHTML. */
function jumpToTs(ts,quiet){state.mode='feed';state.done=false;state.loading=false;state.searching=false;
state.q='';qEl.value='';state.day='';state.anchor=ts;
if(orderEl.value==='newest'){state.before=ts+1;state.after=null;}else{state.after=ts-1;state.before=null;}
clearFeed();loader.style.display='';loadMore();
if(!quiet)writeHash(true);
window.scrollTo(0,0);}
function wordLine(text){var p=document.createElement('p');p.textContent=text;wout.appendChild(p);}
function wordRow(label,e){if(!e)return null;var d=document.createElement('div');d.className='wordex';
var b=document.createElement('b');b.textContent=label+' \\u00b7 '+e.sender+' \\u00b7 '+e.dt;
var body=document.createElement('div');body.textContent=e.content||'(no text)';
var j=document.createElement('button');j.type='button';j.textContent='open in the feed';
j.addEventListener('click',function(){jumpToTs(e.ts);});
d.appendChild(b);d.appendChild(body);d.appendChild(j);return d;}
function renderWord(p){wout.textContent='';
var h=document.createElement('h3');
h.textContent=p.word+' \\u2014 '+p.uses.toLocaleString()+' use'+(p.uses===1?'':'s')+' in '+
p.messages.toLocaleString()+' message'+(p.messages===1?'':'s')+(p.folded?' (spellings counted together)':'');
wout.appendChild(h);
var all=document.createElement('button');all.className='wordall';all.type='button';
all.textContent='Show all '+p.messages.toLocaleString()+' in the feed, oldest first';
all.addEventListener('click',function(){showWordMentions(p.word,p.folded);});
wout.appendChild(all);
var meta=document.createElement('p');meta.className='muted';
meta.textContent='First '+p.first.dt+' ('+p.first.sender+') \\u00b7 last '+p.last.dt+
' \\u00b7 peak '+p.peak_year+(p.reaction_pull?' \\u00b7 pulls '+p.reaction_pull+'x the usual reactions':'')+
' \\u00b7 said on its own '+p.alone_pct+'% of the time';wout.appendChild(meta);
var t=document.createElement('table');p.per_member.forEach(function(r){var tr=t.insertRow();
[r.member,r.uses.toLocaleString()+' \\u00d7',r.per_1k+' per 1k messages'].forEach(function(v){
tr.insertCell().textContent=v;});});wout.appendChild(t);
if(p.variants.length)wordLine('Also spelled: '+p.variants.map(function(x){
return x.word+' ('+x.uses+')';}).join(', '));
if(p.collocations.length)wordLine('Keeps company with: '+p.collocations.map(function(x){
return x.word+' \\u00d7'+x.ratio;}).join(', '));
var caught=p.adoption.filter(function(x){return x.first;});
if(caught.length>1)wordLine('Caught on: '+caught.map(function(x){
return x.member+' (+'+x.days_after+'d)';}).join(' \\u2192 '));
var never=p.adoption.filter(function(x){return !x.first;});
if(never.length)wordLine('Never said it: '+never.map(function(x){return x.member;}).join(', '));
[['First ever',p.examples.first],['Most reacted',p.examples.most_reacted]].concat(
p.examples.random.map(function(e){return ['Somewhere in the middle',e];})).forEach(function(pair){
var row=wordRow(pair[0],pair[1]);if(row)wout.appendChild(row);});}
function wordLookup(){var q=wq.value.trim();if(!q){wout.textContent='';return;}
fetch('/t/'+SLUG+'/api/word?q='+encodeURIComponent(q)+'&variants='+(wfold.checked?'1':'0'))
.then(function(r){
if(r.status===404){wout.textContent='Never said in this chat.';return null;}
if(r.status===503){wout.textContent='Word index disabled (--no-index).';return null;}
return r.json();}).then(function(p){if(p)renderWord(p);})
.catch(function(){wout.textContent='Error looking that up.';});}
/* The address bar carries the reader's state, so the back button undoes a
   search and another page can link straight to a moment or a member. */
function writeHash(push){if(applying)return;
var p=[];
if(state.q)p.push('q='+encodeURIComponent(state.q));
if(state.regex)p.push('re=1');
if(state.member)p.push('member='+encodeURIComponent(state.member));
if(orderEl.value==='oldest')p.push('order=oldest');
if(state.mode==='word'&&state.wordq)p.push('word='+encodeURIComponent(state.wordq));
if(state.mode==='day'&&state.day)p.push('day='+state.day);
if(state.mode==='feed'&&state.anchor!=null)p.push('ts='+state.anchor);
var url=location.pathname+(p.length?'#'+p.join('&'):'');
if(url===location.pathname+location.hash)return;
if(push)history.pushState(null,'',url);else history.replaceState(null,'',url);}
function parseHash(){var out={};
(location.hash||'').replace(/^#/,'').split('&').forEach(function(kv){
if(!kv)return;var i=kv.indexOf('=');if(i<0)return;
out[kv.slice(0,i)]=decodeURIComponent(kv.slice(i+1).replace(/\\+/g,' '));});
return out;}
function applyHash(){var h=parseHash();
applying=true;
state.q=h.q||'';qEl.value=state.q;
state.regex=h.re==='1';reBtn.classList.toggle('on',state.regex);
state.member=h.member||'';memberEl.value=state.member;
orderEl.value=h.order==='oldest'?'oldest':'newest';
applying=false;
if(h.word){wordPanel.classList.add('open');wordBtn.classList.add('on');
wq.value=h.word;state.wordfold=false;
state.mode='word';state.wordq=h.word;state.offset=0;state.done=false;state.loading=false;
clearFeed();document.body.setAttribute('data-wordmode','1');loader.style.display='';
countEl.textContent='Loading mentions of "'+h.word+'"...';loadWordPage();return;}
if(h.day){jumpEl.value=h.day;onThisDay(h.day);return;}
if(h.ts){jumpToTs(parseInt(h.ts,10),true);return;}
reset(state.q?'search':'');}
window.addEventListener('popstate',applyHash);
wq.addEventListener('input',function(){wfold.disabled=wq.value.trim().indexOf(' ')>=0;
clearTimeout(wtimer);wtimer=setTimeout(function(){
fetch('/t/'+SLUG+'/api/suggest?q='+encodeURIComponent(wq.value))
.then(function(r){return r.json();}).then(function(d){wsug.textContent='';
d.words.forEach(function(w){var o=document.createElement('option');o.value=w;wsug.appendChild(o);});
}).catch(function(){});},150);});
wq.addEventListener('change',wordLookup);
wfold.addEventListener('change',wordLookup);
wordBtn.addEventListener('click',function(){var open=wordPanel.classList.toggle('open');
wordBtn.classList.toggle('on',open);if(open)wq.focus();measure();});
moreBtn.addEventListener('click',function(){var open=moreBox.classList.toggle('open');
moreBtn.setAttribute('aria-expanded',open?'true':'false');measure();});
var io=new IntersectionObserver(function(entries){if(entries[0].isIntersecting)loadMore();},{rootMargin:'600px'});
var qtimer=null;
qEl.addEventListener('input',function(){var v=qEl.value.trim();if(state.q===v)return;state.q=v;
reset(v?'search':'');clearTimeout(qtimer);qtimer=setTimeout(function(){writeHash(false);},400);});
memberEl.addEventListener('change',function(){state.member=memberEl.value;
reset(state.q?'search':'');writeHash(true);});
orderEl.addEventListener('change',function(){reset(state.q?'search':'');writeHash(false);});
jumpEl.addEventListener('change',function(){if(!jumpEl.value)return;var d=new Date(jumpEl.value+'T00:00:00');
if(orderEl.value==='newest'){state.before=d.getTime()+864e5;state.after=null;}else{state.after=d.getTime()-864e5;state.before=null;}
state.mode='feed';state.done=false;state.anchor=d.getTime();clearFeed();loadMore();writeHash(true);});
reBtn.addEventListener('click',function(){state.regex=!state.regex;reBtn.classList.toggle('on',state.regex);
if(state.q){reset('search');writeHash(false);}});
odayBtn.addEventListener('click',function(){onThisDay(jumpEl.value||new Date().toISOString().slice(0,10));
writeHash(true);});
surpriseBtn.addEventListener('click',randomMemory);
fetch('/t/'+SLUG+'/api/thread').then(function(r){return r.json();}).then(function(t){
THREAD=t;
memberEl.innerHTML='<option value="">Everyone</option>'+t.members.map(function(m){return '<option value="'+esc(m.name)+'">'+esc(m.name)+' ('+m.count.toLocaleString()+')</option>';}).join('');
document.title=t.title+' | chat-flashback';
applyHash();
io.observe(loader);});
</script></body></html>"""


def run_server(threads, port, output_dir, build_index=True):
    indexed = {}
    for t in threads:
        if build_index:
            print(f"  Indexing {t['title']} for word search...", flush=True)
        indexed[t["slug"]] = ThreadIndex(t["slug"], t["title"], t["thread_dir"],
                                         t.get("msgs"), build_index=build_index,
                                         store=t.get("store"))
    handler = make_handler(indexed, Path(output_dir))
    # Windows honours SO_REUSEADDR by letting a second process bind a port that
    # is already in use, which leaves two readers answering at random and one of
    # them serving whatever the code said when it started. Refuse the port
    # instead, and say so.
    class _Server(ThreadingHTTPServer):
        allow_reuse_address = False

    try:
        server = _Server(("127.0.0.1", port), handler)
    except OSError:
        print(f"[error] port {port} is already in use. Stop the other reader, "
              f"or pass --port with a free one.")
        return 1
    for t in indexed.values():
        print(f"  {t.title}: {t.total:,} messages  ->  "
              f"http://127.0.0.1:{port}/t/{t.slug}/")
    print("  Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopping.")
        server.shutdown()
    return 0
